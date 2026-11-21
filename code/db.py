#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PC 端 SQLite 持久化数据库层（Navicat 可直接连接 code/data/beet_platform.db 管理）。"""
from __future__ import annotations

import base64
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DATA_DIR / "beet_platform.db"
PHOTO_DIR = DATA_DIR / "mp_photos"
_SALT = "beet_stress_platform_2025"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hash_pw(password: str) -> str:
    return hashlib.sha256(f"{_SALT}:{password}".encode("utf-8")).hexdigest()


def get_conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """初始化数据库表结构并写入默认管理员与巡检员账号（仅首次空表时）。"""
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS accounts (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                role_label TEXT NOT NULL DEFAULT '农技巡检员',
                phone TEXT DEFAULT '',
                disabled INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS plants (
                id TEXT PRIMARY KEY,
                code TEXT NOT NULL,
                bed TEXT NOT NULL,
                note TEXT DEFAULT '',
                operator_id TEXT DEFAULT '',
                operator_name TEXT DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY,
                plant_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                level INTEGER NOT NULL DEFAULT 0,
                level_name TEXT DEFAULT '',
                note TEXT DEFAULT '',
                photo_path TEXT DEFAULT '',
                canopy_pct REAL,
                yellow_pct REAL,
                exg_mean REAL,
                vari_mean REAL,
                operator_id TEXT DEFAULT '',
                operator_name TEXT DEFAULT '',
                reviewed INTEGER NOT NULL DEFAULT 0,
                expert_level INTEGER,
                expert_note TEXT DEFAULT '',
                reviewed_by TEXT DEFAULT '',
                reviewed_at TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS actions (
                id TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
                bed TEXT DEFAULT '全棚',
                type TEXT NOT NULL,
                note TEXT DEFAULT '',
                operator_id TEXT DEFAULT '',
                operator_name TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                detail TEXT DEFAULT ''
            );
            """
        )
        cnt = conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
        if cnt == 0:
            now = _now_iso()
            conn.execute(
                "INSERT INTO accounts (id, username, name, password_hash, role, role_label, phone, disabled, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)",
                ("acc_admin_01", "admin", "李导师", _hash_pw("123456"), "admin", "科研管理员", "139****8801", now),
            )
            conn.execute(
                "INSERT INTO accounts (id, username, name, password_hash, role, role_label, phone, disabled, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)",
                ("acc_inspector_01", "inspector", "张农技", _hash_pw("123456"), "user", "农技巡检员", "138****6012", now),
            )
            conn.execute(
                "INSERT INTO audit_logs (ts, actor, action, detail) VALUES (?, ?, ?, ?)",
                (now, "system", "INIT_DB", "初始化 SQLite 数据库与默认账号 admin / inspector"),
            )


def log_audit(actor: str, action: str, detail: str = "") -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO audit_logs (ts, actor, action, detail) VALUES (?, ?, ?, ?)",
            (_now_iso(), actor, action, detail),
        )


def authenticate(username: str, password: str) -> tuple[dict[str, Any] | None, str]:
    uname = username.strip()
    if not uname or not password:
        return None, "请输入账号和密码"
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM accounts WHERE username = ?", (uname,)).fetchone()
    if row is None:
        return None, "账号不存在"
    if row["password_hash"] != _hash_pw(password):
        return None, "密码错误"
    if row["disabled"]:
        return None, "该账号已被管理员停用"
    user = dict(row)
    log_audit(user["username"], "LOGIN", f"{user['name']} ({user['role_label']}) 登录系统")
    return user, ""


def register_user(username: str, name: str, password: str, phone: str = "") -> tuple[bool, str]:
    uname = username.strip()
    rname = name.strip()
    if not uname or not rname or len(password) < 4:
        return False, "账号、姓名不能为空且密码至少 4 位"
    acc_id = f"acc_u_{int(datetime.now(timezone.utc).timestamp())}"
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO accounts (id, username, name, password_hash, role, role_label, phone, disabled, created_at) "
                "VALUES (?, ?, ?, ?, 'user', '农技巡检员', ?, 0, ?)",
                (acc_id, uname, rname, _hash_pw(password), phone.strip(), _now_iso()),
            )
        log_audit(uname, "REGISTER", f"新用户自行注册：{rname}")
        return True, "注册成功，请使用新账号登录"
    except sqlite3.IntegrityError:
        return False, "该登录账号已存在，请换一个用户名"


def list_accounts() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM accounts ORDER BY created_at ASC").fetchall()
    return [dict(r) for r in rows]


def create_account(
    username: str,
    name: str,
    role: str,
    role_label: str,
    password: str = "123456",
    phone: str = "",
    actor: str = "admin",
) -> tuple[bool, str]:
    uname = username.strip()
    rname = name.strip()
    if not uname or not rname:
        return False, "登录账号和姓名不能为空"
    acc_id = f"acc_pc_{int(datetime.now(timezone.utc).timestamp())}"
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO accounts (id, username, name, password_hash, role, role_label, phone, disabled, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)",
                (acc_id, uname, rname, _hash_pw(password or "123456"), role, role_label, phone.strip(), _now_iso()),
            )
        log_audit(actor, "CREATE_ACCOUNT", f"新增账号 {uname} ({rname} / {role_label})")
        return True, f"已新增账号 {uname}"
    except sqlite3.IntegrityError:
        return False, f"账号 {uname} 已存在"


def update_account(
    acc_id: str,
    role: str,
    role_label: str,
    disabled: bool,
    new_password: str = "",
    actor: str = "admin",
) -> None:
    with get_conn() as conn:
        if new_password.strip():
            conn.execute(
                "UPDATE accounts SET role = ?, role_label = ?, disabled = ?, password_hash = ? WHERE id = ?",
                (role, role_label, 1 if disabled else 0, _hash_pw(new_password.strip()), acc_id),
            )
        else:
            conn.execute(
                "UPDATE accounts SET role = ?, role_label = ?, disabled = ? WHERE id = ?",
                (role, role_label, 1 if disabled else 0, acc_id),
            )
    log_audit(actor, "UPDATE_ACCOUNT", f"更新账号 {acc_id}: role={role_label}, disabled={disabled}")


def _save_base64_photo(obs_id: str, b64_str: str, ext: str = "jpg") -> str:
    if not b64_str:
        return ""
    safe_ext = ext if ext.isalnum() else "jpg"
    out_file = PHOTO_DIR / f"{obs_id}.{safe_ext}"
    try:
        out_file.write_bytes(base64.b64decode(b64_str))
        return str(out_file)
    except Exception:
        return ""


def ingest_miniprogram_payload(payload: dict[str, Any], actor: str = "admin") -> dict[str, int]:
    """将小程序导出的 JSON 数据包合并入库 SQLite，并解码保存内嵌照片。"""
    accounts = payload.get("accounts") or []
    plants = payload.get("plants") or []
    observations = payload.get("observations") or []
    actions = payload.get("actions") or []
    now = _now_iso()

    cnt_acc = 0
    cnt_plant = 0
    cnt_obs = 0
    cnt_act = 0

    with get_conn() as conn:
        for a in accounts:
            aid = a.get("id")
            if not aid:
                continue
            rlabel = a.get("role") or "农技巡检员"
            sys_role = "admin" if "管理员" in rlabel else "user"
            uname = a.get("username") or aid
            ex = conn.execute("SELECT id FROM accounts WHERE id = ?", (aid,)).fetchone()
            if ex is None:
                # 避免 username 冲突
                u_ex = conn.execute("SELECT id FROM accounts WHERE username = ?", (uname,)).fetchone()
                if u_ex is not None:
                    uname = f"{uname}_{aid[-4:]}"
                conn.execute(
                    "INSERT INTO accounts (id, username, name, password_hash, role, role_label, phone, disabled, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        aid,
                        uname,
                        a.get("name") or "巡检员",
                        _hash_pw("123456"),
                        sys_role,
                        rlabel,
                        a.get("phone") or "",
                        1 if a.get("disabled") else 0,
                        now,
                    ),
                )
                cnt_acc += 1

        for p in plants:
            pid = p.get("id")
            if not pid:
                continue
            op = p.get("operator") or {}
            conn.execute(
                "INSERT INTO plants (id, code, bed, note, operator_id, operator_name, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET code=excluded.code, bed=excluded.bed, note=excluded.note",
                (
                    pid,
                    str(p.get("code", "")),
                    str(p.get("bed", "")),
                    p.get("note", ""),
                    op.get("id", ""),
                    op.get("name", "本地巡检员"),
                    p.get("createdAt") or now,
                ),
            )
            cnt_plant += 1

        for o in observations:
            oid = o.get("id")
            if not oid:
                continue
            op = o.get("operator") or {}
            metrics = o.get("metrics") or {}
            photo_path = _save_base64_photo(oid, o.get("imageBase64", ""), o.get("imageExt", "jpg"))
            conn.execute(
                "INSERT INTO observations (id, plant_id, ts, level, level_name, note, photo_path, "
                "canopy_pct, yellow_pct, exg_mean, vari_mean, operator_id, operator_name) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "photo_path=CASE WHEN excluded.photo_path != '' THEN excluded.photo_path ELSE observations.photo_path END, "
                "canopy_pct=COALESCE(excluded.canopy_pct, observations.canopy_pct), "
                "yellow_pct=COALESCE(excluded.yellow_pct, observations.yellow_pct), "
                "exg_mean=COALESCE(excluded.exg_mean, observations.exg_mean), "
                "vari_mean=COALESCE(excluded.vari_mean, observations.vari_mean)",
                (
                    oid,
                    o.get("plantId", ""),
                    o.get("ts", now),
                    int(o.get("level") or 0),
                    o.get("levelName", ""),
                    o.get("note", ""),
                    photo_path,
                    metrics.get("canopyPct"),
                    metrics.get("yellowPct"),
                    metrics.get("exgMean"),
                    metrics.get("variMean"),
                    op.get("id", ""),
                    op.get("name", "本地巡检员"),
                ),
            )
            cnt_obs += 1

        for act in actions:
            act_id = act.get("id")
            if not act_id:
                continue
            op = act.get("operator") or {}
            conn.execute(
                "INSERT OR IGNORE INTO actions (id, ts, bed, type, note, operator_id, operator_name) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    act_id,
                    act.get("ts", now),
                    act.get("bed", "全棚"),
                    act.get("type", "农事处置"),
                    act.get("note", ""),
                    op.get("id", ""),
                    op.get("name", "本地巡检员"),
                ),
            )
            cnt_act += 1

    log_audit(actor, "INGEST_JSON", f"入库 {cnt_acc} 账号, {cnt_plant} 株位, {cnt_obs} 观测, {cnt_act} 台账")
    return {"accounts": cnt_acc, "plants": cnt_plant, "observations": cnt_obs, "actions": cnt_act}


def list_plants() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM plants ORDER BY bed, code").fetchall()
    return [dict(r) for r in rows]


def list_observations(operator_name: str | None = None) -> list[dict[str, Any]]:
    sql = (
        "SELECT o.*, p.bed AS plant_bed, p.code AS plant_code "
        "FROM observations o LEFT JOIN plants p ON o.plant_id = p.id "
    )
    params: tuple[Any, ...] = ()
    if operator_name:
        sql += "WHERE o.operator_name = ? "
        params = (operator_name,)
    sql += "ORDER BY o.ts DESC"
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def review_observation(obs_id: str, expert_level: int, expert_note: str, reviewer_name: str) -> None:
    now = _now_iso()
    with get_conn() as conn:
        conn.execute(
            "UPDATE observations SET reviewed = 1, expert_level = ?, level = ?, expert_note = ?, "
            "reviewed_by = ?, reviewed_at = ? WHERE id = ?",
            (expert_level, expert_level, expert_note.strip(), reviewer_name, now, obs_id),
        )
    log_audit(reviewer_name, "REVIEW_OBS", f"复核观测 {obs_id} -> L{expert_level}")


def list_actions(operator_name: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM actions "
    params: tuple[Any, ...] = ()
    if operator_name:
        sql += "WHERE operator_name = ? "
        params = (operator_name,)
    sql += "ORDER BY ts DESC"
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def list_audit_logs(limit: int = 80) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def export_sync_payload() -> dict[str, Any]:
    """从 SQLite 数据库导出标准同步包供小程序「备份恢复」导入。"""
    accounts = [
        {
            "id": a["id"],
            "username": a["username"],
            "name": a["name"],
            "role": a["role_label"],
            "phone": a["phone"],
            "disabled": bool(a["disabled"]),
        }
        for a in list_accounts()
    ]
    plants = [
        {
            "id": p["id"],
            "code": p["code"],
            "bed": p["bed"],
            "note": p["note"],
            "createdAt": p["created_at"],
            "operator": {"id": p["operator_id"], "name": p["operator_name"]},
        }
        for p in list_plants()
    ]
    observations = []
    for o in list_observations():
        row: dict[str, Any] = {
            "id": o["id"],
            "plantId": o["plant_id"],
            "ts": o["ts"],
            "level": o["level"],
            "levelName": o["level_name"],
            "note": o["note"],
            "metrics": {
                "canopyPct": o["canopy_pct"],
                "yellowPct": o["yellow_pct"],
                "exgMean": o["exg_mean"],
                "variMean": o["vari_mean"],
            }
            if o["canopy_pct"] is not None
            else None,
            "operator": {"id": o["operator_id"], "name": o["operator_name"]},
        }
        if o["reviewed"]:
            row["reviewed"] = True
            row["expertLevel"] = o["expert_level"]
            row["expertNote"] = o["expert_note"]
            row["reviewedBy"] = o["reviewed_by"]
            row["reviewedAt"] = o["reviewed_at"]
        observations.append(row)
    actions = [
        {
            "id": a["id"],
            "ts": a["ts"],
            "bed": a["bed"],
            "type": a["type"],
            "note": a["note"],
            "operator": {"id": a["operator_id"], "name": a["operator_name"]},
        }
        for a in list_actions()
    ]
    return {
        "app": "beet-warning-miniprogram",
        "version": 3,
        "kind": "pc-db-sync",
        "exportedAt": _now_iso(),
        "accounts": accounts,
        "plants": plants,
        "observations": observations,
        "actions": actions,
    }
