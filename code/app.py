#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ADA2025 甜菜缺水 · 3D 数字孪生与实证审计工作台（桌面端）。"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import tifffile

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
V8_DIR = DATA_DIR / "v8_exp"
V9_DIR = DATA_DIR / "v9_exp"
FIG_DIR = PROJECT_ROOT / "output" / "v8_figures"
DASH_DIR = PROJECT_ROOT / "mobile" / "dashboard"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import db  # noqa: E402
import importlib  # noqa: E402
import ui_theme  # noqa: E402
import bigscreen  # noqa: E402

importlib.reload(db)
importlib.reload(bigscreen)
importlib.reload(ui_theme)  # 热重载：避免 Streamlit 缓存旧版模块导致属性缺失
from did_analysis import WINDOW_END, WINDOW_START, build_pairs  # noqa: E402
from exp_v9_early_warning_and_hotspot import run_causal_cusum  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(
    page_title="设施甜菜水分胁迫监测平台",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

GRADE_RULES = [
    (0, "正常 (Normal)", "🟢", "维持常规对照观测。"),
    (1, "轻度胁迫 (Mild)", "🟡", "加密热红外观测与趋势复核。"),
    (2, "中度胁迫 (Moderate)", "🟠", "纳入优先复水观察序列。"),
    (3, "重度胁迫 (Severe)", "🔴", "列为重点复水与热红外复核对象。"),
]


def classify_residual(val: float) -> tuple[int, str, str, str]:
    """基于跨床配对热红外残差（℃）的四级工程判级。"""
    if not np.isfinite(val) or val < 1.0:
        return GRADE_RULES[0]
    if val < 2.5:
        return GRADE_RULES[1]
    if val < 4.5:
        return GRADE_RULES[2]
    return GRADE_RULES[3]


def build_audit_report_html(sum_df: pd.DataFrame, feature: str) -> str:
    """生成标准化可打印 A4 温室水分胁迫审计与复水核验报告（HTML）。"""
    rows_html = []
    for _, r in sum_df.sort_values("窗末3次均值(℃)", ascending=False).iterrows():
        rows_html.append(
            f"<tr><td>{r['处理单元']}</td><td>{r['床位']}</td>"
            f"<td>{r['pre均值(℃)']:+.3f}</td><td>{r['in均值(℃)']:+.3f}</td>"
            f"<td><b>{r['窗末3次均值(℃)']:+.3f}</b></td><td>{r['工程判级']}</td>"
            f"<td>{r['实证观测建议']}</td></tr>"
        )
    severe_n = int((sum_df["等级"] >= 2).sum())
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>ADA2025 温室水分胁迫实证审计报告</title>
<style>
  body{{font-family:"Microsoft YaHei",sans-serif;color:#1e293b;margin:28px auto;max-width:960px;line-height:1.6}}
  h1{{color:#0e3f5c;border-bottom:2px solid #16794b;padding-bottom:8px;font-size:22px}}
  .meta{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0}}
  .card{{background:#f4f8fb;border:1px solid #d5e2ec;border-radius:6px;padding:10px 12px}}
  .card span{{font-size:12px;color:#64748b;display:block}} .card b{{font-size:17px;color:#0f4c81}}
  table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:12px}}
  th,td{{border:1px solid #cbd5e1;padding:7px 9px;text-align:center}}
  th{{background:#0f4c81;color:#fff}}
  .note{{margin-top:16px;background:#fef3c7;border-left:4px solid #d97706;padding:10px 14px;font-size:12.5px}}
</style></head><body>
<h1>ADA2025 温室甜菜水分胁迫实证审计与复水核验台账</h1>
<div class="meta">
  <div class="card"><span>审计指标</span><b>{feature}</b></div>
  <div class="card"><span>有效配对单元</span><b>{len(sum_df)} 组</b></div>
  <div class="card"><span>中重度预警</span><b>{severe_n} 单元</b></div>
  <div class="card"><span>复水期回落</span><b>79.9% (CUSUM 中位 73h)</b></div>
</div>
<table><thead><tr><th>处理单元</th><th>床位</th><th>断水前均值(℃)</th><th>断水期均值(℃)</th><th>窗末3次均值(℃)</th><th>工程判级</th><th>实证观测建议</th></tr></thead>
<tbody>{"".join(rows_html)}</tbody></table>
<div class="note"><b>科学边界声明：</b>本台账基于跨床同期配对残差与在线 CUSUM 审计生成，不输出黑盒未来预测概率；四级工程阈值仅供设施农业巡检与复水核验参考。浏览器按 Ctrl+P 可直接打印或另存为 PDF。</div>
</body></html>"""


@st.cache_data(show_spinner=False)
def load_v8_table() -> pd.DataFrame:
    """加载 v8 特征表（含 Otsu 纯冠层特征），兜底 v7。"""
    p_v8 = V8_DIR / "features_v8_with_canopy.csv"
    p_v7 = DATA_DIR / "features_multimodal_v7.csv"
    df = pd.read_csv(p_v8 if p_v8.exists() else p_v7)
    df["code"] = df["plant_code"]
    df["slot"] = df["slot"].astype(str)
    df["dt"] = pd.to_datetime(
        df["date"].str.replace("_", "-") + " " + df["slot"].str.zfill(2) + ":00"
    )
    df["phase"] = np.where(
        df["dt"] < WINDOW_START, "pre",
        np.where(df["dt"] <= WINDOW_END, "in", "post")
    )
    return df


@st.cache_data(show_spinner=False)
def load_physics_table() -> pd.DataFrame | None:
    p = V8_DIR / "features_v8_physics.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["dt"] = pd.to_datetime(
        df["date"].str.replace("_", "-") + " " + df["slot"].astype(str).str.zfill(2) + ":00"
    )
    return df


@st.cache_data(show_spinner=False)
def build_embedded_3d_html() -> str | None:
    """复用 mobile/dashboard 的离线 Three.js 高精度 Venlo 温室引擎，构建桌面端自包含组件。"""
    three_js = DASH_DIR / "vendor" / "three.min.js"
    ctrl_js = DASH_DIR / "vendor" / "OrbitControls.js"
    rgbe_js = DASH_DIR / "vendor" / "RGBELoader.js"
    data_js = DASH_DIR / "data.js"
    gh_js = DASH_DIR / "greenhouse3d.js"
    if not (three_js.exists() and ctrl_js.exists() and data_js.exists() and gh_js.exists()):
        return None
    rgbe_tag = f"<script>{rgbe_js.read_text(encoding='utf-8')}</script>" if rgbe_js.exists() else ""

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<style>
  html,body{{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#04101a;font-family:"Microsoft YaHei",sans-serif;}}
  #bar{{position:absolute;top:10px;left:12px;right:12px;z-index:15;display:flex;justify-content:space-between;align-items:center;pointer-events:none;}}
  #bar>*{{pointer-events:auto;}}
  .pill-grp{{display:flex;gap:6px;background:rgba(4,16,26,.84);padding:5px 8px;border-radius:8px;border:1px solid rgba(53,224,192,.35);backdrop-filter:blur(8px);}}
  .tool,.cam-btn,.rx-btn{{border:1px solid rgba(53,224,192,.35);background:rgba(255,255,255,.06);color:#cfe8ff;border-radius:5px;padding:4px 9px;font-size:11.5px;cursor:pointer;white-space:nowrap;}}
  .tool.on,.cam-btn.on,.rx-btn.on{{background:#35e0c0;color:#04101a;font-weight:700;border-color:#35e0c0;}}
  .rx-panel{{display:none;position:absolute;top:52px;right:12px;width:295px;background:rgba(4,16,26,.94);border:1px solid rgba(74,158,255,.55);border-radius:10px;padding:11px 13px;color:#e6fffb;z-index:20;font-size:12px;line-height:1.5;}}
  .rx-panel.on{{display:block;}}
  .rx-hd{{display:flex;justify-content:space-between;font-weight:700;color:#7ce8ff;margin-bottom:6px;}}
  .rx-item{{background:rgba(255,255,255,.05);border-left:3px solid #ffb020;padding:5px 8px;margin-bottom:6px;border-radius:4px;}}
  .rx-item.sev{{border-left-color:#ff4d67;}} .rx-item.ok{{border-left-color:#3ddc84;}}
  .rx-actions{{display:flex;gap:8px;margin-top:8px;}}
  .rx-act{{flex:1;padding:6px;border-radius:5px;border:none;font-weight:700;cursor:pointer;font-size:11.5px;}}
  .rx-act.go{{background:#35e0c0;color:#04101a;}} .rx-act.rst{{background:rgba(255,255,255,.12);color:#e6fffb;}}
  #legend-bar{{position:absolute;bottom:10px;left:12px;z-index:12;display:flex;gap:12px;background:rgba(4,16,26,.84);padding:5px 12px;border-radius:8px;border:1px solid rgba(53,224,192,.3);font-size:11px;color:#d6f5f0;}}
  .dotc{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;}}
  .cb-wrap{{display:none;position:absolute;right:12px;bottom:10px;z-index:12;align-items:center;gap:8px;padding:5px 10px;border-radius:8px;background:rgba(4,16,26,.86);border:1px solid rgba(53,224,192,.35);font-size:10.5px;color:#d6f5ef;}}
  .cb-bar{{width:120px;height:8px;border-radius:4px;background:linear-gradient(90deg,#2563eb 0%,#06b6d4 28%,#eab308 58%,#f97316 80%,#ef4444 100%);}}
</style></head>
<body>
<div id="bar">
  <div class="pill-grp">
    <button class="tool on" data-mode="grade">胁迫等级</button>
    <button class="tool" data-mode="resid">残差温度</button>
    <button class="cam-btn" data-theme="real" id="btn-theme">日光物理</button>
    <button class="cam-btn" id="btn-roof" type="button">掀顶透视</button>
    <button class="cam-btn" id="btn-iso">全景</button>
    <button class="cam-btn" id="btn-top">俯视</button>
    <button class="rx-btn" id="btn-rx" type="button">💧 复水推演</button>
  </div>
</div>
<div id="scene-wrap" style="position:relative;width:100%;height:100%;background:linear-gradient(180deg,#cfe4f2 0%,#e5f0f5 62%,#b9c6cc 100%);">
  <canvas id="scene3d" style="width:100%;height:100%;display:block;cursor:grab;"></canvas>
  <div id="labels3d" style="position:absolute;inset:0;pointer-events:none;overflow:hidden;"></div>
  <div id="tip3d" style="display:none;position:absolute;pointer-events:none;background:rgba(4,16,26,.94);border:1px solid rgba(53,224,192,.6);border-radius:8px;padding:8px 11px;font-size:12px;color:#e6fffb;z-index:19;line-height:1.5;"></div>
  <div class="cb-wrap" id="thermal-colorbar"><span>0℃</span><div class="cb-bar"></div><span>+8℃ 热红外 ΔT</span></div>
  <div class="rx-panel" id="rx-panel">
    <div class="rx-hd"><span>💧 复水响应推演（实证回落演示）</span><span style="cursor:pointer" id="rx-close">✕</span></div>
    <div id="rx-body"></div>
    <div class="rx-actions">
      <button class="rx-act go" id="rx-irrigate" type="button">模拟潮汐补水</button>
      <button class="rx-act rst" id="rx-reset" type="button">恢复快照</button>
    </div>
  </div>
</div>
<div id="legend-bar">
  <span><i class="dotc" style="background:#3ddc84"></i>0 健康</span>
  <span><i class="dotc" style="background:#ffb020"></i>1 轻度</span>
  <span><i class="dotc" style="background:#ff7a45"></i>2 中度</span>
  <span><i class="dotc" style="background:#ff4d67"></i>3 重度</span>
  <span style="color:#4a9eff"><i class="dotc" style="background:#4a9eff"></i>W1 对照</span>
</div>
<div id="rows" style="display:none"></div>
<script>{data_js.read_text(encoding="utf-8")}</script>
<script>{three_js.read_text(encoding="utf-8")}</script>
<script>{ctrl_js.read_text(encoding="utf-8")}</script>
{rgbe_tag}
<script>{gh_js.read_text(encoding="utf-8")}</script>
<script>
  window.initGreenhouse3D({{
    data: window.DASH_DATA,
    LV: ['#3ddc84','#ffb020','#ff7a45','#ff4d67'],
    LVN: ['健康','早期轻度','中度','重度']
  }});
</script></body></html>"""


def otsu_segment_thermal(arr: np.ndarray) -> tuple[np.ndarray, float, dict]:
    """对单帧热红外数组执行 Otsu 低温冠层分离，返回掩膜、阈值及统计量。"""
    vmin, vmax = np.percentile(arr, [1, 99])
    if vmax <= vmin:
        mask = np.ones_like(arr, dtype=bool)
        return mask, float(arr.mean()), {"canopy_frac": 1.0, "canopy_p25": float(np.percentile(arr, 25)), "bg_gap": 0.0}
    u8 = np.clip((arr - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
    thr_u8, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(vmin + (thr_u8 / 255.0) * (vmax - vmin))
    low = arr <= thr
    c_vals = arr[low] if low.any() else arr.ravel()
    bg_vals = arr[~low] if (~low).any() else arr.ravel()
    stats = {
        "canopy_frac": round(float(low.mean()), 4),
        "canopy_mean": round(float(c_vals.mean()), 3),
        "canopy_p25": round(float(np.percentile(c_vals, 25)), 3),
        "raw_p25": round(float(np.percentile(arr, 25)), 3),
        "raw_p90": round(float(np.percentile(arr, 90)), 3),
        "bg_gap": round(float(bg_vals.mean() - c_vals.mean()), 3),
    }
    return low, thr, stats


def main() -> None:
    ui_theme.inject()
    st.markdown("""
    <div class="hero-hdr">
      <div><h2>设施甜菜水分胁迫数字孪生监测平台</h2></div>
    </div>
    """, unsafe_allow_html=True)

    db.init_db()
    pc_user = st.session_state.get("pc_user")
    if pc_user is None:
        left_p, right_p = st.columns([1.15, 1.0], gap="large")
        with left_p:
            st.markdown("""
            <div class="portal-box">
              <h3>设施甜菜水分胁迫智能监控中心</h3>
              <div class="portal-sub">
                集成温室 3D 数字孪生、RGB 冠层 Otsu 分割、热红外差分判级与 CUSUM 早期预警，
                支持移动巡检端与桌面管理平台数据入库、专家复核及权限分级管控。
              </div>
              <div class="portal-stat">
                <div class="k">监测设施与规模</div>
                <div class="v">ADA2025 试验温室 · 33 株位 · 22 个跨床配对单元</div>
              </div>
              <div class="portal-stat">
                <div class="k">核心算法引擎</div>
                <div class="v">Otsu 冠层去背景 · 热红外因果差分 · 在线 CUSUM 预警</div>
              </div>
              <div class="portal-stat">
                <div class="k">数据持久化服务</div>
                <div class="v">本地 SQLite 关系数据库 · 支持 Navicat 实时管理</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with right_p:
            auth_tab1, auth_tab2 = st.tabs(["账号登录", "新用户注册"])
            with auth_tab1:
                u_in = st.text_input("登录账号", placeholder="请输入账号", key="login_u")
                p_in = st.text_input("登录密码", type="password", placeholder="请输入密码", key="login_p")
                b1, b2 = st.columns(2)
                if b1.button("登录工作台", type="primary", use_container_width=True):
                    user, err = db.authenticate(u_in, p_in)
                    if user is None:
                        st.error(err)
                    else:
                        st.session_state["pc_user"] = user
                        st.rerun()
                if b2.button("只读访客浏览", use_container_width=True):
                    st.session_state["pc_user"] = {
                        "id": "guest",
                        "username": "guest",
                        "name": "访客用户",
                        "role": "guest",
                        "role_label": "只读访客",
                    }
                    st.rerun()
                with st.expander("初始账户说明", expanded=False):
                    st.caption("系统内置管理员 `admin`（密码 `123456`）与普通巡检员 `inspector`（密码 `123456`），登录后可在后台修改密码或新增团队成员。")

            with auth_tab2:
                ru = st.text_input("登录账号", placeholder="如：wang01", key="reg_u")
                rn = st.text_input("姓名", placeholder="如：王农艺", key="reg_n")
                rp = st.text_input("密码（至少 4 位）", type="password", key="reg_p")
                rph = st.text_input("联系电话（可选）", placeholder="138****0000", key="reg_ph")
                if st.button("完成注册", type="primary", use_container_width=True):
                    ok, msg = db.register_user(ru, rn, rp, rph)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
        return

    df = load_v8_table()

    with st.sidebar:
        st.markdown(f"""
        <div class="user-card">
          <div class="u-name">{pc_user.get('name')} <span style="font-size:0.8rem;color:#64748B;font-weight:500;">({pc_user.get('username')})</span></div>
          <div class="u-role">{pc_user.get('role_label')}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("切换 / 退出账号", use_container_width=True):
            st.session_state.pop("pc_user", None)
            st.rerun()
        st.markdown("---")
        big_on = st.toggle("📺 大屏展示模式", key="bigscreen_on", help="切换为深蓝科技风大屏，指标区每 10 秒自动刷新")
        st.markdown("---")
        st.header("监测参数")
        feat_candidates = [c for c in ("tm_canopy_p25", "tm_p05", "tm_p25", "tm_canopy_mean") if c in df.columns]
        feature = st.selectbox("热红外指标", feat_candidates, index=0)
        daytime_only = st.checkbox("仅白天高蒸腾时段 (10–16h)", value=False)
        st.markdown("---")
        st.metric("对称安慰剂越线", "0 / 55")
        st.metric("CUSUM 检出", "22 / 22")
        st.metric("复水回落", "79.9%")

    work_df = df[df["slot"].isin({"10", "12", "13", "15", "16"})].copy() if daytime_only else df
    pairs = build_pairs(work_df, feature).sort_values(["plant", "dt"])

    if big_on:
        bigscreen.render(
            pairs, classify_residual, build_embedded_3d_html(),
            (WINDOW_START, WINDOW_END), ui_theme.risk_list_html,
        )
        return

    is_admin = pc_user.get("role") == "admin"
    tab_titles = [
        "🏠 监测总览",
        "🧪 数据处理中心",
        "🌿 温室全景孪生",
        "🔬 冠层视觉分割",
        "📊 因果反证审计",
        "🌡️ 微气候耦合",
        "🚨 早期胁迫预警",
        "👤 巡检工作台",
    ]
    if is_admin:
        tab_titles.append("⚙️ 后台与数据中心")
    tabs = st.tabs(tab_titles)
    tab0, tab_dp, tab1, tab2, tab3, tab4, tab5, tab6 = tabs[:8]

    # ---------- Tab 0: 监测总览驾驶舱 ----------
    with tab0:
        ui_theme.page_head("🏠", "监测总览", f"欢迎回来，{pc_user.get('name')}。此处汇总全棚判级、断水响应趋势与团队数据状态。")
        ov_rows = []
        for p_id, g in pairs.groupby("plant"):
            g_in = g[g["phase"] == "in"]
            l3 = float(g_in["resid"].tail(3).mean()) if not g_in.empty else float("nan")
            lvl, g_name, _, _ = classify_residual(l3)
            ov_rows.append({"unit": p_id, "bed": p_id.split("_")[0], "l3": l3, "lvl": lvl, "name": g_name.split(" ")[0]})
        ov_df = pd.DataFrame(ov_rows)
        n_units = len(ov_df)
        n_warn = int((ov_df["lvl"] >= 2).sum()) if n_units else 0
        mean_l3 = float(ov_df["l3"].mean()) if n_units else 0.0
        db_obs = len(db.list_observations())
        ui_theme.kpi_cards([
            {"label": "监测配对单元", "value": f"{n_units}", "sub": "W2/W3 缺水床 vs W1 对照床", "icon": "🌱", "tone": "g1"},
            {"label": "中重度预警单元", "value": f"{n_warn}", "sub": f"窗末残差 ≥ 2.5℃ · 占比 {n_warn / max(n_units, 1) * 100:.0f}%", "icon": "🚨", "tone": "g2"},
            {"label": "窗末平均升温", "value": f"{mean_l3:+.2f}℃", "sub": f"热红外指标 {feature}", "icon": "🌡️", "tone": "g3"},
            {"label": "平台已入库巡检", "value": f"{db_obs}", "sub": "手机端同步 · SQLite 持久化", "icon": "📱", "tone": "g4"},
        ])

        r1c1, r1c2 = st.columns([1.0, 1.9], gap="medium")
        with r1c1:
            ui_theme.section("四级判级构成")
            counts = [int((ov_df["lvl"] == k).sum()) for k in range(4)] if n_units else [0, 0, 0, 0]
            fig_d, ax_d = plt.subplots(figsize=(4.2, 3.6), dpi=140)
            cols_d = ["#22C55E", "#F59E0B", "#F97316", "#EF4444"]
            labels_d = ["正常", "轻度", "中度", "重度"]
            nz = [(c, l, col) for c, l, col in zip(counts, labels_d, cols_d) if c > 0]
            if nz:
                ax_d.pie([x[0] for x in nz], labels=[f"{x[1]} {x[0]}" for x in nz], colors=[x[2] for x in nz],
                         startangle=90, counterclock=False, wedgeprops={"width": 0.42, "edgecolor": "white"},
                         textprops={"fontsize": 9})
                ax_d.text(0, 0, f"{n_units}\n单元", ha="center", va="center", fontsize=13, fontweight="bold", color="#0F172A")
            fig_d.tight_layout()
            st.pyplot(fig_d)
            plt.close(fig_d)
        with r1c2:
            ui_theme.section("断水—复水全程趋势（按床日均残差）")
            tr = pairs.assign(day=pairs["dt"].dt.normalize(), bed=pairs["plant"].str.split("_").str[0])
            tr = tr.groupby(["day", "bed"])["resid"].mean().unstack()
            fig_t, ax_t = plt.subplots(figsize=(8.2, 3.6), dpi=140)
            bed_col = {"W2": "#F97316", "W3": "#DC2626"}
            for bed in tr.columns:
                ax_t.plot(tr.index, tr[bed], linewidth=2.0, marker="o", markersize=3, color=bed_col.get(bed, "#0F766E"), label=f"{bed} 床")
            ax_t.axvspan(WINDOW_START, WINDOW_END, color="#FEE2E2", alpha=0.55, label="断水窗")
            ax_t.axhline(0, color="#94A3B8", linestyle="--", linewidth=0.9)
            ax_t.set_ylabel("配对残差 (℃)")
            ax_t.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")
            ax_t.grid(True, linestyle="--", alpha=0.35)
            fig_t.autofmt_xdate()
            fig_t.tight_layout()
            st.pyplot(fig_t)
            plt.close(fig_t)

        r2c1, r2c2 = st.columns([1.0, 1.3], gap="medium")
        with r2c1:
            top = ov_df.sort_values("l3", ascending=False).head(6) if n_units else ov_df
            vmax = max(float(top["l3"].max()), 1.0) if n_units else 1.0
            pal = {0: "#22C55E", 1: "#F59E0B", 2: "#F97316", 3: "#EF4444"}
            risk_rows = [{"name": r["unit"], "val": r["l3"], "pct": max(4.0, r["l3"] / vmax * 100), "color": pal[int(r["lvl"])],
                          "grade": int(r["lvl"]), "tag": r["name"]} for _, r in top.iterrows()]
            ui_theme.panel("高风险单元榜 Top 6", ui_theme.risk_list_html(risk_rows), "窗末 3 次残差均值")
        with r2c2:
            ui_theme.section("单元 × 日期 残差热力图")
            heat = pairs.assign(day=pairs["dt"].dt.strftime("%m-%d")).groupby(["plant", "day"])["resid"].mean().unstack()
            heat = heat.loc[sorted(heat.index, key=lambda s: (s.split("_")[0], s))]
            fig_h, ax_h = plt.subplots(figsize=(8.2, 3.9), dpi=140)
            im = ax_h.imshow(heat.values, aspect="auto", cmap="RdYlGn_r", vmin=-1.5, vmax=6.0)
            ax_h.set_yticks(range(len(heat.index)))
            ax_h.set_yticklabels(heat.index, fontsize=6.5)
            xt = list(range(0, len(heat.columns), max(1, len(heat.columns) // 10)))
            ax_h.set_xticks(xt)
            ax_h.set_xticklabels([heat.columns[i] for i in xt], fontsize=7, rotation=0)
            for sp in ax_h.spines.values():
                sp.set_visible(False)
            cb = fig_h.colorbar(im, ax=ax_h, fraction=0.025, pad=0.01)
            cb.ax.tick_params(labelsize=7)
            fig_h.tight_layout()
            st.pyplot(fig_h)
            plt.close(fig_h)

        r3c1, r3c2 = st.columns([1.5, 1.0], gap="medium")
        with r3c1:
            ui_theme.panel("平台能力一览", """
              <div class="feat-grid">
                <div class="feat"><div class="f-i">🌿</div><div class="f-t">3D 数字孪生</div><div class="f-d">Venlo 温室实时判级着色，点选株位查看处方。</div></div>
                <div class="feat"><div class="f-i">🔬</div><div class="f-t">Otsu 冠层分割</div><div class="f-d">自动去除盆壁与土壤背景，输出纯冠层温度。</div></div>
                <div class="feat"><div class="f-i">🚨</div><div class="f-t">CUSUM 早期预警</div><div class="f-d">在线累积和检测，可推送企业微信 / 飞书 / 钉钉。</div></div>
                <div class="feat"><div class="f-i">📱</div><div class="f-t">端云协同</div><div class="f-d">小程序巡检包一键入库，专家复核后下发回手机。</div></div>
              </div>""")
        with r3c2:
            n_acc = len(db.list_accounts())
            n_rev = sum(1 for o in db.list_observations() if o["reviewed"])
            ui_theme.panel("系统运行状态", f"""
              <div class="status-row"><span>数据库引擎</span><span class="dot-ok">● SQLite 在线</span></div>
              <div class="status-row"><span>团队账号</span><b>{n_acc} 个</b></div>
              <div class="status-row"><span>专家已复核</span><b>{n_rev} / {db_obs}</b></div>
              <div class="status-row"><span>当前角色</span><b>{pc_user.get('role_label')}</b></div>
              <div class="status-row"><span>判级口径</span><b>跨床配对残差</b></div>""", "实时")

        ui_theme.panel("标准作业流程", """
          <div class="flow">
            <div class="step"><span class="n">1</span><div class="s-t">现场巡检</div><div class="s-d">小程序拍照 / 算图，记录株位判级。</div></div>
            <div class="step"><span class="n">2</span><div class="s-t">数据入库</div><div class="s-d">导出 JSON 包，在巡检工作台同步入库。</div></div>
            <div class="step"><span class="n">3</span><div class="s-t">分析研判</div><div class="s-d">孪生、分割与反证审计交叉核验。</div></div>
            <div class="step"><span class="n">4</span><div class="s-t">专家复核</div><div class="s-d">后台校准判级并填写复核意见。</div></div>
            <div class="step"><span class="n">5</span><div class="s-t">处置下发</div><div class="s-d">生成同步包回传手机，闭环复水核验。</div></div>
          </div>""")

    # ---------- 数据处理中心：热像 / CSV / 巡检包 三类数据的统一入口 ----------
    with tab_dp:
        ui_theme.page_head("🧪", "数据处理中心", "上传热红外热像、观测 CSV 或手机巡检包，一站式完成分割、判级重算与入库。")
        ui_theme.kpi_cards([
            {"label": "热像批处理", "value": "TIF", "sub": "Otsu 去背景 · 纯冠层温度", "icon": "🔬", "tone": "g1"},
            {"label": "CSV 判级重算", "value": "CSV", "sub": "自定义阈值 · 四级判级", "icon": "📈", "tone": "g3"},
            {"label": "巡检包入库", "value": "JSON", "sub": "小程序数据 · 照片归档", "icon": "📱", "tone": "g4"},
            {"label": "结果导出", "value": "CSV", "sub": "每类处理均可下载", "icon": "⬇️", "tone": "g2"},
        ])
        dp_a, dp_b, dp_c = st.tabs(["热像批量处理", "CSV 判级重算", "巡检包预览与入库"])

        with dp_a:
            ui_theme.section("热红外热像 Otsu 批量去背景")
            st.caption("拖入一张或多张热红外 `.tif`（二维温度矩阵，单位 ℃），逐张分离纯冠层并汇总统计；文件名含 `W3_I8` 这类标识会被识别为株位。")
            tif_files = st.file_uploader("选择热像文件（可多选）", type=["tif", "tiff"], accept_multiple_files=True, key="dp_tifs")
            if tif_files:
                rows_t, first_pack = [], None
                prog = st.progress(0.0, text="正在处理热像…")
                for i, f in enumerate(tif_files):
                    try:
                        arr_f = tifffile.imread(f).astype(np.float32)
                        if arr_f.ndim > 2:
                            arr_f = arr_f.squeeze()
                        if arr_f.ndim != 2:
                            raise ValueError(f"需要二维温度矩阵，实际维度 {arr_f.shape}")
                        m_f, thr_f, st_f = otsu_segment_thermal(arr_f)
                        rows_t.append({
                            "文件": f.name,
                            "冠层占比%": round(st_f["canopy_frac"] * 100, 1),
                            "纯冠层均温(℃)": st_f.get("canopy_mean", float(arr_f[m_f].mean())),
                            "纯冠层P25(℃)": st_f["canopy_p25"],
                            "背景-冠层差(℃)": st_f["bg_gap"],
                            "Otsu阈值(℃)": round(thr_f, 2),
                            "状态": "完成",
                        })
                        if first_pack is None:
                            first_pack = (f.name, arr_f, m_f)
                    except Exception as exc:
                        rows_t.append({"文件": f.name, "状态": f"失败：{exc}"})
                    prog.progress((i + 1) / len(tif_files), text=f"已处理 {i + 1} / {len(tif_files)}")
                prog.empty()
                res_t = pd.DataFrame(rows_t)
                ok_n = int((res_t["状态"] == "完成").sum())
                m1, m2, m3 = st.columns(3)
                m1.metric("上传文件数", len(tif_files))
                m2.metric("处理成功", ok_n)
                m3.metric("平均冠层占比", f"{res_t['冠层占比%'].mean():.1f}%" if ok_n else "—")
                st.dataframe(res_t, use_container_width=True, hide_index=True)
                if first_pack is not None:
                    nm, arr_p, mask_p = first_pack
                    fig_p, axs = plt.subplots(1, 2, figsize=(9.6, 3.2), dpi=130)
                    axs[0].imshow(arr_p, cmap="inferno")
                    axs[0].set_title(f"原始热像（{nm}）", fontsize=9)
                    axs[1].imshow(np.where(mask_p, arr_p, np.nan), cmap="inferno")
                    axs[1].set_title("Otsu 纯冠层掩膜", fontsize=9)
                    for ax_p in axs:
                        ax_p.axis("off")
                    fig_p.tight_layout()
                    st.pyplot(fig_p)
                    plt.close(fig_p)
                st.download_button("导出热像处理结果 CSV", data=res_t.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="thermal_batch_result.csv", mime="text/csv")
            else:
                st.info("尚未上传热像。需要配对判级或 ROI 局部审计时，请使用「🔬 冠层视觉分割」页的专用诊断模式。")

        with dp_b:
            ui_theme.section("观测 CSV 四级判级重算")
            st.caption("上传含残差列的 CSV（如 `resid`），按自定义阈值重新判级，适合新基地或新品种的现场重算。")
            tpl = pd.DataFrame({"unit": ["W3_I8", "W2_B3", "W1_D5"], "date": ["2025-09-15"] * 3, "resid": [5.27, 2.31, 0.18]})
            csv_c1, csv_c2 = st.columns([2.2, 1.0])
            up_csv = csv_c1.file_uploader("选择观测 CSV", type=["csv"], key="dp_csv")
            csv_c2.download_button("下载 CSV 模板", data=tpl.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="resid_template.csv", mime="text/csv", use_container_width=True)
            t1c, t2c, t3c = st.columns(3)
            thr1 = t1c.number_input("L1 轻度阈值 (℃)", value=1.0, step=0.1, key="dp_t1")
            thr2 = t2c.number_input("L2 中度阈值 (℃)", value=2.5, step=0.1, key="dp_t2")
            thr3 = t3c.number_input("L3 重度阈值 (℃)", value=4.5, step=0.1, key="dp_t3")
            if up_csv is not None:
                try:
                    raw_c = pd.read_csv(up_csv)
                    num_cols = [c for c in raw_c.columns if pd.to_numeric(raw_c[c], errors="coerce").notna().mean() > 0.6]
                    if not num_cols:
                        st.error("未找到数值列，请检查文件。")
                    else:
                        rc_idx = num_cols.index("resid") if "resid" in num_cols else 0
                        val_col = st.selectbox("残差所在列", num_cols, index=rc_idx, key="dp_valcol")
                        vals = pd.to_numeric(raw_c[val_col], errors="coerce")
                        lv = np.select([vals >= thr3, vals >= thr2, vals >= thr1], [3, 2, 1], default=0)
                        names = {0: "正常", 1: "轻度", 2: "中度", 3: "重度"}
                        out_c = raw_c.copy()
                        out_c["判级"] = [f"L{x} {names[x]}" if np.isfinite(v) else "无效" for x, v in zip(lv, vals)]
                        valid = vals.notna()
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("有效记录", int(valid.sum()))
                        k2.metric("残差均值", f"{vals.mean():+.2f} ℃")
                        k3.metric("中重度占比", f"{(lv[valid.values] >= 2).mean() * 100:.1f}%" if valid.any() else "—")
                        k4.metric("峰值残差", f"{vals.max():+.2f} ℃")
                        cc_l, cc_r = st.columns([1.0, 1.4], gap="medium")
                        with cc_l:
                            fig_c, ax_c = plt.subplots(figsize=(4.4, 3.2), dpi=140)
                            cnt = [int(((lv == k) & valid.values).sum()) for k in range(4)]
                            ax_c.bar(["正常", "轻度", "中度", "重度"], cnt, color=["#22C55E", "#F59E0B", "#F97316", "#EF4444"])
                            ax_c.set_ylabel("记录数")
                            ax_c.set_title("判级分布", fontsize=10)
                            fig_c.tight_layout()
                            st.pyplot(fig_c)
                            plt.close(fig_c)
                        with cc_r:
                            st.dataframe(out_c, use_container_width=True, hide_index=True, height=290)
                        st.download_button("导出判级重算结果 CSV", data=out_c.to_csv(index=False).encode("utf-8-sig"),
                                           file_name="resid_regrade.csv", mime="text/csv")
                except Exception as exc:
                    st.error(f"CSV 解析失败：{exc}")
            else:
                st.info("尚未上传 CSV。可先下载模板，按格式填写后再上传。")

        with dp_c:
            ui_theme.section("手机巡检包预览与入库")
            if pc_user.get("role") == "guest":
                st.warning("访客账号不能写入数据库，请登录巡检员或管理员账号。")
            else:
                st.caption("先预览数据包内容，确认无误后再写入平台数据库（照片自动归档）。")
                dp_json = st.file_uploader("选择手机端导出的 JSON 数据包", type=["json"], key="dp_json")
                if dp_json is not None:
                    try:
                        pl_dp = json.loads(dp_json.getvalue().decode("utf-8"))
                        if pl_dp.get("app") != "beet-warning-miniprogram":
                            st.error("非本系统标准巡检数据包。")
                        else:
                            obs_dp = pl_dp.get("observations") or []
                            n_photo = sum(1 for o in obs_dp if o.get("imageBase64"))
                            q1, q2, q3, q4, q5 = st.columns(5)
                            q1.metric("账号", len(pl_dp.get("accounts") or []))
                            q2.metric("株位", len(pl_dp.get("plants") or []))
                            q3.metric("观测", len(obs_dp))
                            q4.metric("含照片", n_photo)
                            q5.metric("处置台账", len(pl_dp.get("actions") or []))
                            if obs_dp:
                                st.dataframe(pd.DataFrame([{
                                    "时间": str(o.get("ts", ""))[:19].replace("T", " "),
                                    "株位ID": o.get("plantId"),
                                    "判级": f"L{o.get('level')}",
                                    "冠层覆盖%": (o.get("metrics") or {}).get("canopyPct", "—"),
                                    "操作人": (o.get("operator") or {}).get("name", "—"),
                                    "备注": o.get("note", ""),
                                } for o in obs_dp[:200]]), use_container_width=True, hide_index=True, height=240)
                            if st.button("确认写入数据库", type="primary", key="dp_ingest_btn"):
                                cnts = db.ingest_miniprogram_payload(pl_dp, actor=pc_user.get("username", "user"))
                                st.success(f"入库完成：账号 {cnts['accounts']}、株位 {cnts['plants']}、观测 {cnts['observations']}、台账 {cnts['actions']}。")
                    except Exception as exc:
                        st.error(f"数据包解析失败：{exc}")
                else:
                    st.info("尚未选择数据包。")

    # ---------- Tab 1: 3D 温室与判级全景 ----------
    with tab1:
        ui_theme.page_head("🌿", "温室全景孪生", "3D 数字孪生温室 · 逐单元判级与处方联动，点选株位查看热红外残差与处置建议。")
        html_3d = build_embedded_3d_html()
        if html_3d is not None:
            components.html(html_3d, height=540, scrolling=False)

        summary_rows = []
        for p_id, g in pairs.groupby("plant"):
            g_in = g[g["phase"] == "in"]
            l3 = float(g_in["resid"].tail(3).mean()) if not g_in.empty else float("nan")
            pre_m = float(g.loc[g["phase"] == "pre", "resid"].mean())
            in_m = float(g_in["resid"].mean()) if not g_in.empty else float("nan")
            lvl, g_name, _, g_advice = classify_residual(l3)
            summary_rows.append({
                "处理单元": p_id,
                "床位": p_id.split("_")[0],
                "pre均值(℃)": round(pre_m, 3),
                "in均值(℃)": round(in_m, 3),
                "单元DiD(℃)": round(in_m - pre_m, 3),
                "窗末3次均值(℃)": round(l3, 3),
                "等级": lvl,
                "工程判级": g_name,
                "实证观测建议": g_advice,
            })
        sum_df = pd.DataFrame(summary_rows)

        c_left, c_right = st.columns([1.1, 1.0])
        with c_left:
            # 22 单元窗末升温横向对比图
            fig_bar, ax_b = plt.subplots(figsize=(6.4, 4.3), dpi=140)
            sdf = sum_df.sort_values("窗末3次均值(℃)", ascending=True)
            pal = {0: "#3ddc84", 1: "#ffb020", 2: "#ff7a45", 3: "#ff4d67"}
            colors = [pal.get(int(x), "#35e0c0") for x in sdf["等级"]]
            ax_b.barh(sdf["处理单元"], sdf["窗末3次均值(℃)"], color=colors, height=0.72)
            ax_b.axvline(1.0, color="#ffb020", linestyle="--", linewidth=1.0)
            ax_b.axvline(2.5, color="#ff7a45", linestyle="--", linewidth=1.0)
            ax_b.axvline(4.5, color="#ff4d67", linestyle="--", linewidth=1.0)
            ax_b.spines["top"].set_visible(False)
            ax_b.spines["right"].set_visible(False)
            ax_b.set_xlabel("窗末 3 次配对残差均值 (℃)")
            ax_b.set_title("22 单元缺水升温分布与四级阈值", fontsize=10.5, fontweight="bold")
            ax_b.tick_params(axis="y", labelsize=8)
            ax_b.grid(True, axis="x", linestyle="--", alpha=0.3)
            fig_bar.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

        with c_right:
            sel_plant = st.selectbox("单株轨迹钻取", sorted(pairs["plant"].unique()))
            sub_p = pairs[pairs["plant"] == sel_plant]
            fig, ax = plt.subplots(figsize=(6.0, 3.65), dpi=140)
            ax.plot(sub_p["dt"], sub_p["resid"], marker="o", markersize=2.5, linewidth=1.4, color="#0F4C81")
            ax.axvspan(WINDOW_START, WINDOW_END, color="#FEE2E2", alpha=0.65, label="断水窗")
            ax.axhline(0.0, color="#64748B", linestyle="--", linewidth=0.9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_ylabel("残差 (℃)")
            ax.set_title(f"{sel_plant} 42 天配对残差", fontsize=10.5, fontweight="bold")
            ax.legend(fontsize=8.5, frameon=False)
            ax.grid(True, linestyle="--", alpha=0.35)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.dataframe(sum_df.drop(columns=["等级"]), width="stretch", hide_index=True)
        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "导出 22 单元判级汇总 (CSV)",
            data=sum_df.drop(columns=["等级"]).to_csv(index=False).encode("utf-8-sig"),
            file_name=f"beet_audit_{feature}.csv",
            mime="text/csv",
        )
        dl2.download_button(
            "导出可打印实证审计台账 (HTML/PDF)",
            data=build_audit_report_html(sum_df, feature).encode("utf-8"),
            file_name=f"beet_audit_report_{feature}.html",
            mime="text/html",
        )

        with st.expander("📐 按床热像传感器布点与工程降本测算器（基于 Levene p=0.519 床级一致性）", expanded=False):
            sc1, sc2, sc3, sc4 = st.columns(4)
            n_beds = sc1.number_input("温室苗床数 (张)", min_value=1, max_value=100, value=6, step=1)
            bed_len = sc2.number_input("单床长度 (m)", min_value=2.0, max_value=100.0, value=10.5, step=0.5)
            plants_per_bed = sc3.number_input("单床种植株数", min_value=10, max_value=2000, value=120, step=10)
            cam_cov = sc4.number_input("单台热像仪有效覆盖床长 (m)", min_value=2.0, max_value=30.0, value=5.5, step=0.5)
            cams_per_bed = int(np.ceil(bed_len / max(0.5, cam_cov)))
            total_cams = int(n_beds * cams_per_bed)
            total_plants = int(n_beds * plants_per_bed)
            save_pct = (1.0 - total_cams / max(1, total_plants)) * 100.0
            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("按床布点需热像仪", f"{total_cams} 台", f"每床 {cams_per_bed} 台")
            cm2.metric("传统逐株探头点位", f"{total_plants} 个")
            cm3.metric("硬件布点压缩率", f"{save_pct:.1f}%", "依托床内同向响应特征")

        with st.expander("🌱 新大棚 / 新品种 7 天基线自动标定器（跨温室通用迁移）", expanded=False):
            st.caption("在新温室或新作物上连续采集 5–7 天正常灌溉期（pre 期）配对热红外残差，系统自动生成专属阈值配置 `site_profile.json`。")
            cal_col1, cal_col2 = st.columns([1.2, 1.0])
            cal_csv = cal_col1.file_uploader("上传新基地正常期观测 CSV（需含 resid 列）", type=["csv"], key="cal_csv")
            site_name = cal_col2.text_input("基地 / 作物品种名称", value="示范基地-1号棚-甜菜")
            if cal_csv is not None:
                cal_df = pd.read_csv(cal_csv)
                cal_vals = pd.to_numeric(cal_df.get("resid", pd.Series(dtype=float)), errors="coerce").dropna()
            else:
                cal_vals = pairs.loc[pairs["phase"] == "pre", "resid"].dropna()
            if len(cal_vals) >= 10:
                mu0, sd0 = float(cal_vals.mean()), float(cal_vals.std(ddof=1))
                l1_c, l2_c, l3_c = mu0 + 1.645 * sd0, mu0 + 3.0 * sd0, mu0 + 4.5 * sd0
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("标定基线 μ₀", f"{mu0:+.3f} ℃")
                cc2.metric("基线波动 σ₀", f"{sd0:.3f} ℃")
                cc3.metric("L1 / L2 阈值", f"+{l1_c:.2f} / +{l2_c:.2f}℃")
                cc4.metric("L3 重度阈值", f"+{l3_c:.2f} ℃")
                profile_obj = {
                    "site": site_name,
                    "n_baseline_obs": int(len(cal_vals)),
                    "baseline_mean": round(mu0, 4),
                    "baseline_sd": round(sd0, 4),
                    "thresholds_degC": {"L1": round(l1_c, 3), "L2": round(l2_c, 3), "L3": round(l3_c, 3)},
                    "cusum": {"k_sigma": 0.5, "h_sigma": 4.5},
                }
                st.download_button(
                    "导出新基地专属标定配置 (site_profile.json)",
                    data=json.dumps(profile_obj, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name="site_profile.json",
                    mime="application/json",
                )

    # ---------- Tab 2: Otsu 冠层分割、双帧配对与整棚批量巡检 ----------
    with tab2:
        ui_theme.page_head("🔬", "冠层视觉分割", "Otsu 冠层去背景 · 双帧配对判级 · 整棚批量巡检红黑榜。")
        diag_mode = st.radio(
            "诊断模式",
            ["单帧 Otsu 冠层分割检视", "现场双帧配对快速诊断 (处理 vs 对照)", "整棚批量热像自动巡检队列"],
            horizontal=True,
        )
        sample_tifs = sorted((DATA_DIR / "official_tree" / "thermal").glob("*/*/15/thermal_002.tif"))[:20]
        if diag_mode == "单帧 Otsu 冠层分割检视":
            uploaded_tif = st.file_uploader("上传热红外 .tif（或选本地观测帧）", type=["tif", "tiff"])
            arr = None
            src_label = ""
            if uploaded_tif is not None:
                arr = tifffile.imread(uploaded_tif).astype(np.float32)
                src_label = uploaded_tif.name
            elif sample_tifs:
                pick = st.selectbox("观测帧", sample_tifs, format_func=lambda p: "/".join(p.parts[-4:]))
                arr = tifffile.imread(pick).astype(np.float32)
                src_label = "/".join(pick.parts[-4:])

            if arr is not None:
                low_mask, thr, stats = otsu_segment_thermal(arr)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("冠层占比", f"{stats['canopy_frac']*100:.1f}%")
                c2.metric("纯冠层 P25", f"{stats['canopy_p25']:.2f} ℃")
                c3.metric("背景-冠层差", f"+{stats['bg_gap']:.2f} ℃")
                c4.metric("Otsu 阈值", f"{thr:.2f} ℃")

                fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), dpi=130)
                im0 = axes[0].imshow(arr, cmap="inferno")
                axes[0].set_title(f"原始热像 ({src_label})", fontsize=9)
                plt.colorbar(im0, ax=axes[0], fraction=0.046)

                im1 = axes[1].imshow(np.where(low_mask, arr, np.nan), cmap="inferno")
                axes[1].set_title("Otsu 纯冠层掩膜", fontsize=9)
                plt.colorbar(im1, ax=axes[1], fraction=0.046)

                axes[2].hist(arr[low_mask].ravel(), bins=35, alpha=0.75, color="#009e73", label="冠层")
                if (~low_mask).any():
                    axes[2].hist(arr[~low_mask].ravel(), bins=35, alpha=0.65, color="#d55e00", label="背景")
                axes[2].axvline(thr, color="#2c3e50", linestyle="--", label=f"{thr:.1f}℃")
                axes[2].set_title("温度双峰分布", fontsize=9)
                axes[2].legend(fontsize=8)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                with st.expander("🔍 局部叶片 ROI 像素温度切片审计（对标 LemnaGrid 区域分析）", expanded=False):
                    rc1, rc2 = st.columns(2)
                    x_rng = rc1.slider("ROI 水平范围 (%)", 0, 100, (25, 75), key="roi_x")
                    y_rng = rc2.slider("ROI 垂直范围 (%)", 0, 100, (25, 75), key="roi_y")
                    h_img, w_img = arr.shape[:2]
                    x0, x1 = int(w_img * x_rng[0] / 100), max(int(w_img * x_rng[1] / 100), int(w_img * x_rng[0] / 100) + 2)
                    y0, y1 = int(h_img * y_rng[0] / 100), max(int(h_img * y_rng[1] / 100), int(h_img * y_rng[0] / 100) + 2)
                    roi_patch = arr[y0:y1, x0:x1]
                    roi_canopy = roi_patch[low_mask[y0:y1, x0:x1]]
                    rm1, rm2, rm3, rm4 = st.columns(4)
                    rm1.metric("ROI 像素数", f"{roi_patch.size:,}")
                    rm2.metric("ROI 均温", f"{float(np.mean(roi_patch)):.2f} ℃")
                    rm3.metric("ROI 冠层 P25", f"{float(np.percentile(roi_canopy, 25)) if roi_canopy.size else float(np.percentile(roi_patch, 25)):.2f} ℃")
                    rm4.metric("ROI 最高温", f"{float(np.max(roi_patch)):.2f} ℃")
                    from matplotlib.patches import Rectangle
                    fig_r, (ax_r1, ax_r2) = plt.subplots(1, 2, figsize=(9.6, 3.0), dpi=130)
                    ax_r1.imshow(arr, cmap="inferno")
                    ax_r1.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#35e0c0", linewidth=2.0))
                    ax_r1.set_title("选区 ROI 空间定位", fontsize=9)
                    ax_r2.hist(roi_patch.ravel(), bins=28, color="#35e0c0", alpha=0.85, edgecolor="#0e3f5c")
                    ax_r2.set_title("ROI 局部像素温度直方图", fontsize=9)
                    fig_r.tight_layout()
                    st.pyplot(fig_r)
                    plt.close(fig_r)
        elif diag_mode == "现场双帧配对快速诊断 (处理 vs 对照)":
            st.caption("分别上传或选取“处理床”与“对照床”同期热红外 .tif，系统自动剥离盆壁/土壤背景并输出纯冠层配对判级。")
            p_col1, p_col2 = st.columns(2)
            up_treat = p_col1.file_uploader("上传处理床热像 (.tif)", type=["tif", "tiff"], key="up_treat")
            up_ctrl = p_col2.file_uploader("上传对照床热像 (.tif)", type=["tif", "tiff"], key="up_ctrl")
            treat_samples = [p for p in sample_tifs if "W3_" in str(p) or "W2_" in str(p)] or sample_tifs
            ctrl_samples = [p for p in sample_tifs if "W1_" in str(p)] or sample_tifs
            arr_t = tifffile.imread(up_treat).astype(np.float32) if up_treat is not None else (
                tifffile.imread(p_col1.selectbox("或选处理床样本帧", treat_samples, format_func=lambda p: "/".join(p.parts[-4:]))).astype(np.float32) if treat_samples else None
            )
            arr_c = tifffile.imread(up_ctrl).astype(np.float32) if up_ctrl is not None else (
                tifffile.imread(p_col2.selectbox("或选对照床样本帧", ctrl_samples, format_func=lambda p: "/".join(p.parts[-4:]))).astype(np.float32) if ctrl_samples else None
            )
            if arr_t is not None and arr_c is not None:
                _, _, st_t = otsu_segment_thermal(arr_t)
                _, _, st_c = otsu_segment_thermal(arr_c)
                delta_t = st_t["canopy_p25"] - st_c["canopy_p25"]
                z_val = (delta_t - 0.289) / 1.753
                _, g_name, g_icon, g_adv = classify_residual(delta_t)
                dm1, dm2, dm3, dm4 = st.columns(4)
                dm1.metric("处理床纯冠层 P25", f"{st_t['canopy_p25']:.2f} ℃")
                dm2.metric("对照床纯冠层 P25", f"{st_c['canopy_p25']:.2f} ℃")
                dm3.metric("纯冠层配对 ΔT", f"{delta_t:+.2f} ℃", f"z = {z_val:.2f}")
                dm4.metric("实时工程判级", f"{g_icon} {g_name}")
                st.info(f"**现场实证观测建议：** {g_adv}")
                slip = (
                    f"<html><meta charset='utf-8'><body style='font-family:Microsoft YaHei;padding:24px;'>"
                    f"<h2>温室单次热像配对巡检诊断单</h2>"
                    f"<p>处理床纯冠层 P25：<b>{st_t['canopy_p25']:.2f} ℃</b> | 对照床纯冠层 P25：<b>{st_c['canopy_p25']:.2f} ℃</b></p>"
                    f"<p>纯冠层配对残差 ΔT：<b>{delta_t:+.2f} ℃</b>（z = {z_val:.2f}）</p>"
                    f"<p>工程判级：<b>{g_name}</b> | 建议：{g_adv}</p></body></html>"
                )
                st.download_button("导出本次热像巡检诊断单 (HTML)", data=slip.encode("utf-8"), file_name="thermal_pair_diagnosis.html", mime="text/html")
        else:
            st.caption("一次性批量导入多张热像（或一键扫描当日全棚样本目录），自动按 W1 同株位配对、批量 Otsu 去背景并生成全棚异常红黑榜。")
            batch_files = st.file_uploader("批量拖入多张热红外 .tif（文件名建议含如 W3_I8）", type=["tif", "tiff"], accept_multiple_files=True, key="batch_tifs")
            use_local_day = st.button("一键批量扫描 2025-09-15 全棚实测热像帧")
            batch_rows = []
            if batch_files:
                ctrl_arrs = []
                treat_items = []
                for f in batch_files:
                    a = tifffile.imread(f).astype(np.float32)
                    _, _, st_f = otsu_segment_thermal(a)
                    if "W1" in f.name.upper():
                        ctrl_arrs.append(st_f["canopy_p25"])
                    else:
                        treat_items.append((f.name, st_f["canopy_p25"], st_f["canopy_frac"]))
                ctrl_ref = float(np.mean(ctrl_arrs)) if ctrl_arrs else 21.5
                for fname, p25_t, frac_t in treat_items:
                    dt_b = p25_t - ctrl_ref
                    _, g_n, _, g_a = classify_residual(dt_b)
                    batch_rows.append({"影像文件": fname, "冠层占比": f"{frac_t*100:.1f}%", "纯冠层P25(℃)": round(p25_t, 2), "配对ΔT(℃)": round(dt_b, 2), "判级": g_n, "处置建议": g_a})
            elif use_local_day:
                day_tifs = sorted((DATA_DIR / "official_tree" / "thermal").glob("*/2025_09_15/15/thermal_002.tif"))
                ctrl_map = {}
                treat_map = {}
                for p in day_tifs:
                    uid = p.parts[-4]
                    _, _, st_p = otsu_segment_thermal(tifffile.imread(p).astype(np.float32))
                    if uid.startswith("W1_"):
                        ctrl_map[uid.split("_")[1]] = st_p["canopy_p25"]
                    else:
                        treat_map[uid] = (st_p["canopy_p25"], st_p["canopy_frac"])
                for uid, (p25_t, frac_t) in sorted(treat_map.items()):
                    code = uid.split("_")[1]
                    c_ref = ctrl_map.get(code, float(np.mean(list(ctrl_map.values()))) if ctrl_map else 21.5)
                    dt_b = p25_t - c_ref
                    _, g_n, _, g_a = classify_residual(dt_b)
                    batch_rows.append({"处理单元": uid, "冠层占比": f"{frac_t*100:.1f}%", "纯冠层P25(℃)": round(p25_t, 2), "对照P25(℃)": round(c_ref, 2), "配对ΔT(℃)": round(dt_b, 2), "判级": g_n, "处置建议": g_a})
            if batch_rows:
                bdf = pd.DataFrame(batch_rows).sort_values("配对ΔT(℃)", ascending=False)
                st.dataframe(bdf, width="stretch", hide_index=True)
                st.download_button("导出全棚批量巡检红黑榜 (CSV)", data=bdf.to_csv(index=False).encode("utf-8-sig"), file_name="batch_thermal_patrol.csv", mime="text/csv")

    # ---------- Tab 3: 四象限反证审计 ----------
    with tab3:
        ui_theme.page_head("📊", "因果反证审计", "四象限反证与安慰剂检验，核验判级信号并非来自背景或时间漂移。")
        vc1, vc2 = st.columns(2)
        if (FIG_DIR / "fig1_model_audit_quadrant.png").exists():
            vc1.image(str(FIG_DIR / "fig1_model_audit_quadrant.png"), width="stretch")
        if (FIG_DIR / "fig2_otsu_and_diurnal.png").exists():
            vc2.image(str(FIG_DIR / "fig2_otsu_and_diurnal.png"), width="stretch")
        step4_path = V8_DIR / "step4_daytime_and_multimodal.json"
        if step4_path.exists():
            step4 = json.loads(step4_path.read_text(encoding="utf-8"))
            st.dataframe(pd.DataFrame([r for r in step4["model"] if r["regime"] == "all_slots"]),
                         width="stretch", hide_index=True)

    # ---------- Tab 4: 微气候耦合分析 ----------
    with tab4:
        ui_theme.page_head("🌡️", "微气候耦合", "冠层温度 / VPD / CWSI 与棚内微气象的耦合机理分析。")
        mc1, mc2 = st.columns([1.25, 1.0])
        if (FIG_DIR / "fig3_microclimate_coupling.png").exists():
            mc1.image(str(FIG_DIR / "fig3_microclimate_coupling.png"), width="stretch")
        phys_df = load_physics_table()
        if phys_df is not None:
            grp = phys_df.groupby(["bed", "phase"])[["ta", "rh", "vpd", "tm_p25"]].mean().round(3).reset_index()
            in_grp = grp[grp["phase"] == "in"]
            fig_vpd, ax_v = plt.subplots(figsize=(5.0, 3.6))
            x = np.arange(len(in_grp))
            ax_v.bar(x - 0.18, in_grp["ta"], width=0.35, label="气温 Ta (℃)", color="#56b4e9")
            ax_v.bar(x + 0.18, in_grp["tm_p25"], width=0.35, label="冠层 P25 (℃)", color="#d55e00")
            ax_v.set_xticks(x)
            ax_v.set_xticklabels(in_grp["bed"])
            ax_v.set_title("断水期各床 Ta vs 冠层温度", fontsize=10, fontweight="bold")
            ax_v.legend(fontsize=8.5)
            ax_v.grid(True, axis="y", linestyle="--", alpha=0.35)
            fig_vpd.tight_layout()
            mc2.pyplot(fig_vpd)
            plt.close(fig_vpd)
            # 相对 CWSI 经验估算（基于观测 (Tc-Ta) 上下界归一化，仅供物理机制横向对比）
            phys_df["tc_ta"] = phys_df["tm_p25"] - phys_df["ta"]
            ll = float(phys_df["tc_ta"].quantile(0.02))
            ul = float(phys_df["tc_ta"].quantile(0.98))
            phys_df["cwsi_rel"] = np.clip((phys_df["tc_ta"] - ll) / max(0.5, ul - ll), 0.0, 1.0)
            cwsi_grp = phys_df.groupby(["bed", "phase"])[["tc_ta", "cwsi_rel"]].mean().round(3).reset_index()
            grp = grp.merge(cwsi_grp, on=["bed", "phase"])
            st.dataframe(grp, width="stretch", hide_index=True)
            st.caption(f"注：相对 CWSI 采用本数据集观测 (Tc−Ta) 2%–98% 分位（{ll:+.2f}℃ ~ {ul:+.2f}℃）作经验上下界参考，未经风速/净辐射标定，仅作横向机理解释。")

    # ---------- Tab 5: CUSUM 早期预警 ----------
    with tab5:
        ui_theme.page_head("🚨", "早期胁迫预警", "在线 CUSUM 早期检出与首警延迟分布，可推送企业微信 / 飞书 / 钉钉。")
        c_k, c_h, c_p = st.columns([1, 1, 1.6])
        k_val = c_k.slider("松弛量 k (σ)", 0.2, 1.2, 0.5, 0.1)
        h_val = c_h.slider("报警阈值 h (σ)", 2.5, 8.0, 4.5, 0.5)
        cusum_plant = c_p.selectbox("单株轨迹", sorted(pairs["plant"].unique()), key="cusum_plant")

        tr_list = [run_causal_cusum(grp_p, k_allow=k_val, h_thresh=h_val) for _, grp_p in pairs.groupby("plant")]
        tr_list = [t for t in tr_list if not t.empty]
        if tr_list:
            all_tr = pd.concat(tr_list, ignore_index=True)
            pre_fpr = float(all_tr.loc[all_tr["phase"] == "pre", "alarm"].mean())
            in_det = all_tr[all_tr["phase"] == "in"].groupby("plant")["alarm"].max()
            post_end = all_tr[all_tr["phase"] == "post"].groupby("plant")["alarm"].last()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("断水检出", f"{int((in_det == 1).sum())} / {len(in_det)}")
            m2.metric("断水前误报", f"{pre_fpr*100:.2f}%")
            m3.metric("复水解除", f"{int((post_end == 0).sum())} / {len(post_end)}")
            v9_json = V9_DIR / "cusum_summary.json"
            if v9_json.exists():
                v9s = json.loads(v9_json.read_text(encoding="utf-8"))
                m4.metric("安慰剂中位误报", f"{v9s['placebo_in_alarm_rate_median']*100:.1f}%")

            cc1, cc2 = st.columns([1.4, 1.0])
            sub_tr = all_tr[all_tr["plant"] == cusum_plant]
            fig_c, ax_c = plt.subplots(figsize=(6.8, 3.2))
            ax_c.plot(sub_tr["dt"], sub_tr["cusum_s"], color="#0072b2", linewidth=1.6, label="S+")
            ax_c.axhline(h_val, color="#c0392b", linestyle="--", linewidth=1.2, label=f"h={h_val}")
            ax_c.axvspan(WINDOW_START, WINDOW_END, color="#f9e79f", alpha=0.45, label="断水窗")
            ax_c.set_title(f"{cusum_plant} 在线 CUSUM 轨迹", fontsize=10, fontweight="bold")
            ax_c.legend(fontsize=8)
            ax_c.grid(True, linestyle="--", alpha=0.35)
            fig_c.tight_layout()
            cc1.pyplot(fig_c)
            plt.close(fig_c)

            # 22 单元首次报警延迟分布直方图
            in_al = all_tr[(all_tr["phase"] == "in") & (all_tr["alarm"] == 1)]
            delays = ((in_al.groupby("plant")["dt"].min() - WINDOW_START).dt.total_seconds() / 3600.0).dropna()
            fig_d, ax_d = plt.subplots(figsize=(4.8, 3.2))
            ax_d.hist(delays, bins=10, color="#009e73", edgecolor="white", alpha=0.88)
            ax_d.axvline(delays.median(), color="#d55e00", linestyle="--", label=f"中位 {delays.median():.0f}h")
            ax_d.set_xlabel("首次报警延迟 (小时)")
            ax_d.set_ylabel("单元数")
            ax_d.set_title("22 单元早期检出延迟分布", fontsize=10, fontweight="bold")
            ax_d.legend(fontsize=8)
            ax_d.grid(True, axis="y", linestyle="--", alpha=0.35)
            fig_d.tight_layout()
            cc2.pyplot(fig_d)
            plt.close(fig_d)

            with st.expander("🔔 Webhook 手机告警推送与热像网关接入协议", expanded=False):
                st.caption("支持将在线 CUSUM 预警与 L2/L3 异常株位推送至企业微信 / 飞书 / 钉钉群机器人，并附标准硬件接入 JSON Schema。")
                wh1, wh2 = st.columns([1.8, 1.0])
                wh_url = wh1.text_input("机器人 Webhook URL（可选，留空仅预览报文）", value="", placeholder="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=...")
                wh_plat = wh2.selectbox("推送格式", ["企业微信 / 钉钉 Markdown", "飞书交互卡片文本"])
                alert_units = sorted(in_det[in_det == 1].index.tolist())[:5]
                msg_text = (
                    f"【温室甜菜水分胁迫 CUSUM 在线预警】\n"
                    f"- 检出单元数：{int((in_det == 1).sum())} / {len(in_det)}（参数 k={k_val}, h={h_val}）\n"
                    f"- 重点复核株位：{', '.join(alert_units)}\n"
                    f"- 处置建议：请巡检员携带移动终端优先核验上述苗床滴灌与冠层热红外状态。"
                )
                payload_wh = {"msgtype": "text", "text": {"content": msg_text}} if "飞书" in wh_plat else {"msgtype": "markdown", "markdown": {"content": msg_text}}
                st.code(json.dumps(payload_wh, ensure_ascii=False, indent=2), language="json")
                if st.button("发送实时测试告警"):
                    if not wh_url.strip().startswith("http"):
                        st.warning("请先填入有效的 http(s) Webhook URL。")
                    else:
                        import urllib.request
                        req = urllib.request.Request(
                            wh_url.strip(),
                            data=json.dumps(payload_wh, ensure_ascii=False).encode("utf-8"),
                            headers={"Content-Type": "application/json"},
                        )
                        try:
                            with urllib.request.urlopen(req, timeout=5) as resp:
                                st.success(f"告警推送成功，HTTP {resp.status}")
                        except Exception as exc:
                            st.error(f"推送失败：{exc}")

    # ---------- Tab 6: 巡检工作台（所有登录用户可用） ----------
    with tab6:
        ui_theme.page_head("👤", "巡检工作台", "移动端巡检包入库与个人台账查看，权限范围内的数据一站式管理。")
        if pc_user.get("role") == "guest":
            st.info("当前处于只读访客视图，可浏览全部分析看板；登录巡检员或管理员账号后可管理个人巡检台账与上传数据。")
        else:
            my_obs = db.list_observations(operator_name=None if is_admin else pc_user.get("name"))
            k1, k2, k3 = st.columns(3)
            k1.metric("名下观测总数", len(my_obs))
            k2.metric("已完成专家复核", sum(1 for o in my_obs if o["reviewed"]))
            k3.metric("当前操作员", pc_user.get("name"))

            u_left, u_right = st.columns([1.0, 2.1], gap="large")
            with u_left:
                st.markdown("**移动端数据包上传**")
                st.caption("支持导入微信小程序导出的 JSON 巡检包（含照片与 Otsu 指标）。")
                my_up = st.file_uploader(
                    "选择备份数据包 (甜菜预警备份_*.json)",
                    type=["json"],
                    key="user_mp_upload",
                )
                if my_up is not None and st.button("同步入库", type="primary", use_container_width=True, key="user_ingest_btn"):
                    try:
                        pl = json.loads(my_up.getvalue().decode("utf-8"))
                        if pl.get("app") != "beet-warning-miniprogram":
                            st.error("非本系统标准巡检数据包。")
                        else:
                            cnts = db.ingest_miniprogram_payload(pl, actor=pc_user.get("username", "user"))
                            st.success(f"同步完成：株位 {cnts['plants']}、观测 {cnts['observations']}、台账 {cnts['actions']}。")
                            st.rerun()
                    except Exception as exc:
                        st.error(f"数据包解析失败：{exc}")

            with u_right:
                st.markdown("**个人巡检台账**")
                if my_obs:
                    my_df = pd.DataFrame([{
                        "记录时间": str(o["ts"])[:19].replace("T", " "),
                        "床位": o.get("plant_bed") or "—",
                        "株号": o.get("plant_code") or o["plant_id"],
                        "判级": f"L{o['level']}",
                        "复核状态": f"专家已复核 L{o['expert_level']}" if o["reviewed"] else "待复核",
                        "复核意见": o.get("expert_note") or "—",
                        "冠层覆盖%": o["canopy_pct"] if o["canopy_pct"] is not None else "—",
                        "黄化占比%": o["yellow_pct"] if o["yellow_pct"] is not None else "—",
                        "备注": o["note"],
                    } for o in my_obs])
                    st.dataframe(my_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "导出巡检台账 CSV",
                        data=my_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"inspections_{pc_user.get('username')}.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("暂无巡检记录，可通过左侧上传移动端数据包同步入库。")

    # ---------- Tab 7: 后台与数据中心（仅 admin 可见） ----------
    if is_admin:
        with tabs[8]:
            ui_theme.page_head("⚙️", "后台与数据中心", "团队账号权限、巡检数据入库、专家复核与端云同步下发。")
            all_accs = db.list_accounts()
            all_plants = db.list_plants()
            all_obs = db.list_observations()
            all_acts = db.list_actions()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("团队账号数", len(all_accs))
            m2.metric("在管株位数", len(all_plants))
            m3.metric("累计巡检观测", len(all_obs))
            m4.metric("专家已复核", f"{sum(1 for o in all_obs if o['reviewed'])} / {len(all_obs)}")

            sub_overview, sub_users, sub_review, sub_logs = st.tabs([
                "数据同步与库管理",
                "团队账号管理",
                "巡检记录与专家复核",
                "系统操作日志",
            ])

            with sub_overview:
                c_in, c_out = st.columns(2, gap="large")
                with c_in:
                    st.markdown('<div class="sec-hdr">移动端数据入库</div>', unsafe_allow_html=True)
                    st.caption("将巡检员手机端导出的 JSON 数据包合并写入平台数据库，并自动归档现场照片。")
                    adm_file = st.file_uploader(
                        "上传小程序数据包 (beet_backup_*.json)",
                        type=["json"],
                        key="admin_mp_upload",
                    )
                    if adm_file is not None and st.button("入库持久化", type="primary", use_container_width=True, key="admin_ingest_btn"):
                        try:
                            pl = json.loads(adm_file.getvalue().decode("utf-8"))
                            if pl.get("app") != "beet-warning-miniprogram":
                                st.error("非本系统标准巡检数据包。")
                            else:
                                cnts = db.ingest_miniprogram_payload(pl, actor=pc_user.get("username", "admin"))
                                st.success(f"入库完成：新增账号 {cnts['accounts']}、株位 {cnts['plants']}、观测 {cnts['observations']}、台账 {cnts['actions']}。")
                                st.rerun()
                        except Exception as exc:
                            st.error(f"入库失败：{exc}")

                with c_out:
                    st.markdown('<div class="sec-hdr">端云同步下发与数据库</div>', unsafe_allow_html=True)
                    st.caption("生成包含最新团队账号权限与专家复核意见的同步包，下发至微信端「我的 → 备份恢复」生效。")
                    sync_json = json.dumps(db.export_sync_payload(), ensure_ascii=False).encode("utf-8")
                    st.download_button(
                        "导出下发同步包 (beet_pc_sync.json)",
                        data=sync_json,
                        file_name="beet_pc_sync.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                    if all_acts:
                        st.download_button(
                            "导出农事处置台账 CSV",
                            data=pd.DataFrame(all_acts).to_csv(index=False).encode("utf-8-sig"),
                            file_name="pc_db_actions.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    with st.expander("Navicat 数据库连接信息", expanded=False):
                        st.code(str(db.DB_PATH), language="text")
                        st.caption("在 Navicat 中选择「新建连接 → SQLite」，选择上方 `.db` 路径即可直接管理 `accounts / plants / observations / actions / audit_logs` 表。")

            with sub_users:
                t_left, t_right = st.columns([1.45, 1.0], gap="large")
                with t_left:
                    st.markdown('<div class="sec-hdr">团队成员列表</div>', unsafe_allow_html=True)
                    acc_df = pd.DataFrame([{
                        "登录账号": a["username"],
                        "姓名": a["name"],
                        "岗位角色": a["role_label"],
                        "联系方式": a["phone"] or "—",
                        "状态": "已停用" if a["disabled"] else "正常",
                        "创建时间": str(a["created_at"])[:16].replace("T", " "),
                    } for a in all_accs])
                    st.dataframe(acc_df, use_container_width=True, hide_index=True)

                with t_right:
                    st.markdown('<div class="sec-hdr">账号配置与权限调整</div>', unsafe_allow_html=True)
                    u_op_tab1, u_op_tab2 = st.tabs(["新增成员", "编辑 / 停用"])
                    with u_op_tab1:
                        nu = st.text_input("登录账号", placeholder="如：zhao02", key="adm_new_u")
                        nn = st.text_input("姓名", placeholder="如：赵农艺", key="adm_new_n")
                        nr_label = st.selectbox("岗位角色", ["农技巡检员", "科研管理员", "温室操作员"], key="adm_new_r")
                        npw = st.text_input("初始密码", value="123456", key="adm_new_pw")
                        nph = st.text_input("联系电话（可选）", key="adm_new_ph")
                        if st.button("创建账号", type="primary", use_container_width=True, key="adm_add_acc_btn"):
                            sys_r = "admin" if nr_label == "科研管理员" else "user"
                            ok, msg = db.create_account(nu, nn, sys_r, nr_label, npw, nph, actor=pc_user.get("username", "admin"))
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)

                    with u_op_tab2:
                        if all_accs:
                            labels = [f"{a['username']} · {a['name']} ({a['role_label']})" for a in all_accs]
                            idx_a = st.selectbox("选择成员", range(len(all_accs)), format_func=lambda i: labels[i], key="adm_edit_sel")
                            t_acc = all_accs[idx_a]
                            r_opts = ["农技巡检员", "科研管理员", "温室操作员"]
                            r_idx = r_opts.index(t_acc["role_label"]) if t_acc["role_label"] in r_opts else 0
                            e_rlabel = st.selectbox("调整角色", r_opts, index=r_idx, key="adm_edit_r")
                            e_dis = st.radio("账号状态", ["正常", "已停用"], index=1 if t_acc["disabled"] else 0, horizontal=True, key="adm_edit_st")
                            e_pw = st.text_input("重置密码（留空不改）", type="password", key="adm_edit_pw")
                            if st.button("保存变更", type="primary", use_container_width=True, key="adm_save_acc_btn"):
                                sys_r = "admin" if e_rlabel == "科研管理员" else "user"
                                db.update_account(t_acc["id"], sys_r, e_rlabel, e_dis == "已停用", e_pw, actor=pc_user.get("username", "admin"))
                                st.success(f"已更新 {t_acc['username']}")
                                st.rerun()

            with sub_review:
                st.markdown('<div class="sec-hdr">全员田间观测与专家复核</div>', unsafe_allow_html=True)
                if not all_obs:
                    st.info("数据库尚无观测记录，请在「数据同步与库管理」中导入手机端巡检包。")
                else:
                    obs_df = pd.DataFrame([{
                        "时间": str(o["ts"])[:19].replace("T", " "),
                        "床位": o.get("plant_bed") or "—",
                        "株号": o.get("plant_code") or o["plant_id"],
                        "操作人": o["operator_name"],
                        "原判级": f"L{o['level']}",
                        "专家复核": f"已复核 L{o['expert_level']}" if o["reviewed"] else "待复核",
                        "复核意见": o.get("expert_note") or "—",
                        "冠层覆盖%": o["canopy_pct"] if o["canopy_pct"] is not None else "—",
                        "黄化占比%": o["yellow_pct"] if o["yellow_pct"] is not None else "—",
                        "备注": o["note"],
                    } for o in all_obs])
                    st.dataframe(obs_df, use_container_width=True, hide_index=True)

                    rev_col, pic_col = st.columns([1.0, 1.4], gap="large")
                    with rev_col:
                        st.markdown("**专家复核校准**")
                        o_labels = [
                            f"{o.get('plant_bed') or ''}_{o.get('plant_code') or o['plant_id']} · {o['operator_name']} · 原L{o['level']}"
                            for o in all_obs
                        ]
                        sel_oi = st.selectbox("选择观测记录", range(len(all_obs)), format_func=lambda i: o_labels[i], key="adm_rev_sel")
                        t_obs = all_obs[sel_oi]
                        def_lv = int(t_obs["expert_level"] if t_obs["reviewed"] and t_obs["expert_level"] is not None else t_obs["level"])
                        r_lv = st.selectbox("复核等级", [0, 1, 2, 3], index=min(max(def_lv, 0), 3), format_func=lambda x: f"L{x}", key="adm_rev_lv")
                        r_nt = st.text_input("复核意见", value=t_obs.get("expert_note") or "", placeholder="输入专家判定依据", key="adm_rev_nt")
                        if st.button("提交复核结果", type="primary", use_container_width=True, key="adm_rev_btn"):
                            db.review_observation(t_obs["id"], r_lv, r_nt, pc_user.get("name", "管理员"))
                            st.success("复核结论已保存。")
                            st.rerun()

                    with pic_col:
                        st.markdown("**现场影像核验**")
                        photos = [o for o in all_obs if o.get("photo_path") and Path(o["photo_path"]).exists()]
                        if photos:
                            pcols = st.columns(2)
                            for idx_p, o in enumerate(photos[:6]):
                                cap = f"{o.get('plant_bed') or ''}_{o.get('plant_code') or ''} · L{o['level']} · {o['operator_name']}"
                                pcols[idx_p % 2].image(o["photo_path"], caption=cap, use_container_width=True)
                        else:
                            st.caption("当前入库记录暂无附带现场照片。")

            with sub_logs:
                st.markdown('<div class="sec-hdr">数据库操作审计日志</div>', unsafe_allow_html=True)
                logs = db.list_audit_logs(80)
                if logs:
                    log_df = pd.DataFrame([{
                        "时间": str(lg["ts"])[:19].replace("T", " "),
                        "操作人": lg["actor"],
                        "动作": lg["action"],
                        "详情": lg["detail"],
                    } for lg in logs])
                    st.dataframe(log_df, use_container_width=True, hide_index=True)
                else:
                    st.info("暂无操作日志。")


if __name__ == "__main__":
    main()
