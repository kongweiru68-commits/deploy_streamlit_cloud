"""跨床同株位配对的双重差分（DiD）：把"2 株个案"升级成带置信区间的估计。

为什么必须换配对方式：
  `THERMAL_UNSUPERVISED.md` 的做法是"目标株 − 全体对照均值"。那只在只有 1 个缺水株时
  无可奈何。面板补齐后，每株缺水株都有**同株位的对照**（W2_A4 ↔ W1_A4），
  这能消掉"株位固有差异"（同一苗床位置的光照/风口/滴灌头差异），是 DiD 成立的前提。
  用全体对照均值反而会把这个株位效应留在残差里。

统计口径（写死在代码里，不靠调用方记得）：
  * 残差 `resid = T_缺水 − T_同株位对照`，同时次、同一天，两级都对齐才用；
  * **按时次分层**：`THERMAL_UNSUPERVISED.md` 已证明时次效应（15:00 d=2.42 vs 06:00 d=0.22）
    大于处理效应，混在一起会把信号稀释掉 5 天；
  * DiD = mean(resid | 断水窗口内) − mean(resid | 断水前)，逐株算，再对株汇总；
  * 置信区间用**按株位聚类的 bootstrap**（重采样 unit 是株位 code，不是观测），
    因为 W2_A4 与 W3_A4 共用同一个对照 W1_A4，观测之间不独立；
  * 株位数只有 11 个，CI 必然偏宽——如实报告，不粉饰。

未知/不做的：
  * 不做显著性宣称（n=11 cluster，任何 p 值都不可靠），只报点估计 + CI 区间；
  * 不控制天气（没有独立气象数据），只用同天同时刻配对来约掉天气。

用法：
  python scripts/did_analysis.py --features data/thermal_features_all.csv
  python scripts/did_analysis.py --features ... --feature p90 --boot 2000
  python scripts/did_analysis.py --features ... --out data/did_result.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
WINDOW_START = pd.Timestamp("2025-09-03 11:00")
WINDOW_END = pd.Timestamp("2025-09-15 18:35")
# 面板株位（`sampling_candidates.csv` 里 hard_confound_flags 为空且三床同名）
PANEL_LAYOUT = ("A4", "C4", "D3", "D5", "G4", "G6", "I8", "L5", "M8", "N5", "O6")


def split_id(plant: str) -> tuple[str, str]:
    bed, _, code = plant.partition("_")
    return bed, code


def load_features(path: Path) -> pd.DataFrame:
    f = pd.read_csv(path, dtype={"slot": str})
    f["bed"], f["code"] = zip(*f["plant"].map(split_id))
    f["dt"] = pd.to_datetime(
        f["date"].str.replace("_", "-") + " " + f["slot"].str.zfill(2) + ":00")
    f["phase"] = np.where(f["dt"] < WINDOW_START, "pre",
                          np.where(f["dt"] <= WINDOW_END, "in", "post"))
    return f


def build_pairs(f: pd.DataFrame, value: str) -> pd.DataFrame:
    """生成 (code, date, slot) 上的 缺水 − 同株位对照 残差，带床标签。"""
    ctrl = f[f["bed"] == "W1"][["code", "date", "slot", "dt", "phase", value]]
    ctrl = ctrl.rename(columns={value: "ctrl"})
    # 同一 (code,date,slot) 若出现多个对照株（理论上每床每株位仅 1 株），取均值兜底
    ctrl = ctrl.groupby(["code", "date", "slot", "dt", "phase"], as_index=False)["ctrl"].mean()

    dfc = f[f["bed"].isin(["W2", "W3"])][
        ["plant", "bed", "code", "date", "slot", "dt", "phase", value]]
    m = dfc.merge(ctrl, on=["code", "date", "slot", "dt", "phase"], how="inner")
    m["resid"] = m[value] - m["ctrl"]
    return m


def slot_center(pairs: pd.DataFrame, mode: str) -> pd.DataFrame:
    """按时次分层。mode=pre 只用断水前估计中心（避免信息泄漏），all 用全期。"""
    src = pairs[pairs["phase"] == "pre"] if mode == "pre" else pairs
    stat = (src.groupby("slot")["resid"]
               .agg(center="median", scale=lambda s: 1.4826 * (s - s.median()).abs().median())
               .reset_index())
    stat["scale"] = stat["scale"].replace(0, np.nan)
    out = pairs.merge(stat, on="slot", how="left")
    out["resid_z"] = (out["resid"] - out["center"]) / out["scale"]
    return out


def did_point(pairs: pd.DataFrame, col: str) -> dict:
    pre = pairs.loc[pairs["phase"] == "pre", col].dropna()
    inn = pairs.loc[pairs["phase"] == "in", col].dropna()
    post = pairs.loc[pairs["phase"] == "post", col].dropna()
    if len(pre) < 3 or len(inn) < 3:
        return {"pre": float("nan"), "in": float("nan"), "did": float("nan"),
                "n_pre": len(pre), "n_in": len(inn)}
    return {"pre": float(pre.mean()), "in": float(inn.mean()),
            "did": float(inn.mean() - pre.mean()),
            "post": float(post.mean()) if len(post) >= 3 else float("nan"),
            "n_pre": int(len(pre)), "n_in": int(len(inn)), "n_post": int(len(post))}


def cluster_bootstrap(pairs: pd.DataFrame, col: str, n_boot: int,
                      seed: int = 20261117) -> dict:
    """按 code 聚类重采样（同一株位的所有观测一起抽），给 DiD 的百分位区间。"""
    codes = sorted(pairs["code"].unique())
    if len(codes) < 3:
        return {"lo": float("nan"), "hi": float("nan"), "n_clusters": len(codes), "draws": 0}
    groups = {c: g for c, g in pairs.groupby("code")}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(codes, size=len(codes), replace=True)
        samp = pd.concat([groups[c] for c in pick], ignore_index=True)
        d = did_point(samp, col)["did"]
        if np.isfinite(d):
            draws.append(d)
    if not draws:
        return {"lo": float("nan"), "hi": float("nan"), "n_clusters": len(codes), "draws": 0}
    arr = np.array(draws)
    return {"lo": float(np.percentile(arr, 2.5)), "hi": float(np.percentile(arr, 97.5)),
            "median": float(np.median(arr)), "n_clusters": len(codes), "draws": len(arr)}


def per_plant(pairs: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for (plant, bed, code), g in pairs.groupby(["plant", "bed", "code"]):
        d = did_point(g, col)
        # 避免用 "in" 当列名：itertuples 会把它当成关键字参数位置名
        rows.append({"plant": plant, "bed": bed, "code": code,
                     "pre": d["pre"], "in_val": d["in"], "did": d["did"],
                     "n_pre": d["n_pre"], "n_in": d["n_in"]})
    return pd.DataFrame(rows).sort_values("did", ascending=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path,
                    default=PROJECT / "data/thermal_features_all.csv")
    ap.add_argument("--feature", default="upper_mean",
                    help="温度特征列（upper_mean/p50/p75/p90）")
    ap.add_argument("--center", choices=("pre", "all"), default="pre",
                    help="时次中心怎么估：pre=只用断水前（默认，防信息泄漏）")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=PROJECT / "data/did_result.json")
    a = ap.parse_args()

    if not a.features.exists():
        print(f"缺特征表 {a.features}", file=sys.stderr)
        return 2
    f = load_features(a.features)
    print(f"特征表 {a.features.name}：{len(f)} 行 | {f['plant'].nunique()} 株 | "
          f"{f['code'].nunique()} 个株位 | 床 {sorted(f['bed'].unique())}")

    have_ctrl = set(f.loc[f["bed"] == "W1", "code"])
    have_def = set(f.loc[f["bed"].isin(["W2", "W3"]), "code"])
    usable = sorted(have_ctrl & have_def)
    print(f"有同株位对照的缺水株位：{len(usable)} 个 → {usable}")
    if not usable:
        print("\n[STOP] 没有任何株位同时具备对照与缺水株，DiD 无法成立。\n"
              "       当前数据只能做 THERMAL_UNSUPERVISED.md 那种弱配对（全体对照均值），"
              "不能做跨床同株位配对。", file=sys.stderr)
        return 3

    pairs = build_pairs(f, a.feature)
    pairs = pairs[pairs["code"].isin(usable)]
    if pairs.empty:
        print("[STOP] 配对后为空（同日同时次没有重叠）", file=sys.stderr)
        return 3

    centered = slot_center(pairs, a.center)
    print(f"\n配对观测 {len(pairs)} 个 | 覆盖 {pairs['date'].nunique()} 天 | "
          f"时次 {sorted(pairs['slot'].unique())}")

    raw = did_point(pairs, "resid")
    z = did_point(centered, "resid_z")
    ci_raw = cluster_bootstrap(pairs, "resid", a.boot)
    ci_z = cluster_bootstrap(centered, "resid_z", a.boot)

    print(f"\n=== 原始残差（℃）特征={a.feature} ===")
    print(f"  断水前 {raw['pre']:+.3f} (n={raw['n_pre']})  →  窗口内 {raw['in']:+.3f} "
          f"(n={raw['n_in']})  恢复后 "
          f"{raw.get('post', float('nan')):+.3f}")
    print(f"  DiD = {raw['did']:+.3f} ℃   95%CI [{ci_raw['lo']:+.3f}, {ci_raw['hi']:+.3f}] "
          f"(聚类 {ci_raw['n_clusters']}，{ci_raw['draws']} 次重采样)")
    print(f"\n=== 按时次标准化后（z，中心用{'断水前' if a.center=='pre' else '全期'}估计）===")
    print(f"  断水前 {z['pre']:+.3f}  →  窗口内 {z['in']:+.3f}")
    print(f"  DiD_z = {z['did']:+.3f}   95%CI [{ci_z['lo']:+.3f}, {ci_z['hi']:+.3f}]")

    ci_ok = (ci_raw["draws"] > 0 and np.isfinite(ci_raw["lo"])
             and np.isfinite(ci_raw["hi"]))
    if not ci_ok:
        # 聚类数 <3 时无法做有意义的按株位重采样，区间是 NaN。
        # 此时绝不能拿 NaN 去比较 0（nan>0 为 False 会被误判成"方向相反"）。
        ci_crosses_zero = None
        verdict = (f"区间不可估计（聚类 {ci_raw['n_clusters']}<3，0 次重采样）；"
                   f"点估计 DiD={raw['did']:+.3f}℃ 仅供个案参考，不能声称显著性")
    else:
        ci_crosses_zero = bool(ci_raw["lo"] <= 0 <= ci_raw["hi"])
        verdict = ("区间跨 0，不能声称存在一致的升温响应" if ci_crosses_zero
                   else ("区间整体为正，方向一致" if ci_raw["lo"] > 0
                         else "区间整体为负，方向与假设相反"))
    print(f"\n  判定：{verdict}")

    pp = per_plant(pairs, "resid")
    print(f"\n=== 逐株 DiD（{len(pp)} 株，按大小排序）===")
    print(f"  {'株':8} {'code':5} {'pre':>8} {'in':>8} {'DiD':>8} {'n':>10}")
    for r in pp.itertuples():
        print(f"  {r.plant:8} {r.code:5} {r.pre:+8.2f} {r.in_val:+8.2f} {r.did:+8.2f} "
              f"{r.n_pre:4d}/{r.n_in:<4d}")
    n_pos = int((pp["did"] > 0).sum())
    print(f"\n  {n_pos}/{len(pp)} 株 DiD 为正")

    daily = (centered.groupby(["date", "phase"])["resid"]
                     .agg(["mean", "count"]).reset_index())
    print(f"\n=== 逐日残差（跨株位均值）===")
    for r in daily.itertuples():
        bar = "#" * int(max(0.0, r.mean + 3) * 2)
        print(f"  {r.date} [{r.phase:4}] {r.mean:+7.2f}℃ n={int(r.count):3d} {bar}")

    res = {"feature": a.feature, "center_mode": a.center,
           "window": [str(WINDOW_START), str(WINDOW_END)],
           "usable_codes": usable,
           "n_pairs": int(len(pairs)), "n_days": int(pairs["date"].nunique()),
           "raw": raw, "ci_raw": ci_raw, "z": z, "ci_z": ci_z,
           "ci_crosses_zero": ci_crosses_zero, "verdict": verdict,
           "per_plant": pp.to_dict(orient="records"),
           "daily": daily.to_dict(orient="records")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] 结果写入 {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
