#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v9 高级功能实验：在线因果变点早期预警引擎（Causal CUSUM / EWMA）。

严格防信息泄漏：
- 仅使用每个配对单元自身断水前（pre）前 7 天（08-12 ~ 08-18）的数据估计个人基准 (mu0, mad0)；
- 从 08-19 起逐时次严格因果在线递推，不看未来任何观测；
- 统计：
  1) 断水前验证期（08-19 ~ 09-03 11:00）的假阳性率（FPR）；
  2) 断水窗内（09-03 11:00 ~ 09-15 18:35）22/22 单元的首次报警延迟（小时）与检出率；
  3) 复水期（09-15 18:35 后）首次解除报警时间与解除率；
  4) W1 内部 55 对从未断水配对走完全相同在线流程的安慰剂假阳性率。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from did_analysis import WINDOW_END, WINDOW_START, build_pairs  # noqa: E402

DATA = PROJECT / "data"
V8_DIR = DATA / "v8_exp"
V9_DIR = DATA / "v9_exp"
V9_DIR.mkdir(parents=True, exist_ok=True)

CALIB_END = pd.Timestamp("2025-08-19 00:00")


def run_causal_cusum(series_df: pd.DataFrame, k_allow: float = 0.5, h_thresh: float = 4.5,
                     decay_reset: float = 0.5) -> pd.DataFrame:
    """对单条配对残差时间序列执行严格因果单边上行 CUSUM 预警。

    参数
    ----
    k_allow : 允许松弛量（标准差倍数），过滤日常微弱波动；
    h_thresh : 触发缺水报警的累积阈值（标准差倍数）；
    decay_reset : 超过 6 小时未观测时对累积量做衰减，防止隔夜累积。
    """
    df = series_df.sort_values("dt").copy()
    calib = df[df["dt"] < CALIB_END]["resid"].dropna()
    if len(calib) < 10:
        return pd.DataFrame()
    mu0 = float(calib.median())
    mad0 = float(1.4826 * (calib - mu0).abs().median())
    sd0 = max(mad0, 0.5)  # 防止极小方差导致阈值退化

    s_pos = 0.0
    prev_dt = None
    out_rows = []
    for _, row in df.iterrows():
        dt = row["dt"]
        if dt < CALIB_END:
            continue
        val = row["resid"]
        if not np.isfinite(val):
            continue
        if prev_dt is not None and (dt - prev_dt).total_seconds() > 6 * 3600:
            s_pos *= decay_reset
        z = (val - mu0) / sd0
        s_pos = max(0.0, s_pos + z - k_allow)
        prev_dt = dt
        out_rows.append({
            "plant": row.get("plant", ""),
            "bed": row.get("bed", ""),
            "dt": dt,
            "phase": row["phase"],
            "resid": float(val),
            "z_online": round(float(z), 3),
            "cusum_s": round(float(s_pos), 3),
            "alarm": int(s_pos >= h_thresh),
        })
    return pd.DataFrame(out_rows)


def main() -> int:
    df = pd.read_csv(V8_DIR / "features_v8_with_canopy.csv")
    df["code"] = df["plant_code"]
    df["slot"] = df["slot"].astype(str)
    df["dt"] = pd.to_datetime(df["date"].str.replace("_", "-") + " " + df["slot"].str.zfill(2) + ":00")
    df["phase"] = np.where(df["dt"] < WINDOW_START, "pre", np.where(df["dt"] <= WINDOW_END, "in", "post"))

    feature = "tm_canopy_p25"
    pairs = build_pairs(df, feature)

    all_traces = []
    unit_summary = []
    for plant, g in pairs.groupby("plant"):
        tr = run_causal_cusum(g, k_allow=0.5, h_thresh=4.5)
        if tr.empty:
            continue
        all_traces.append(tr)

        pre_val = tr[tr["phase"] == "pre"]
        in_val = tr[tr["phase"] == "in"]
        post_val = tr[tr["phase"] == "post"]

        pre_fpr = float(pre_val["alarm"].mean()) if len(pre_val) else float("nan")
        in_alarms = in_val[in_val["alarm"] == 1]
        first_alarm_dt = in_alarms["dt"].min() if not in_alarms.empty else pd.NaT
        delay_hours = float((first_alarm_dt - WINDOW_START).total_seconds() / 3600.0) if pd.notna(first_alarm_dt) else float("nan")

        # 复水后首次解除报警（alarm==0）的时间
        post_clear = post_val[post_val["alarm"] == 0]
        first_clear_dt = post_clear["dt"].min() if not post_clear.empty else pd.NaT
        clear_hours = float((first_clear_dt - WINDOW_END).total_seconds() / 3600.0) if pd.notna(first_clear_dt) else float("nan")

        unit_summary.append({
            "plant": plant,
            "bed": plant.split("_")[0],
            "pre_fpr": round(pre_fpr, 4),
            "detected_in_window": int(not in_alarms.empty),
            "detect_delay_hours": round(delay_hours, 2) if np.isfinite(delay_hours) else None,
            "in_alarm_rate": round(float(in_val["alarm"].mean()), 4),
            "clear_delay_hours": round(clear_hours, 2) if np.isfinite(clear_hours) else None,
            "post_end_alarm": int(post_val.iloc[-1]["alarm"]) if len(post_val) else None,
        })

    trace_df = pd.concat(all_traces, ignore_index=True)
    trace_df.to_csv(V9_DIR / "cusum_online_traces.csv", index=False)
    sum_df = pd.DataFrame(unit_summary)
    sum_df.to_csv(V9_DIR / "cusum_unit_summary.csv", index=False)

    # W1 内部 55 对安慰剂走完全相同的在线 CUSUM 预警器
    w1 = df[df["bed"] == "W1"].copy()
    codes = sorted(w1["code"].unique())
    plac_in_rates = []
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            a = w1[w1["code"] == codes[i]][["date", "slot", "dt", "phase", feature]]
            b = w1[w1["code"] == codes[j]][["date", "slot", "dt", "phase", feature]]
            m = a.merge(b, on=["date", "slot", "dt", "phase"], suffixes=("_a", "_b"))
            m["resid"] = m[f"{feature}_a"] - m[f"{feature}_b"]
            m["plant"] = f"{codes[i]}-{codes[j]}"
            m["bed"] = "W1_placebo"
            tr_p = run_causal_cusum(m, k_allow=0.5, h_thresh=4.5)
            if not tr_p.empty:
                in_p = tr_p[tr_p["phase"] == "in"]
                if len(in_p):
                    plac_in_rates.append(float(in_p["alarm"].mean()))

    summary = {
        "feature": feature,
        "calib_window": "2025-08-12 ~ 2025-08-18",
        "n_units": len(sum_df),
        "detected_units": int(sum_df["detected_in_window"].sum()),
        "detect_delay_median_hours": round(float(sum_df["detect_delay_hours"].median()), 2),
        "detect_delay_min_hours": round(float(sum_df["detect_delay_hours"].min()), 2),
        "detect_delay_max_hours": round(float(sum_df["detect_delay_hours"].max()), 2),
        "pre_fpr_mean": round(float(sum_df["pre_fpr"].mean()), 4),
        "in_alarm_rate_mean": round(float(sum_df["in_alarm_rate"].mean()), 4),
        "clear_delay_median_hours": round(float(sum_df["clear_delay_hours"].median()), 2),
        "post_end_zero_alarm_units": int((sum_df["post_end_alarm"] == 0).sum()),
        "placebo_n_pairs": len(plac_in_rates),
        "placebo_in_alarm_rate_median": round(float(np.median(plac_in_rates)), 4),
        "placebo_in_alarm_rate_max": round(float(np.max(plac_in_rates)), 4),
    }
    (V9_DIR / "cusum_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
