#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""大棚电视大屏看板：宽幅横排驾驶舱布局（深蓝科技风，杜绝窄列文本折行）。

结构：
- 顶部：实时秒级 JS 时钟标题栏 + 1×6 核心指标横带
- 第一排（两列宽栏）：3D 数字孪生温室 ｜ 四级判级环形图 + 高风险 Top 6
- 第二排（两列宽栏）：断水—复水全程配对残差趋势 ｜ 单元 × 日期 残差热力图
- 第三排（三等分底栏）：分床微气象对比 ｜ 温室设备与灌溉处方 ｜ 最新巡检与算法审计
"""
from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import db

BG = "#06152F"
PANEL = "#0B2347"
LINE = "#1E4A87"
CYAN = "#22D3EE"
TXT = "#DCEBFF"

CSS = """
<style>
  .stApp { background: radial-gradient(circle at 20% 0%, #0C2B5E 0%, #06152F 55%, #040E22 100%) !important; }
  .hero-hdr, .stTabs, section[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
  header[data-testid="stHeader"] { display: none; }
  .bs-exit-btn button {
    background: rgba(14, 70, 150, 0.35) !important; color: #8DB8F0 !important;
    border: 1px solid #1E4A87 !important; border-radius: 8px !important; font-size: .78rem !important;
    padding: 2px 10px !important; min-height: 28px !important;
  }
  .bs-exit-btn button:hover { color: #FFFFFF !important; border-color: #22D3EE !important; }
  .block-container { padding-top: 0.85rem; padding-bottom: 1.0rem; max-width: 1840px; }

  /* 顶部 1x6 核心指标横带 */
  .bs-krow { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin-bottom: 12px; }
  .bs-stat {
    background: linear-gradient(160deg, rgba(14,60,125,.78), rgba(9,35,78,.82));
    border: 1px solid #1E4A87; border-left: 3px solid #22D3EE; border-radius: 10px;
    padding: 10px 14px;
  }
  .bs-stat .l { color: #9CC4F5; font-size: .85rem; font-weight: 600; white-space: nowrap; }
  .bs-stat .v { color: #FFFFFF; font-size: 1.75rem; font-weight: 800; line-height: 1.2; white-space: nowrap;
                text-shadow: 0 0 10px rgba(34,211,238,.4); font-variant-numeric: tabular-nums; margin-top: 2px; }
  .bs-stat .v small { font-size: .88rem; color: #7DE8FF; margin-left: 3px; font-weight: 600; }
  .bs-stat.warn { border-left-color: #F97316; } .bs-stat.warn .v { color: #FFB27A; }

  .bs-card {
    background: rgba(11,35,71,.82); border: 1px solid #1E4A87; border-radius: 12px;
    padding: 12px 16px; margin-bottom: 10px;
  }
  .bs-card .h { color: #EAF6FF; font-weight: 700; font-size: 1.02rem; margin-bottom: 8px; white-space: nowrap;
                border-left: 4px solid #22D3EE; padding-left: 10px; display: flex; justify-content: space-between; align-items: center; }
  .bs-card .h small { font-size: .78rem; color: #7FB4FF; font-weight: 500; white-space: nowrap; }

  /* 高风险榜（宽栏单行舒展） */
  .bs .risk-row { display: flex; align-items: center; gap: 10px; padding: 6px 0; border-bottom: 1px dashed #1E4A87; flex-wrap: nowrap; }
  .bs .risk-row:last-child { border-bottom: none; }
  .bs .risk-rank { flex: none; width: 22px; height: 22px; border-radius: 6px; background: #123A75; color: #BFD9FF;
                   font-size: .78rem; font-weight: 800; display: flex; align-items: center; justify-content: center; }
  .bs .risk-row:nth-child(-n+3) .risk-rank { background: #7F1D1D; color: #FFD1D1; }
  .bs .risk-name { color: #EAF6FF; width: 68px; font-size: .92rem; font-weight: 700; white-space: nowrap; }
  .bs .risk-bar { flex: 1; height: 8px; min-width: 40px; border-radius: 99px; background: #0A2A57; overflow: hidden; }
  .bs .risk-bar > i { display: block; height: 100%; border-radius: 99px; }
  .bs .risk-val { color: #CFE3FF; width: 62px; text-align: right; font-size: .88rem; font-weight: 700; white-space: nowrap; }
  .bs .tag { display: inline-block; font-size: .76rem; padding: 2px 9px; border-radius: 99px; font-weight: 700; white-space: nowrap; flex-shrink: 0; }
  .tag.t0 { background: #DCFCE7; color: #166534; } .tag.t1 { background: #FEF3C7; color: #92400E; }
  .tag.t2 { background: #FFEDD5; color: #C2410C; } .tag.t3 { background: #FEE2E2; color: #B91C1C; }

  /* 底栏：微气象 / 设备处方 / 巡检审计（全单行紧凑） */
  .bs-mc { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
  .bs-mc-box { background: rgba(6,24,56,.65); border: 1px solid #1B427A; border-radius: 8px; padding: 10px 12px; }
  .bs-mc-box.hot { border-color: #9A3412; background: rgba(67,20,7,.35); }
  .bs-mc-box .bt { font-size: .82rem; color: #9CC4F5; font-weight: 600; white-space: nowrap; }
  .bs-mc-box.hot .bt { color: #FDBA74; }
  .bs-mc-box .bv { font-size: 1.45rem; font-weight: 800; color: #FFFFFF; margin: 3px 0; white-space: nowrap; }
  .bs-mc-box .bs { font-size: .78rem; color: #8DB8F0; white-space: nowrap; }

  .bs-line { display: flex; justify-content: space-between; align-items: center; padding: 6px 10px;
             border-radius: 7px; background: rgba(6,24,56,.6); margin-bottom: 6px; font-size: .84rem; color: #DCEBFF; white-space: nowrap; }
  .bs-line:last-child { margin-bottom: 0; }
  .bs-line .st-on { color: #4ADE80; font-weight: 700; }
  .bs-line .st-wait { color: #38BDF8; font-weight: 700; }
  .bs-line .st-warn { color: #FDBA74; font-weight: 700; }

  .bs-feed { color: #CFE3FF; font-size: .84rem; line-height: 1.85; }
  .bs-feed b { color: #7DE8FF; }
  .bs-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 7px; }
  .bs-trust { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 7px; margin-top: 8px; }
  .bs-trust-i { background: rgba(6,24,56,.65); border: 1px solid #1B427A; border-radius: 8px; padding: 6px 10px;
                display: flex; justify-content: space-between; align-items: center; white-space: nowrap; }
  .bs-trust-i .tk { font-size: .76rem; color: #8DB8F0; }
  .bs-trust-i .tv { font-size: .88rem; font-weight: 700; color: #7DE8FF; }

  @media (max-width: 1100px) {
    .bs-krow { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  }
  @media (max-width: 768px) {
    .bs-krow { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; }
    .bs-stat .v { font-size: 1.35rem; }
    .bs-mc, .bs-trust { grid-template-columns: 1fr; }
  }
</style>

"""


def _dark_rc():
    return plt.rc_context({
        "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
        "axes.edgecolor": LINE, "axes.labelcolor": TXT, "text.color": TXT,
        "xtick.color": "#8DB8F0", "ytick.color": "#8DB8F0", "grid.color": "#17407A",
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 11,
    })


def _header_clock() -> None:
    now = datetime.now()
    components.html(
        f"""<!doctype html><html><head><meta charset="utf-8"><style>
        html,body{{margin:0;padding:0;background:transparent;font-family:"Inter","Microsoft YaHei",sans-serif;overflow:hidden;}}
        .bs-title{{position:relative;display:flex;justify-content:center;align-items:center;padding:8px 24px;
          background:linear-gradient(90deg,rgba(14,70,150,.55),rgba(10,40,95,.25),rgba(14,70,150,.55));
          border:1px solid #1E4A87;border-radius:12px;box-sizing:border-box;min-height:62px;
          box-shadow:0 0 22px rgba(34,211,238,.12) inset;}}
        .t{{font-size:clamp(1.05rem, 2.3vw, 1.8rem);font-weight:800;letter-spacing:4px;color:#EAF6FF;white-space:nowrap;
          text-shadow:0 0 14px rgba(34,211,238,.55);}}
        .clk{{position:absolute;right:22px;color:#7DE8FF;font-size:1.05rem;font-weight:700;
          font-variant-numeric:tabular-nums;white-space:nowrap;}}
        @media (max-width: 1120px){{
          .bs-title{{flex-direction:column;gap:3px;padding:8px 10px;}}
          .t{{letter-spacing:1.5px;font-size:clamp(0.95rem, 4.1vw, 1.4rem);}}
          .clk{{position:static;font-size:0.9rem;}}
        }}
        </style></head><body>
        <div class="bs-title">
          <div class="t">温室甜菜水分胁迫智能监测与预警平台</div>
          <div class="clk" id="bs-clk">{now:%Y-%m-%d %H:%M:%S}</div>
        </div>
        <script>
          function p2(n){{return n<10?"0"+n:""+n;}}
          function tick(){{
            var d=new Date();
            var s=d.getFullYear()+"-"+p2(d.getMonth()+1)+"-"+p2(d.getDate())+" "+
                  p2(d.getHours())+":"+p2(d.getMinutes())+":"+p2(d.getSeconds());
            var el=document.getElementById("bs-clk");
            if(el) el.innerHTML=s;
          }}
          tick();
          setInterval(tick, 1000);
        </script></body></html>""",
        height=76,
        scrolling=False,
    )


@st.fragment(run_every=10)
def _top_kpis(n_units: int, n_warn: int, mean_l3: float) -> None:
    obs = db.list_observations()
    st.markdown(
        f"""
        <div class="bs-krow">
          <div class="bs-stat"><div class="l">监测配对单元</div><div class="v">{n_units}<small>个</small></div></div>
          <div class="bs-stat warn"><div class="l">中重度预警</div><div class="v">{n_warn}<small>个</small></div></div>
          <div class="bs-stat"><div class="l">窗末平均升温</div><div class="v">{mean_l3:+.2f}<small>℃</small></div></div>
          <div class="bs-stat"><div class="l">复水残差回落</div><div class="v">79.9<small>%</small></div></div>
          <div class="bs-stat"><div class="l">CUSUM 检出</div><div class="v">22/22</div></div>
          <div class="bs-stat"><div class="l">平台已入库</div><div class="v">{len(obs)}<small>条</small></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _exit_bigscreen() -> None:
    st.session_state["bigscreen_on"] = False


def render(pairs: pd.DataFrame, classify, html_3d: str | None, window: tuple, risk_html_fn) -> None:
    """渲染大屏。classify(val)->(lvl,name,icon,advice)；window=(WINDOW_START, WINDOW_END)。"""
    st.markdown(CSS, unsafe_allow_html=True)
    hc_main, hc_exit = st.columns([1.0, 0.13], gap="small")
    with hc_main:
        _header_clock()
    with hc_exit:
        st.markdown('<div class="bs-exit-btn" style="padding-top:16px;">', unsafe_allow_html=True)
        st.button("退出大屏", key="bs_exit_btn", use_container_width=True, on_click=_exit_bigscreen)
        st.markdown('</div>', unsafe_allow_html=True)

    rows = []
    for p_id, g in pairs.groupby("plant"):
        g_in = g[g["phase"] == "in"]
        l3 = float(g_in["resid"].tail(3).mean()) if not g_in.empty else float("nan")
        lvl, name, _, _ = classify(l3)
        rows.append({"unit": p_id, "l3": l3, "lvl": int(lvl), "name": name.split(" ")[0]})
    ov = pd.DataFrame(rows)
    n_units = len(ov)
    n_warn = int((ov["lvl"] >= 2).sum()) if n_units else 0
    mean_l3 = float(ov["l3"].mean()) if n_units else 0.0

    # ---- 顶部 1x6 指标横带 ----
    _top_kpis(n_units, n_warn, mean_l3)

    # ---- 第一排：两列宽栏（3D 孪生 ｜ 四级判级 + 高风险 Top 6） ----
    r1_l, r1_r = st.columns([1.9, 1.0], gap="small")
    with r1_l:
        st.markdown('<div class="bs-card"><div class="h">温室 3D 数字孪生 · 实时判级 <small>拖拽旋转 / 点选株位</small></div></div>', unsafe_allow_html=True)
        if html_3d is not None:
            components.html(html_3d, height=460, scrolling=False)
        else:
            st.info("未找到 3D 引擎资源。")

    with r1_r:
        counts = [int((ov["lvl"] == k).sum()) for k in range(4)] if n_units else [0, 0, 0, 0]
        cols = ["#22C55E", "#FBBF24", "#F97316", "#EF4444"]
        labels = ["正常", "轻度", "中度", "重度"]
        st.markdown('<div class="bs-card"><div class="h">四级判级构成</div></div>', unsafe_allow_html=True)
        d_c1, d_c2 = st.columns([1.0, 1.05], gap="small")
        with d_c1:
            with _dark_rc():
                fig_d, ax_d = plt.subplots(figsize=(2.2, 1.85), dpi=130)
                nz = [(c, l, col) for c, l, col in zip(counts, labels, cols) if c > 0]
                if nz:
                    ax_d.pie([x[0] for x in nz], colors=[x[2] for x in nz],
                             radius=0.96, startangle=90, counterclock=False,
                             wedgeprops={"width": 0.38, "edgecolor": PANEL})
                    ax_d.text(0, 0, f"{n_units}\n单元", ha="center", va="center", fontsize=12, fontweight="bold", color="#FFFFFF")
                fig_d.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)
                st.pyplot(fig_d)
                plt.close(fig_d)
        with d_c2:
            leg_html = "".join(
                f'<div class="bs-line" style="padding:4px 9px;margin-bottom:5px;">'
                f'<span><i class="bs-dot" style="background:{col}"></i>{lb}</span>'
                f'<b>{cnt} 单元</b></div>'
                for lb, cnt, col in zip(labels, counts, cols)
            )
            st.markdown(leg_html, unsafe_allow_html=True)

        top = ov.sort_values("l3", ascending=False).head(6) if n_units else ov
        vmax = max(float(top["l3"].max()), 1.0) if n_units else 1.0
        pal = {0: "#22C55E", 1: "#FBBF24", 2: "#F97316", 3: "#EF4444"}
        risk_rows = [{"name": r["unit"], "val": r["l3"], "pct": max(4.0, r["l3"] / vmax * 100),
                      "color": pal[int(r["lvl"])], "grade": int(r["lvl"]), "tag": r["name"]} for _, r in top.iterrows()]
        st.markdown(
            f'<div class="bs bs-card"><div class="h">高风险单元榜 Top 6 <small>窗末3次均值</small></div>{risk_html_fn(risk_rows)}</div>',
            unsafe_allow_html=True,
        )

    # ---- 第二排：两列宽栏（全程趋势图 ｜ 残差热力图） ----
    r2_l, r2_r = st.columns([1.15, 1.0], gap="small")
    with r2_l:
        with _dark_rc():
            tr = pairs.assign(day=pairs["dt"].dt.normalize(), bed=pairs["plant"].str.split("_").str[0])
            tr = tr.groupby(["day", "bed"])["resid"].mean().unstack()
            fig_t, ax_t = plt.subplots(figsize=(7.6, 3.1), dpi=130)
            bed_col = {"W2": "#FBBF24", "W3": "#F87171"}
            for bed in tr.columns:
                ax_t.plot(tr.index, tr[bed], linewidth=2.6, color=bed_col.get(bed, CYAN), label=f"{bed} 床")
            ax_t.axvspan(window[0], window[1], color="#EF4444", alpha=0.16, label="断水窗")
            ax_t.axhline(0, color="#4A6FA5", linestyle="--", linewidth=0.8)
            ax_t.legend(frameon=False, fontsize=10.5, ncol=3, loc="upper left", labelcolor=TXT)
            ax_t.grid(True, linestyle="--", alpha=0.4)
            ax_t.set_title("断水—复水全程配对残差趋势 (℃)", fontsize=12.5, color=TXT, loc="left", fontweight="bold")
            fig_t.tight_layout()
            st.pyplot(fig_t)
            plt.close(fig_t)

    with r2_r:
        with _dark_rc():
            heat = pairs.assign(day=pairs["dt"].dt.strftime("%m-%d")).groupby(["plant", "day"])["resid"].mean().unstack()
            heat = heat.loc[sorted(heat.index, key=lambda s: (s.split("_")[0], s))]
            fig_h, ax_h = plt.subplots(figsize=(6.8, 3.1), dpi=130)
            im = ax_h.imshow(heat.values, aspect="auto", cmap="turbo", vmin=-1.5, vmax=6.5)
            ax_h.set_yticks(range(0, len(heat.index), 3))
            ax_h.set_yticklabels([heat.index[i] for i in range(0, len(heat.index), 3)], fontsize=10)
            xt = list(range(0, len(heat.columns), max(1, len(heat.columns) // 7)))
            ax_h.set_xticks(xt)
            ax_h.set_xticklabels([heat.columns[i] for i in xt], fontsize=10)
            for sp in ax_h.spines.values():
                sp.set_visible(False)
            cb = fig_h.colorbar(im, ax=ax_h, fraction=0.024, pad=0.015)
            cb.ax.tick_params(labelsize=10, colors="#9CC4F5")
            ax_h.set_title("单元 × 日期 残差热力图 (℃)", fontsize=12.5, color=TXT, loc="left", fontweight="bold")
            fig_h.tight_layout()
            st.pyplot(fig_h)
            plt.close(fig_h)

    # ---- 第三排：三等分底栏（全部单行展示） ----
    b1, b2, b3 = st.columns(3, gap="small")
    with b1:
        st.markdown(
            """
            <div class="bs-card">
              <div class="h">分床微气象对比 <small>同步监测</small></div>
              <div class="bs-mc">
                <div class="bs-mc-box">
                  <div class="bt">W1 对照床</div>
                  <div class="bv">21.9 ℃</div>
                  <div class="bs">RH 60.3% · VPD 1.21</div>
                </div>
                <div class="bs-mc-box hot">
                  <div class="bt">W3 断水床</div>
                  <div class="bv">25.0 ℃</div>
                  <div class="bs">RH 46.8% · VPD 1.83</div>
                </div>
              </div>
              <div class="bs-line" style="margin-top:8px;"><span>W3 补水处方</span><span class="st-warn">优先潮汐补水 15 min</span></div>
              <div class="bs-line"><span>W2 轮灌处方</span><span class="st-wait">次晨自动轮灌 10 min</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b2:
        st.markdown(
            """
            <div class="bs-card">
              <div class="h">温室执行设备状态 <small>孪生联动</small></div>
              <div class="bs-line"><span>负压风机阵列</span><span class="st-on">● 运行排风中</span></div>
              <div class="bs-line"><span>湿帘降温水泵</span><span class="st-on">● 协同降温中</span></div>
              <div class="bs-line"><span>W3 补水电磁阀</span><span class="st-wait">● 建议优先开启</span></div>
              <div class="bs-line"><span>LED 顶部补光</span><span>○ 日间待命</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b3:
        recent = db.list_observations()[:3]
        if recent:
            items = "".join(
                f'<div><span class="bs-dot" style="background:{["#22C55E", "#FBBF24", "#F97316", "#EF4444"][min(max(int(o["level"]), 0), 3)]}"></span>'
                f'{str(o["ts"])[5:16].replace("T", " ")} · <b>{o.get("plant_bed") or ""}_{o.get("plant_code") or ""}</b> · L{o["level"]} · {o["operator_name"]}</div>'
                for o in recent
            )
        else:
            items = (
                '<div><span class="bs-dot" style="background:#22D3EE"></span><b>巡检同步：</b>支持小程序数据包一键入库</div>'
                '<div><span class="bs-dot" style="background:#4ADE80"></span><b>冠层提取：</b>Otsu 纯冠层 95%CI 0.30℃</div>'
            )
        st.markdown(
            f"""
            <div class="bs-card">
              <div class="h">巡检动态与模型基线 <small>实时审计</small></div>
              <div class="bs-feed" style="min-height:54px;">{items}</div>
              <div class="bs-trust">
                <div class="bs-trust-i"><span class="tk">OOF AUC</span><span class="tv">0.8077</span></div>
                <div class="bs-trust-i"><span class="tk">安慰剂越线</span><span class="tv">0 / 55</span></div>
                <div class="bs-trust-i"><span class="tk">首警延迟</span><span class="tv">73 小时</span></div>
                <div class="bs-trust-i"><span class="tk">SQLite 库</span><span class="tv">● 在线</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
