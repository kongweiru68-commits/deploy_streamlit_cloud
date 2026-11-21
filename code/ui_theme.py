#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""桌面端统一视觉主题：深色侧边栏 + 浅色内容区的 SaaS 仪表盘风格（纯 CSS，无额外依赖）。

设计参考：Streamlit 官方 theming 指南与常见 SaaS 管理后台模板
（深石板侧栏、圆角白卡片、柔和投影、胶囊式分页）。
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

# 色板：主绿 + 深石板 + 中性灰
PALETTE = {
    "primary": "#15803D",
    "primary_dark": "#166534",
    "ink": "#0F172A",
    "text": "#1E293B",
    "muted": "#64748B",
    "line": "#E2E8F0",
    "bg": "#F4F7F5",
    "card": "#FFFFFF",
    "side": "#0F2A24",
    "side_text": "#CFE3D9",
}

CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"], .stApp { font-family: 'Inter', 'Microsoft YaHei', 'PingFang SC', sans-serif; }
  #MainMenu, footer { visibility: hidden; }
  header[data-testid="stHeader"] { background: transparent; height: 2.6rem; }
  .stApp { background: #F4F7F5; }
  .block-container { padding-top: 3.2rem; padding-bottom: 2.4rem; max-width: 1480px; }

  /* ---------- 侧边栏 ---------- */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F2A24 0%, #12362C 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
  }
  section[data-testid="stSidebar"] * { color: #CFE3D9; }
  section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 { color: #FFFFFF; font-weight: 700; letter-spacing: .3px; }
  section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.10); }
  section[data-testid="stSidebar"] div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.10);
    border-left: 3px solid #4ADE80; box-shadow: none;
  }
  section[data-testid="stSidebar"] div[data-testid="stMetricValue"] { color: #FFFFFF; }
  section[data-testid="stSidebar"] div[data-testid="stMetricLabel"] p { color: #A7C4B6; }
  section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.08); color: #E8F5EE; border: 1px solid rgba(255,255,255,0.16);
  }
  section[data-testid="stSidebar"] .stButton > button:hover { background: rgba(255,255,255,0.16); border-color: #4ADE80; }


  /* 侧栏输入控件：深底上的浅色控件，文字必须为深色 */
  section[data-testid="stSidebar"] div[data-baseweb="select"] > div { background: #FFFFFF; }
  section[data-testid="stSidebar"] div[data-baseweb="select"] *,
  section[data-testid="stSidebar"] div[data-baseweb="input"] input { color: #0F172A !important; -webkit-text-fill-color: #0F172A; }
  section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] label p { color: #E3F2EA !important; font-weight: 600; }

  /* ---------- 顶部品牌横幅 ---------- */
  .hero-hdr {
    background: linear-gradient(120deg, #0F2A24 0%, #166534 60%, #22A053 100%);
    padding: 18px 26px; border-radius: 16px; color: #FFFFFF; margin-bottom: 18px;
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
    box-shadow: 0 8px 22px rgba(15, 42, 36, 0.16);
  }
  .hero-hdr h2 { margin: 0; font-size: 1.4rem; font-weight: 800; color: #FFFFFF; letter-spacing: .4px; }
  .hero-hdr > span { font-size: .84rem; color: #DCF2E5; background: rgba(255,255,255,0.14);
                   padding: 5px 14px; border-radius: 999px; }
  .hero-hdr h2 span, .hero-hdr h2 a { background: none !important; padding: 0 !important; }
  [data-testid="stHeaderActionElements"] { display: none !important; }

  /* ---------- 页面标题条（每个模块顶部） ---------- */
  .page-head { display: flex; align-items: flex-start; gap: 14px; margin: 4px 0 16px; }
  .page-head .ph-ico {
    flex: none; width: 44px; height: 44px; border-radius: 12px; font-size: 1.35rem;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #DCFCE7, #BBF7D0); border: 1px solid #A7E8BE;
  }
  .page-head .ph-t { font-size: 1.28rem; font-weight: 800; color: #0F172A; line-height: 1.25; }
  .page-head .ph-d { font-size: .88rem; color: #64748B; margin-top: 3px; line-height: 1.55; }

  /* ---------- 分区小标题 ---------- */
  .sec-hdr {
    border-left: 4px solid #15803D; padding: 3px 0 3px 12px;
    font-size: 1.04rem; font-weight: 700; color: #0F172A; margin: 16px 0 10px;
  }

  /* ---------- 指标卡 ---------- */
  div[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #15803D;
    padding: 13px 16px; border-radius: 12px; box-shadow: 0 1px 3px rgba(15,23,42,.05);
    transition: box-shadow .2s ease, transform .2s ease;
  }
  div[data-testid="stMetric"]:hover { box-shadow: 0 8px 18px rgba(15,23,42,.08); transform: translateY(-1px); }
  div[data-testid="stMetricLabel"] p { color: #64748B; font-size: .82rem; font-weight: 600; }
  div[data-testid="stMetricValue"] { color: #0F172A; font-weight: 800; }

  /* ---------- 主分页：胶囊导航 ---------- */
  .stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: #FFFFFF; padding: 6px; border-radius: 14px;
    border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(15,23,42,.05);
  }
  .stTabs [data-baseweb="tab"] {
    height: 40px; border-radius: 10px; padding: 0 16px; background: transparent;
    color: #475569; font-weight: 600; border: none;
  }
  .stTabs [data-baseweb="tab"]:hover { background: #F1F5F3; color: #0F172A; }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #15803D, #22A053) !important; color: #FFFFFF !important;
    box-shadow: 0 4px 10px rgba(21,128,61,.28);
  }
  .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

  /* ---------- 按钮 ---------- */
  .stButton > button, .stDownloadButton > button {
    border-radius: 10px; font-weight: 600; border: 1px solid #CBD5E1; transition: all .18s ease;
  }
  .stButton > button:hover, .stDownloadButton > button:hover { border-color: #15803D; color: #15803D; }
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #15803D, #22A053); border: none; color: #FFFFFF;
    box-shadow: 0 4px 12px rgba(21,128,61,.25);
  }
  .stButton > button[kind="primary"]:hover { filter: brightness(1.06); color: #FFFFFF; }

  /* ---------- 输入控件 ---------- */
  div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, textarea {
    border-radius: 10px !important; background: #FFFFFF;
  }
  div[data-testid="stFileUploader"] section {
    border: 1.5px dashed #94C9A8; border-radius: 12px; background: #F8FCF9;
  }

  /* ---------- 表格 / 图表 / 折叠面板 ---------- */
  div[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0; border-radius: 12px; overflow: hidden;
    box-shadow: 0 1px 3px rgba(15,23,42,.05); background: #FFFFFF;
  }
  div[data-testid="stExpander"] {
    border: 1px solid #E2E8F0; border-radius: 12px; background: #FFFFFF;
    box-shadow: 0 1px 3px rgba(15,23,42,.04);
  }
  div[data-testid="stExpander"] summary { font-weight: 600; color: #0F172A; }
  div[data-testid="stImage"] img { border-radius: 10px; }
  div[data-testid="stAlert"] { border-radius: 12px; }

  /* ---------- 登录门户 ---------- */
  .portal-box {
    background: linear-gradient(155deg, #0F2A24 0%, #166534 100%);
    color: #FFFFFF; padding: 30px 28px; border-radius: 18px;
    box-shadow: 0 12px 28px rgba(15, 42, 36, 0.18);
  }
  .portal-box h3 { margin: 0 0 8px; font-size: 1.5rem; color: #FFFFFF; font-weight: 800; }
  .portal-sub { font-size: .92rem; color: #D7EDE2; line-height: 1.65; margin-bottom: 18px; }
  .portal-stat {
    background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.16);
    border-radius: 12px; padding: 12px 14px; margin-bottom: 10px;
  }
  .portal-stat .k { font-size: .78rem; color: #BFE0CE; }
  .portal-stat .v { font-size: 1.0rem; font-weight: 700; color: #FFFFFF; margin-top: 2px; }

  /* ---------- 侧栏账号卡 ---------- */
  .user-card {
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.14);
    border-radius: 12px; padding: 12px 14px; margin-bottom: 10px;
  }
  .user-card .u-name { font-size: 1.0rem; font-weight: 700; color: #FFFFFF; }
  .user-card .u-role {
    display: inline-block; margin-top: 5px; padding: 2px 10px; border-radius: 999px;
    font-size: .74rem; font-weight: 600; background: rgba(74,222,128,.18); color: #86EFAC;
  }

  /* ---------- 驾驶舱 KPI 卡 ---------- */
  .kpi-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-bottom: 18px; }
  .kpi {
    position: relative; overflow: hidden; border-radius: 16px; padding: 16px 18px; color: #FFFFFF;
    box-shadow: 0 8px 20px rgba(15,23,42,.10);
  }
  .kpi::after { content: ""; position: absolute; right: -26px; top: -26px; width: 96px; height: 96px;
                border-radius: 50%; background: rgba(255,255,255,.14); }
  .kpi .k-l { font-size: .8rem; opacity: .9; font-weight: 600; }
  .kpi .k-v { font-size: 1.9rem; font-weight: 800; margin-top: 4px; line-height: 1.15; }
  .kpi .k-s { font-size: .76rem; opacity: .88; margin-top: 4px; }
  .kpi .k-i { position: absolute; right: 16px; bottom: 12px; font-size: 1.6rem; opacity: .85; z-index: 1; }
  .kpi.g1 { background: linear-gradient(135deg, #15803D, #22C55E); }
  .kpi.g2 { background: linear-gradient(135deg, #C2410C, #F97316); }
  .kpi.g3 { background: linear-gradient(135deg, #0E7490, #22D3EE); }
  .kpi.g4 { background: linear-gradient(135deg, #4338CA, #818CF8); }

  /* ---------- 通用白卡片 ---------- */
  .panel { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; padding: 16px 18px;
           box-shadow: 0 1px 3px rgba(15,23,42,.05); margin-bottom: 14px; }
  .panel .p-t { font-size: .98rem; font-weight: 700; color: #0F172A; margin-bottom: 10px;
                display: flex; justify-content: space-between; align-items: center; }
  .panel .p-t small { font-weight: 500; color: #94A3B8; font-size: .76rem; }

  /* ---------- 风险榜 ---------- */
  .risk-row { display: flex; align-items: center; gap: 10px; padding: 8px 0; border-bottom: 1px dashed #E2E8F0; }
  .risk-row:last-child { border-bottom: none; }
  .risk-rank { flex: none; width: 24px; height: 24px; border-radius: 8px; background: #F1F5F9; color: #475569;
               font-size: .78rem; font-weight: 800; display: flex; align-items: center; justify-content: center; }
  .risk-row:nth-child(-n+3) .risk-rank { background: #FEE2E2; color: #B91C1C; }
  .risk-name { width: 74px; font-weight: 700; color: #0F172A; font-size: .9rem; }
  .risk-bar { flex: 1; height: 8px; border-radius: 99px; background: #EEF2F0; overflow: hidden; }
  .risk-bar > i { display: block; height: 100%; border-radius: 99px; }
  .risk-val { width: 62px; text-align: right; font-weight: 700; color: #334155; font-size: .86rem; }
  .tag { display: inline-block; padding: 2px 9px; border-radius: 99px; font-size: .72rem; font-weight: 700; }
  .tag.t0 { background: #DCFCE7; color: #166534; } .tag.t1 { background: #FEF3C7; color: #92400E; }
  .tag.t2 { background: #FFEDD5; color: #C2410C; } .tag.t3 { background: #FEE2E2; color: #B91C1C; }

  /* ---------- 功能卡 / 流程步骤 ---------- */
  .feat-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
  .feat { background: #F8FBF9; border: 1px solid #E1ECE5; border-radius: 12px; padding: 12px 13px; }
  .feat .f-i { font-size: 1.25rem; } .feat .f-t { font-weight: 700; color: #0F172A; margin-top: 4px; font-size: .9rem; }
  .feat .f-d { color: #64748B; font-size: .78rem; margin-top: 2px; line-height: 1.5; }
  .flow { display: flex; align-items: stretch; gap: 8px; flex-wrap: wrap; }
  .flow .step { flex: 1; min-width: 140px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
                padding: 12px 14px; position: relative; }
  .flow .step .n { width: 24px; height: 24px; border-radius: 50%; background: #15803D; color: #FFF; font-weight: 800;
                   font-size: .78rem; display: inline-flex; align-items: center; justify-content: center; }
  .flow .step .s-t { font-weight: 700; color: #0F172A; margin: 6px 0 2px; font-size: .9rem; }
  .flow .step .s-d { color: #64748B; font-size: .78rem; line-height: 1.5; }
  .status-row { display: flex; justify-content: space-between; padding: 7px 0; font-size: .86rem;
                border-bottom: 1px dashed #E2E8F0; color: #475569; }
  .status-row:last-child { border-bottom: none; }
  .dot-ok { color: #16A34A; font-weight: 700; }
</style>
"""


def inject() -> None:
    """注入全局主题样式，并统一 matplotlib 图表风格。"""
    st.markdown(CSS, unsafe_allow_html=True)
    plt.rcParams.update({
        "axes.facecolor": "#FFFFFF",
        "figure.facecolor": "#FFFFFF",
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": "#334155",
        "xtick.color": "#64748B",
        "ytick.color": "#64748B",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#E2E8F0",
        "axes.titleweight": "bold",
    })


def page_head(icon: str, title: str, desc: str = "") -> None:
    """渲染模块顶部标题条（仅图标 + 标题），详细说明收进右侧 💡 弹出气泡。"""
    c_title, c_help = st.columns([1.0, 0.11], gap="small")
    with c_title:
        st.markdown(
            f'<div class="page-head" style="align-items:center;margin:2px 0 12px;">'
            f'<div class="ph-ico">{icon}</div>'
            f'<div class="ph-t">{title}</div></div>',
            unsafe_allow_html=True,
        )
    if desc:
        with c_help:
            with st.popover("💡", help="点击查看本模块说明"):
                st.markdown(f"**{icon} {title} · 模块说明**")
                st.caption(desc)


def section(title: str) -> None:
    """渲染带绿色左边框的分区小标题。"""
    st.markdown(f'<div class="sec-hdr">{title}</div>', unsafe_allow_html=True)


def kpi_cards(items: list[dict]) -> None:
    """渲染驾驶舱 KPI 渐变卡片。items: [{label, value, sub, icon, tone(g1~g4)}]。"""
    html = "".join(
        f'<div class="kpi {it.get("tone", "g1")}"><div class="k-l">{it["label"]}</div>'
        f'<div class="k-v">{it["value"]}</div><div class="k-s">{it.get("sub", "")}</div>'
        f'<div class="k-i">{it.get("icon", "")}</div></div>'
        for it in items
    )
    st.markdown(f'<div class="kpi-grid">{html}</div>', unsafe_allow_html=True)


def panel(title: str, body_html: str, hint: str = "") -> None:
    """渲染带标题的白色信息面板。"""
    small = f"<small>{hint}</small>" if hint else ""
    st.markdown(f'<div class="panel"><div class="p-t">{title}{small}</div>{body_html}</div>', unsafe_allow_html=True)


def risk_list_html(rows: list[dict]) -> str:
    """生成风险榜 HTML。rows: [{name, val, pct, color, grade, tag}]。"""
    out = []
    for i, r in enumerate(rows, 1):
        out.append(
            f'<div class="risk-row"><div class="risk-rank">{i}</div><div class="risk-name">{r["name"]}</div>'
            f'<div class="risk-bar"><i style="width:{r["pct"]:.0f}%;background:{r["color"]}"></i></div>'
            f'<div class="risk-val">{r["val"]:+.2f}℃</div><span class="tag t{r["grade"]}">{r["tag"]}</span></div>'
        )
    return "".join(out)
