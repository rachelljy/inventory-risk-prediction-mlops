"""Shared CSS injected on every page."""
import os

import streamlit as st


def inject():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #f1f5f9 !important;
}
[data-testid="stSidebar"] * {
    color: #475569 !important;
}
[data-testid="stSidebar"] .sidebar-brand {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #0b1120 !important;
    letter-spacing: 0.04em;
}
[data-testid="stSidebar"] .sidebar-sub {
    font-size: 0.73rem;
    color: #64748b !important;
    margin-top: 2px;
}
[data-testid="stSidebar"] hr {
    border-color: #e2e8f0 !important;
}

/* ── page titles ── */
.page-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #0b1120;
    line-height: 1.15;
}
.page-subtitle {
    font-size: 1rem;
    color: #64748b;
    margin-top: 0.3rem;
    margin-bottom: 0.5rem;
    font-weight: 400;
}

/* ── section headers ── */
.sec-head {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #94a3b8;
    margin-top: 2rem;
    margin-bottom: 0.75rem;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 0.4rem;
}

/* ── metric cards ── */
.kpi-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    text-align: center;
}
.kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    color: #0b1120;
    line-height: 1;
}
.kpi-label {
    font-size: 0.73rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #94a3b8;
    margin-top: 0.4rem;
    font-weight: 500;
}
.kpi-sub {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.25rem;
}

/* ── callout box ── */
.callout {
    background: #0b1120;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    color: #f1f5f9;
}
.callout .big-stat {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.6rem;
    font-weight: 600;
    color: #f97316;
    line-height: 1;
}
.callout p {
    color: #94a3b8;
    font-size: 0.9rem;
    line-height: 1.7;
    margin: 0.6rem 0 0 0;
}

/* ── label chips ── */
.chip {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 3px 9px;
    border-radius: 4px;
    margin: 2px 2px;
}
.chip-stockout  { background:#fee2e2; color:#b91c1c; }
.chip-overstock { background:#fef3c7; color:#b45309; }
.chip-safe      { background:#dcfce7; color:#15803d; }
.chip-neutral   { background:#f1f5f9; color:#475569; }

/* ── risk result card ── */
.risk-result {
    border-radius: 14px;
    padding: 2.5rem 2rem;
    text-align: center;
}
.risk-stockout  { background:#fff1f2; border:2px solid #fca5a5; }
.risk-overstock { background:#fffbeb; border:2px solid #fcd34d; }
.risk-safe      { background:#f0fdf4; border:2px solid #86efac; }
.risk-emoji   { font-size: 3.5rem; line-height:1; }
.risk-label   { font-family:'IBM Plex Mono',monospace; font-size:1.6rem; font-weight:600; margin:0.6rem 0 0.3rem; }
.risk-conf    { font-size:0.9rem; color:#64748b; }
.risk-action  { font-size:0.88rem; margin-top:0.75rem; font-style:italic; color:#475569; }

/* ── flow step ── */
.flow-step {
    background:#fff;
    border:1.5px solid #e2e8f0;
    border-radius:10px;
    padding:1rem 0.75rem;
    text-align:center;
    height:140px;
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
}
.flow-icon  { font-size:1.6rem; margin-bottom:0.3rem; }
.flow-title { font-family:'IBM Plex Mono',monospace; font-size:0.75rem; font-weight:600; color:#0b1120; }
.flow-desc  { font-size:0.68rem; color:#94a3b8; margin-top:0.2rem; line-height:1.4; }

/* ── label formula card ── */
.formula-card {
    border-radius:10px;
    padding:1.25rem 1.5rem;
}
.formula-stockout  { background:#fff1f2; border:1.5px solid #fca5a5; }
.formula-overstock { background:#fffbeb; border:1.5px solid #fcd34d; }
.formula-safe      { background:#f0fdf4; border:1.5px solid #86efac; }
.formula-title {
    font-family:'IBM Plex Mono',monospace;
    font-size:0.9rem;
    font-weight:600;
    margin-bottom:0.5rem;
}
.formula-rule {
    font-family:'IBM Plex Mono',monospace;
    font-size:0.78rem;
    background:rgba(0,0,0,0.06);
    padding:5px 10px;
    border-radius:5px;
    display:inline-block;
    margin-bottom:0.5rem;
}
.formula-desc { font-size:0.83rem; color:#475569; line-height:1.5; }

/* ── team card ── */
.team-card {
    background:#fff;
    border:1px solid #e2e8f0;
    border-radius:12px;
    padding:1.5rem 1.25rem;
    text-align:center;
    height:100%;
}
.team-name { font-family:'IBM Plex Mono',monospace; font-size:0.85rem; font-weight:600; color:#0b1120; margin-bottom:0.2rem; }
.team-role { font-size:0.75rem; color:#64748b; margin-bottom:0.75rem; }

/* hide default streamlit decoration */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

def sidebar():
    with st.sidebar:
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
        if os.path.exists(logo_path):
            st.markdown('<div style="text-align: center; margin-bottom: 1rem;">', unsafe_allow_html=True)
            st.image(logo_path, width=200)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sidebar-brand">📦 IRP</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-sub">Inventory Risk Predictor</div>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.73rem;color:#475569!important">IE University · MBDS 2026<br>Section 1 · Group 5</div>', unsafe_allow_html=True)