"""
Home.py  ·  Page 1 — Landing page
Run:  streamlit run streamlit_app/Home.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import style

st.set_page_config(
    page_title="IRP · Inventory Risk Predictor",
    page_icon="streamlit_app/assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
style.inject()
style.sidebar()

# ── Hero ──────────────────────────────────────────────────────────────────────
col_logo, col_title, col_link = st.columns([1.0, 2.5, 0.8])
with col_logo:
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    if os.path.exists(logo_path):
        st.markdown("<br>", unsafe_allow_html=True)
        st.image(logo_path, width=270)
with col_title:
    st.markdown("""
    <div style="display: flex; flex-direction: column; justify-content: center; height: 300px;">
        <div class="page-title">Inventory Risk<br>Predictor</div>
        <div class="page-subtitle">ML-powered early warning for retail inventory risk — one day ahead.</div>
    </div>
    """, unsafe_allow_html=True)
with col_link:
    st.markdown("<br>", unsafe_allow_html=True)
    st.link_button("GitHub →", "https://github.com/christianthomas25/inventory-risk-prediction-mlops",
                   use_container_width=True)

st.markdown("---")

# ── What IRP does ─────────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">What it does</div>', unsafe_allow_html=True)
st.markdown("""
Retail inventory management is a daily battle against two opposite failures: **stockouts** that kill
sales and damage customer trust, and **overstock** that ties up capital and forces costly markdowns.
The Inventory Risk Predictor (IRP) monitors every store-product combination and — using today's
inventory levels, sales velocity, demand forecasts, and engineered lag features — predicts **tomorrow's
risk status** as one of three classes: Stockout Risk, Overstock Risk, or Safe Zone. Store managers
receive actionable alerts every morning, enabling proactive reorders, stock transfers, and markdown
decisions before the problem occurs.
""")

st.markdown("---")

# ── KPI cards ─────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">Headline results</div>', unsafe_allow_html=True)

k1, k2, k3 = st.columns(3)
kpis = [
    ("83%",  "Stockout Recall",       "XGBoost · test set", "#fee2e2", "#b91c1c"),
    ("79.2%","F1 Score",              "XGBoost weighted avg", "#dbeafe", "#1d4ed8"),
    ("100",  "Products Monitored",    "5 stores × 20 products", "#dcfce7", "#15803d"),
]
for col, (val, lbl, sub, bg, fg) in zip([k1, k2, k3], kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-card" style="border-top:3px solid {fg}">
            <div class="kpi-value" style="color:{fg}">{val}</div>
            <div class="kpi-label">{lbl}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.caption("See **Model Performance** for the full comparison table and per-class breakdowns.")

st.markdown("---")

# ── Business case callout ─────────────────────────────────────────────────────
st.markdown('<div class="sec-head">Why this matters</div>', unsafe_allow_html=True)

left, right = st.columns(2)

with left:
    st.markdown("""
    <div class="callout" style="height:100%;min-height:280px">
        <div class="big-stat">$740B</div>
        <p>
        That is the estimated value of inventory held in surplus by US retailers alone — an
        inventory <em>glut</em> that represents dead capital, markdown risk, and wasted logistics
        capacity. On the other side, stockouts cost the global retail industry an estimated
        <strong style="color:#f1f5f9">$1.75 trillion</strong> annually in lost sales.
        Early-warning systems like IRP directly attack both numbers by shifting inventory decisions
        from reactive to predictive.
        </p>
        <p style="margin-top:0.75rem">
        Missing a real stockout is <strong style="color:#f97316">5–10× more costly</strong> than
        firing an extra alert. IRP is tuned accordingly: maximise recall on Stockout Risk first,
        precision second.
        </p>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("""
    <div style="display:flex;flex-direction:column;gap:0.6rem;min-height:280px;justify-content:center">
        <div style="font-weight:600;font-size:0.95rem;margin-bottom:0.25rem">The three risk classes</div>
        <div style="background:#fff1f2;border:1.5px solid #fca5a5;border-radius:8px;padding:0.75rem 1rem">
            <strong>🔴 Stockout Risk</strong><br>
            <span style="font-size:0.82rem;color:#64748b">Stock insufficient to cover tomorrow's forecast demand. Trigger reorder or transfer immediately.</span>
        </div>
        <div style="background:#fffbeb;border:1.5px solid #fcd34d;border-radius:8px;padding:0.75rem 1rem">
            <strong>🟠 Overstock Risk</strong><br>
            <span style="font-size:0.82rem;color:#64748b">Excess stock + low sales velocity. Review purchasing plan, consider markdowns.</span>
        </div>
        <div style="background:#f0fdf4;border:1.5px solid #86efac;border-radius:8px;padding:0.75rem 1rem">
            <strong>🟢 Safe Zone</strong><br>
            <span style="font-size:0.82rem;color:#64748b">Healthy inventory balance. No action required.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Flow diagram ──────────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">How it works</div>', unsafe_allow_html=True)

steps = [
    ("📥", "Raw Data",    "Daily CSV\nstore × product"),
    ("⚙️", "Features",   "18 engineered\nlag · rolling · velocity"),
    ("🤖", "Model",      "XGBoost + SMOTE\nt+1 prediction"),
    ("🚨", "Risk Alert", "Stockout / Overstock\n/ Safe Zone"),
    ("✅", "Action",     "Reorder · Transfer\n· Markdown"),
]

s1,a1,s2,a2,s3,a3,s4,a4,s5 = st.columns([2,.35,2,.35,2,.35,2,.35,2])
step_cols  = [s1, s2, s3, s4, s5]
arrow_cols = [a1, a2, a3, a4]

for col, (ico, title, desc) in zip(step_cols, steps):
    with col:
        st.markdown(f"""
        <div class="flow-step">
            <div class="flow-icon">{ico}</div>
            <div class="flow-title">{title}</div>
            <div class="flow-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

for col in arrow_cols:
    with col:
        st.markdown(
            "<div style='text-align:center;font-size:1.3rem;color:#cbd5e1;padding-top:1.1rem'>→</div>",
            unsafe_allow_html=True)
