"""
4_About.py  ·  Page 5 — Team
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import style

st.set_page_config(page_title="About · IRP", page_icon="👥", layout="wide")
style.inject()
style.sidebar()

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")

def show_photo(filename):
    from PIL import Image
    path = os.path.join(ASSETS, filename)
    if os.path.exists(path):
        img = Image.open(path)
        # crop to square from centre
        w, h = img.size
        size = min(w, h)
        left = (w - size) // 2
        top  = (h - size) // 2
        img  = img.crop((left, top, left + size, top + size))
        img  = img.resize((400, 400), Image.LANCZOS)
        st.image(img, use_container_width=True)
    else:
        st.markdown("""
        <div style="width:100%;aspect-ratio:1/1;background:#f1f5f9;
                    border:1.5px dashed #cbd5e1;border-radius:10px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:2rem;color:#cbd5e1;margin-bottom:0.5rem">
            👤
        </div>""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">👥 The Team</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">IE University · MBDS 2026 · Section 1, Group 5</div>',
            unsafe_allow_html=True)
st.markdown("---")

# ── team cards ────────────────────────────────────────────────────────────────
team = [
    ("maria.jpeg",   "Maria-Irina Popa",         "ML Pipeline & Modeling"),
    ("enzo.jpeg",    "Enzo Jerez",                "MLOps & CI/CD"),
    ("roberto.jpeg", "Roberto Cummings",          "API Development"),
    ("rachel.jpeg",  "Jia Yi Rachel Lee",         "Data & EDA"),
    ("thomas.jpeg",  "Thomas Christian Matenco",  "Architecture & Testing"),
]

cols = st.columns(5)
for col, (photo, name, role) in zip(cols, team):
    with col:
        show_photo(photo)
        st.markdown(f"""
        <div style="text-align:center;padding:0.25rem 0">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;
                        font-weight:600;color:#0b1120;margin-bottom:0.15rem">{name}</div>
            <div style="font-size:0.73rem;color:#94a3b8">{role}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── project info ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**About the project**")
    st.markdown("""
    The Inventory Risk Predictor (IRP) is an end-to-end MLOps system that predicts 
    retail inventory risk one day in advance — Stockout, Overstock, or Safe Zone.  
    Built with XGBoost, FastAPI, MLflow, Docker, and GitHub Actions CI/CD.
    """)
    st.link_button("📁 GitHub Repository",
                   "https://github.com/christianthomas25/inventory-risk-prediction-mlops",
                   use_container_width=True)

with col_right:
    st.markdown("**Pipeline stages**")
    for i, stage in enumerate([
        "Business Understanding & Data",
        "Feature Engineering & Modelling",
        "Experiment Tracking & Model Selection",
        "API Serving & Containerisation",
        "CI/CD & Deployment",
    ], 1):
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.45rem">
            <div style="background:#0b1120;color:#f1f5f9;font-family:'IBM Plex Mono',monospace;
                        font-size:0.68rem;padding:2px 7px;border-radius:4px;flex-shrink:0">{i}</div>
            <div style="font-size:0.83rem;color:#475569">{stage}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")
st.caption("⚠️ Synthetic dataset · Academic project · Not for commercial use")
