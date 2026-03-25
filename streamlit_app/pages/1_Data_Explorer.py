"""
1_Data_Explorer.py  ·  Page 2 — Dataset overview + interactive filter + label definitions
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import style

st.set_page_config(page_title="Data Explorer · IRP", page_icon="🗂️", layout="wide")
style.inject()
style.sidebar()

# ── helpers ───────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "retail_store_inventory.csv")

@st.cache_data(show_spinner="Loading dataset…")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])

    # Reconstruct inventory (mirrors train.py logic)
    df = df.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)

    # Apply risk labels (same thresholds as config.yaml)
    theta_low  = 1.2
    theta_high = 4.5
    low_sales  = 0.8

    df["_stock_risk"]    = df["Inventory Level"] < df["Demand Forecast"] * theta_low
    df["_over_risk"]     = (df["Inventory Level"] > df["Demand Forecast"] * theta_high) & \
                           (df["Units Sold"] < df["Demand Forecast"] * low_sales)
    df["Risk_Label"] = np.where(
        df["_stock_risk"], "Stockout Risk",
        np.where(df["_over_risk"], "Overstock Risk", "Safe Zone")
    )
    df.drop(columns=["_stock_risk","_over_risk"], inplace=True)
    return df


# ── load ──────────────────────────────────────────────────────────────────────
try:
    df = load_data(DATA_PATH)
    data_loaded = True
except FileNotFoundError:
    data_loaded = False

# ── page header ───────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">🗂️ Data Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Dataset overview, interactive filters, class distributions, and label definitions.</div>',
            unsafe_allow_html=True)
st.markdown("---")

if not data_loaded:
    st.warning("""
    **Dataset not found.**  
    Place `retail_store_inventory.csv` in the `data/` folder at the repo root, then reload this page.  
    Expected path: `data/retail_store_inventory.csv`
    """)
    st.stop()

# ── dataset summary stats ─────────────────────────────────────────────────────
st.markdown('<div class="sec-head">Dataset summary</div>', unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6 = st.columns(6)
summary = [
    (f"{len(df):,}",                         "Total rows"),
    (str(df["Date"].min().date()),           "Start date"),
    (str(df["Date"].max().date()),           "End date"),
    (str(df["Store ID"].nunique()),          "Stores"),
    (str(df["Product ID"].nunique()),        "Products per store"),
    (str(df.select_dtypes("number").shape[1]), "Numeric cols"),
]
for col,(val,lbl) in zip([c1,c2,c3,c4,c5,c6], summary):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.2rem;font-weight:600;color:#0b1120">{val}</div>
            <div class="kpi-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("")

# ── interactive filter ────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">Interactive filter</div>', unsafe_allow_html=True)

categories  = sorted(df["Category"].dropna().unique().tolist())  if "Category"   in df.columns else []
regions     = sorted(df["Region"].dropna().unique().tolist())    if "Region"     in df.columns else []
stores      = sorted(df["Store ID"].dropna().unique().tolist())  if "Store ID"   in df.columns else []

fc, fr, fs = st.columns(3)
with fc:
    sel_cat = st.multiselect("Category", categories,  default=categories,  key="cat")
with fr:
    sel_reg = st.multiselect("Region",   regions,     default=regions,     key="reg")
with fs:
    sel_sto = st.multiselect("Store ID", stores,      default=stores,      key="sto")

mask = (
    df["Category"].isin(sel_cat) &
    df["Region"].isin(sel_reg) &
    df["Store ID"].isin(sel_sto)
) if (sel_cat and sel_reg and sel_sto) else pd.Series(False, index=df.index)

filtered = df[mask].copy() if mask.any() else pd.DataFrame()

if filtered.empty:
    st.info("No rows match the current filters.")
else:
    t1 = st.tabs(["📋 Sample rows"]),


    disp_cols = ["Date","Store ID","Product ID","Category","Region",
                 "Inventory Level","Units Sold","Demand Forecast","Risk_Label"]
    disp_cols = [c for c in disp_cols if c in filtered.columns]
    st.dataframe(
        filtered[disp_cols].sort_values("Date", ascending=False).head(200),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Showing up to 200 rows of {len(filtered):,} filtered records.")

st.markdown("---")

# ── global class distribution ─────────────────────────────────────────────────
st.markdown('<div class="sec-head">Overall class distribution</div>', unsafe_allow_html=True)

label_counts  = df["Risk_Label"].value_counts()
label_total   = label_counts.sum()
color_map     = {"Stockout Risk":"#ef4444","Overstock Risk":"#f59e0b","Safe Zone":"#22c55e"}

g1, g2 = st.columns([2, 3])

with g1:
    for lbl, cnt in label_counts.items():
        pct = cnt / label_total * 100
        chip_cls = {"Stockout Risk":"chip-stockout","Overstock Risk":"chip-overstock","Safe Zone":"chip-safe"}.get(lbl,"chip-neutral")
        st.markdown(f'<span class="chip {chip_cls}">{lbl}</span>', unsafe_allow_html=True)
        st.progress(pct / 100, text=f"{cnt:,} rows · {pct:.1f}%")
        st.markdown("")

with g2:
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    bars2 = ax2.bar(
        label_counts.index, label_counts.values,
        color=[color_map.get(l,"#94a3b8") for l in label_counts.index],
        edgecolor="white", linewidth=1.5, width=0.5
    )
    for bar, val in zip(bars2, label_counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + label_total*0.005,
                 f"{val/label_total*100:.1f}%",
                 ha="center", va="bottom", fontsize=9, color="#475569")
    ax2.set_ylabel("Records", fontsize=9)
    ax2.set_title("Full dataset — all records", fontsize=10, fontweight="bold", pad=8)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

st.markdown("---")

# ── subgroup breakdown ────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">Subgroup breakdown</div>', unsafe_allow_html=True)

sub_by = st.radio("Break down by:", ["Category", "Region"], horizontal=True, key="subgroup")

if sub_by in df.columns:
    groups   = sorted(df[sub_by].dropna().unique())
    labels3  = ["Stockout Risk","Overstock Risk","Safe Zone"]
    x        = np.arange(len(groups))
    width    = 0.25
    colors3  = ["#ef4444","#f59e0b","#22c55e"]

    # ── absolute counts ──
    fig3, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    for ax, normalise, title_sfx in zip(axes, [False, True], ["Counts", "% within group"]):
        data_matrix = []
        for lbl in labels3:
            vals = []
            for grp in groups:
                sub = df[df[sub_by] == grp]
                cnt = (sub["Risk_Label"] == lbl).sum()
                if normalise:
                    vals.append(cnt / len(sub) * 100 if len(sub) else 0)
                else:
                    vals.append(cnt)
            data_matrix.append(vals)

        for i, (lbl, vals, clr) in enumerate(zip(labels3, data_matrix, colors3)):
            offset = (i - 1) * width
            bars_g = ax.bar(x + offset, vals, width, label=lbl,
                            color=clr, alpha=0.88, edgecolor="white", linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=9)
        ax.set_ylabel("% within group" if normalise else "Records", fontsize=9)
        ax.set_title(f"Risk class by {sub_by} — {title_sfx}", fontsize=10, fontweight="bold", pad=8)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.spines[["top","right"]].set_visible(False)
        if normalise:
            ax.set_ylim(0, 105)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}%"))
        else:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))

    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close()

st.markdown("---")

# ── label definitions ─────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">Risk label definitions & thresholds</div>', unsafe_allow_html=True)
st.markdown(
    '<span style="font-size:0.82rem;color:#64748b">Labels are applied to the <em>next</em> day\'s inventory state '
    '(t+1 shift) — so today\'s features predict <strong>tomorrow\'s</strong> risk.</span>',
    unsafe_allow_html=True)
st.markdown("")

lc1, lc2, lc3 = st.columns(3)

with lc1:
    st.markdown("""
    <div class="formula-card formula-stockout">
        <div class="formula-title">🔴 Stockout Risk</div>
        <div class="formula-rule">Inventory &lt; Demand × 1.0</div>
        <div class="formula-desc">
            Stock is insufficient to cover even one full day of forecast demand.
            <br><br>
            <strong>Business impact:</strong> lost sales, customer dissatisfaction, potential SLA breach.
            <br><br>
            <strong>Priority:</strong> Highest — recall maximised here because a missed stockout costs
            5–10× more than a false alert.
        </div>
    </div>""", unsafe_allow_html=True)

with lc2:
    st.markdown("""
    <div class="formula-card formula-overstock">
        <div class="formula-title">🟠 Overstock Risk</div>
        <div class="formula-rule">Inventory &gt; Demand × 1.5<br>AND Sales &lt; Demand × 0.5</div>
        <div class="formula-desc">
            Excess stock combined with very low sales velocity.
            <br><br>
            <strong>Business impact:</strong> capital tied up, markdown risk, warehouse capacity strain.
            <br><br>
            <strong>Note:</strong> Rarest class (~3% of records). Day-to-day persistence ~4%, 
            making it the hardest to predict reliably.
        </div>
    </div>""", unsafe_allow_html=True)

with lc3:
    st.markdown("""
    <div class="formula-card formula-safe">
        <div class="formula-title">🟢 Safe Zone</div>
        <div class="formula-rule">Neither condition above</div>
        <div class="formula-desc">
            Inventory is in a healthy balance — no corrective action needed.
            <br><br>
            <strong>Business impact:</strong> None. Baseline healthy state.
            <br><br>
            <strong>Note:</strong> Majority class. Well-predicted across all models. 
            The goal is to correctly detect the risky departures from this baseline.
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("")
st.warning("""
⚠️ **Proxy labels** — risk labels are derived from business-rule thresholds applied to reconstructed 
inventory levels, not from observed real-world stockout or overstock events. Findings should be 
interpreted with this caveat in mind.
""")
