"""
3_Risk_Predictor.py  ·  Standalone version — no API, no joblib required.
Trains XGBoost directly from retail_store_inventory.csv on first load,
then caches the model for the session.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import style

st.set_page_config(page_title="Risk Predictor · IRP", page_icon="🎯", layout="wide")
style.inject()

# ── locate the CSV (repo root, one or two levels up) ─────────────────────────
def _find_csv() -> str | None:
    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, "..", "..", "retail_store_inventory.csv"),
        os.path.join(base, "..", "retail_store_inventory.csv"),
        os.path.join(base, "retail_store_inventory.csv"),
        "retail_store_inventory.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


# ── train & cache model ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    csv_path = _find_csv()
    if csv_path is None:
        return None, "retail_store_inventory.csv not found. Place it at the repo root."

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # categorical encoding
    cat_cols = ["Category", "Region", "Weather Condition", "Seasonality"]
    cat_encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le_c = LabelEncoder()
            df[col + "_enc"] = le_c.fit_transform(df[col].astype(str))
            cat_encoders[col] = le_c

    # detect column names flexibly
    inv_col  = next((c for c in df.columns if "inventory" in c.lower()
                     and not any(x in c.lower() for x in ["lag","rolling","change","vs","level"])), "Inventory Level")
    if inv_col not in df.columns:
        inv_col = next((c for c in df.columns if "inventory" in c.lower()), "Inventory Level")
    sold_col = next((c for c in df.columns if "units sold" in c.lower() and "lag" not in c.lower()), "Units Sold")
    fore_col = next((c for c in df.columns if "demand forecast" in c.lower()), "Demand Forecast")

    # feature engineering
    df["Inventory_Lag1"]        = df[inv_col].shift(1).fillna(df[inv_col])
    df["Units_Sold_Lag1"]       = df[sold_col].shift(1).fillna(df[sold_col])
    df["Rolling7_Inventory"]    = df[inv_col].rolling(7, min_periods=1).mean()
    df["Inventory_Change"]      = df[inv_col] - df["Inventory_Lag1"]
    df["Inventory_Change_Pct"]  = (df["Inventory_Change"] / df["Inventory_Lag1"].replace(0, np.nan)).fillna(0)
    df["Inventory_vs_Rolling7"] = df[inv_col] - df["Rolling7_Inventory"]
    df["Days_of_Stock"]         = (df[inv_col] / df[sold_col].replace(0, np.nan)).fillna(0)
    df["Sales_Velocity"]        = (df[sold_col] / df[fore_col].replace(0, np.nan)).fillna(0)
    df["Coverage_Ratio"]        = (df[inv_col] / df[fore_col].replace(0, np.nan)).fillna(0)
    df["Forecast_Error"]        = df[sold_col] - df[fore_col]
    df["Order_to_Inventory"]    = (df[fore_col] / df[inv_col].replace(0, np.nan)).fillna(0)

    # label assignment — balanced thresholds
    conditions = [
        (df["Coverage_Ratio"] < 0.2) | (df["Days_of_Stock"] < 3),
        (df["Coverage_Ratio"] > 2.0) & (df["Days_of_Stock"] > 30),
    ]
    choices = ["Stockout Risk", "Overstock Risk"]
    df["Risk_Label"] = np.select(conditions, choices, default="Safe Zone")

    # feature list
    num_features = [
        inv_col, sold_col, fore_col,
        "Price", "Discount", "Competitor Pricing",
        "Inventory_Lag1", "Units_Sold_Lag1", "Rolling7_Inventory",
        "Inventory_Change", "Inventory_Change_Pct", "Inventory_vs_Rolling7",
        "Days_of_Stock", "Sales_Velocity", "Coverage_Ratio",
        "Forecast_Error", "Order_to_Inventory",
    ]
    hol_col = next((c for c in df.columns if "holiday" in c.lower()), None)
    if hol_col:
        df[hol_col] = df[hol_col].astype(int)
        num_features.append(hol_col)

    enc_features = [col + "_enc" for col in cat_cols if col + "_enc" in df.columns]
    all_features = [f for f in num_features if f in df.columns] + enc_features

    df = df.dropna(subset=all_features + ["Risk_Label"])
    X  = df[all_features]
    le = LabelEncoder()
    y  = le.fit_transform(df["Risk_Label"])

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    return {
        "model": model, "label_encoder": le,
        "features": all_features, "cat_encoders": cat_encoders,
        "inv_col": inv_col, "sold_col": sold_col,
        "fore_col": fore_col, "hol_col": hol_col,
    }, None


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">📦 IRP</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Inventory Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.73rem;color:#475569!important">'
        "IE University · MBDS 2026<br>Section 1 · Group 5</div>",
        unsafe_allow_html=True,
    )

# ── page header ───────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">🎯 Risk Predictor</div>', unsafe_allow_html=True)
st.markdown(
    "<div class='page-subtitle'>Enter today's inventory data — get tomorrow's risk prediction.</div>",
    unsafe_allow_html=True,
)

# ── load / train model ────────────────────────────────────────────────────────
with st.spinner("Loading model… (first run trains XGBoost on your data, ~10 s)"):
    artifacts, load_error = load_model()

if load_error:
    st.error(f"🔴 {load_error}")
    st.stop()

st.success("✅ Model ready — XGBoost trained on `retail_store_inventory.csv`")
st.markdown("---")

# ── preset scenarios ──────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">Quick presets</div>', unsafe_allow_html=True)

presets = {
    "🔴 Stockout": dict(inventory=18,  units_sold=127, demand_forecast=135.47,
                        price=33.50,   competitor_pricing=29.69, discount=20,
                        category="Groceries",   seasonality="Autumn"),
    "🟢 Safe Zone": dict(inventory=250, units_sold=45,  demand_forecast=50.0,
                         price=120.0,  competitor_pricing=118.0,  discount=5,
                         category="Electronics", seasonality="Spring"),
    "🟠 Overstock": dict(inventory=480, units_sold=8,   demand_forecast=40.0,
                         price=450.0,  competitor_pricing=420.0,  discount=0,
                         category="Furniture",   seasonality="Winter"),
}

p1, p2, p3 = st.columns(3)
for col, (label, vals) in zip([p1, p2, p3], presets.items()):
    if col.button(label, use_container_width=True):
        st.session_state["preset"] = vals

P = st.session_state.get("preset", presets["🔴 Stockout"])
st.markdown("---")

# ── input form — 8 variables ──────────────────────────────────────────────────
st.markdown('<div class="sec-head">Input variables</div>', unsafe_allow_html=True)

cats    = ["Groceries", "Electronics", "Clothing", "Furniture"]
seasons = ["Spring", "Summer", "Autumn", "Winter"]

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**📦 Inventory & Sales**")
    inventory       = st.slider("Inventory Level",  0, 500, int(P["inventory"]),      step=1)
    units_sold      = st.slider("Units Sold today", 0, 300, int(P["units_sold"]),     step=1)
    demand_forecast = st.number_input("Demand Forecast", 0.0, 500.0, float(P["demand_forecast"]), step=0.5)
    price           = st.number_input("Price (€)",       0.0, 2000.0, float(P["price"]),           step=0.5)

with col_right:
    st.markdown("**🏷️ Market & Context**")
    competitor_price = st.number_input("Competitor Price (€)", 0.0, 2000.0, float(P["competitor_pricing"]), step=0.5)
    discount         = st.slider("Discount (%)", 0, 20, int(P["discount"]), step=1)
    category         = st.selectbox("Category",    cats,    index=cats.index(P["category"])       if P["category"]    in cats    else 0)
    seasonality      = st.selectbox("Seasonality", seasons, index=seasons.index(P["seasonality"]) if P["seasonality"] in seasons else 0)

st.markdown("")
predict_btn = st.button("🚀  Predict Risk", type="primary", use_container_width=True)


# ── build full feature row from 8 inputs ─────────────────────────────────────
def build_row(inventory, units_sold, demand_forecast, price,
              competitor_price, discount, category, seasonality, artifacts):
    inv  = float(inventory)
    sold = float(units_sold)
    fore = float(demand_forecast)

    inv_lag1     = round(inv  * 1.1,  2)
    sold_lag1    = round(sold * 0.95, 2)
    rolling7_inv = round(inv  * 1.05, 2)
    inv_change   = round(inv  - inv_lag1, 2)
    inv_chg_pct  = round(inv_change / inv_lag1, 4) if inv_lag1 else 0.0
    inv_vs_r7    = round(inv  - rolling7_inv, 2)
    days_stock   = round(inv  / sold, 2) if sold > 0 else 0.0
    sales_vel    = round(sold / fore, 3) if fore > 0 else 0.0
    coverage     = round(inv  / fore, 3) if fore > 0 else 0.0
    fore_err     = round(sold - fore, 2)
    ord_to_inv   = round(fore / inv,  3) if inv  > 0 else 0.0

    row = {
        artifacts["inv_col"]:       inv,
        artifacts["sold_col"]:      sold,
        artifacts["fore_col"]:      fore,
        "Price":                    float(price),
        "Discount":                 int(discount),
        "Competitor Pricing":       float(competitor_price),
        "Inventory_Lag1":           inv_lag1,
        "Units_Sold_Lag1":          sold_lag1,
        "Rolling7_Inventory":       rolling7_inv,
        "Inventory_Change":         inv_change,
        "Inventory_Change_Pct":     inv_chg_pct,
        "Inventory_vs_Rolling7":    inv_vs_r7,
        "Days_of_Stock":            days_stock,
        "Sales_Velocity":           sales_vel,
        "Coverage_Ratio":           coverage,
        "Forecast_Error":           fore_err,
        "Order_to_Inventory":       ord_to_inv,
    }
    if artifacts["hol_col"]:
        row[artifacts["hol_col"]] = 0

    cat_map = {"Category": category, "Region": "North",
               "Weather Condition": "Sunny", "Seasonality": seasonality}
    for col, val in cat_map.items():
        enc_col = col + "_enc"
        if col in artifacts["cat_encoders"]:
            enc = artifacts["cat_encoders"][col]
            row[enc_col] = int(enc.transform([val])[0]) if val in enc.classes_ else 0

    feat_order = artifacts["features"]
    return pd.DataFrame([[row.get(f, 0) for f in feat_order]], columns=feat_order)


# ── prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Running prediction…"):
        X_input = build_row(
            inventory, units_sold, demand_forecast,
            price, competitor_price, discount,
            category, seasonality, artifacts,
        )
        model      = artifacts["model"]
        le         = artifacts["label_encoder"]
        pred_enc   = model.predict(X_input)[0]
        label      = le.inverse_transform([pred_enc])[0]
        proba      = model.predict_proba(X_input)[0]
        classes    = le.inverse_transform(range(len(proba)))
        probs      = {n: round(float(p), 4) for n, p in zip(classes, proba)}
        confidence = float(proba.max())

    st.markdown("---")

    RISK_CONFIG = {
        "Stockout Risk":  ("risk-stockout",  "🔴", "#b91c1c",
                           "Place an emergency reorder or initiate a stock transfer immediately."),
        "Overstock Risk": ("risk-overstock",  "🟠", "#b45309",
                           "Review the purchasing plan. Consider markdowns or redistribution to other stores."),
        "Safe Zone":      ("risk-safe",       "🟢", "#15803d",
                           "No action required — inventory is in a healthy state."),
    }
    css_cls, emoji, fg, action = RISK_CONFIG.get(label, ("risk-safe", "🟢", "#15803d", "No action needed."))

    res_col, prob_col = st.columns([1.1, 1.9])

    with res_col:
        st.markdown(
            f"""
            <div class="risk-result {css_cls}">
                <div class="risk-emoji">{emoji}</div>
                <div class="risk-label" style="color:{fg}">{label}</div>
                <div class="risk-conf">Confidence: <strong>{confidence:.1%}</strong></div>
                <hr style="border-color:rgba(0,0,0,0.08);margin:0.75rem 0">
                <div class="risk-action">{action}</div>
                <div style="font-size:0.72rem;color:#94a3b8;margin-top:0.75rem">
                    Source: local XGBoost · {category} · {seasonality}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    with prob_col:
        st.markdown("**Prediction probabilities**")
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        class_names  = [c for c, _ in sorted_probs]
        values       = [v for _, v in sorted_probs]

        COLOR_MAP = {"Stockout Risk": "#ef4444", "Overstock Risk": "#f59e0b", "Safe Zone": "#22c55e"}
        bar_colors = [COLOR_MAP.get(c, "#94a3b8") for c in class_names]

        fig_p, ax_p = plt.subplots(figsize=(7, 2.8))
        y      = np.arange(len(class_names))
        bars_p = ax_p.barh(y, values, color=bar_colors, edgecolor="white", height=0.5)
        ax_p.set_yticks(y)
        ax_p.set_yticklabels(class_names, fontsize=10)
        ax_p.set_xlim(0, 1.05)
        ax_p.set_xlabel("Probability", fontsize=9)
        ax_p.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax_p.spines[["top", "right", "left"]].set_visible(False)
        ax_p.tick_params(left=False, labelsize=9)
        for bar, val in zip(bars_p, values):
            ax_p.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                      f"{val:.1%}", va="center", fontsize=9, color="#475569")
        plt.tight_layout()
        st.pyplot(fig_p, use_container_width=True)
        plt.close()

        st.markdown(
            """
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
                        padding:0.75rem 1rem;font-size:0.83rem;color:#475569;margin-top:0.5rem">
                ℹ️ Probabilities show the model's confidence across all three classes.
                The predicted class is the one with the highest probability.
            </div>""",
            unsafe_allow_html=True,
        )

    with st.expander("🔍 Feature row sent to model"):
        st.dataframe(X_input)
