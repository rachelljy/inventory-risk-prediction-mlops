"""
3_Risk_Predictor.py  ·  Page 4 — Live demo
Tries the FastAPI endpoint first; falls back to a local joblib model if the API is unreachable.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import style

st.set_page_config(page_title="Risk Predictor · IRP", page_icon="🎯", layout="wide")
style.inject()

# ── try to import local model (fallback) ──────────────────────────────────────
try:
    import joblib
    _MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "model.joblib")    
    _LOCAL_ARTIFACTS = joblib.load(_MODEL_PATH) if os.path.exists(_MODEL_PATH) else None
except Exception:
    _LOCAL_ARTIFACTS = None

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">📦 IRP</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Inventory Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<span style="font-size:0.73rem;font-weight:600;color:#94a3b8!important;text-transform:uppercase;letter-spacing:0.08em">API Settings</span>',
                unsafe_allow_html=True)
    api_url = st.text_input("FastAPI endpoint", value="http://localhost:8000")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.73rem;color:#475569!important">IE University · MBDS 2026<br>Section 1 · Group 5</div>',
                unsafe_allow_html=True)

# ── page header ───────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">🎯 Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Enter today\'s inventory data — get tomorrow\'s risk prediction.</div>',
            unsafe_allow_html=True)

# ── API health indicator ───────────────────────────────────────────────────────
api_ok = False
try:
    h = requests.get(f"{api_url}/health", timeout=2).json()
    api_ok = h.get("model_loaded", False)
    model_name = h.get("model_name", "XGBoost")
except Exception:
    pass

if api_ok:
    st.success(f"✅ API connected — model: **{model_name}**  |  predictions served from FastAPI")
elif _LOCAL_ARTIFACTS:
    st.warning("⚠️ API not reachable — running predictions locally from `models/model.joblib`")
else:
    st.error(
        "🔴 No prediction backend available.  \n"
        "Either start the API (`uvicorn app:app --reload`) "
        "or ensure `models/model.joblib` exists (run `python train.py`)."
    )

st.markdown("---")

# ── preset scenarios ──────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">Quick presets</div>', unsafe_allow_html=True)

presets = {
    "🔴 Stockout": dict(
        store="Store_1", product="P_001", category="Groceries", region="North",
        inventory=18, units_sold=127, demand_forecast=135.47,
        price=33.50, competitor_pricing=29.69, discount=20,
        holiday=False, weather="Rainy", seasonality="Autumn",
    ),
    "🟢 Safe Zone": dict(
        store="Store_2", product="P_007", category="Electronics", region="South",
        inventory=250, units_sold=45, demand_forecast=50.0,
        price=120.0, competitor_pricing=118.0, discount=5,
        holiday=False, weather="Sunny", seasonality="Spring",
    ),
    "🟠 Overstock": dict(
        store="Store_3", product="P_015", category="Furniture", region="West",
        inventory=480, units_sold=8, demand_forecast=40.0,
        price=450.0, competitor_pricing=420.0, discount=0,
        holiday=False, weather="Cloudy", seasonality="Winter",
    ),
}

p1, p2, p3 = st.columns(3)
for col, (label, vals) in zip([p1, p2, p3], presets.items()):
    if col.button(label, use_container_width=True):
        st.session_state["preset"] = vals

P = st.session_state.get("preset", presets["🔴 Stockout"])

st.markdown("---")

# ── input form ────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">Input variables</div>', unsafe_allow_html=True)

stores   = [f"Store_{i}" for i in range(1, 6)]
products = [f"P_{i:003d}" for i in range(1, 21)]
cats     = ["Groceries", "Electronics", "Clothing", "Furniture"]
regions  = ["North", "South", "East", "West"]
weathers = ["Sunny", "Rainy", "Snowy", "Cloudy"]
seasons  = ["Spring", "Summer", "Autumn", "Winter"]

col_ctx, col_inv, col_eng = st.columns([1.2, 1.4, 1.4])

with col_ctx:
    st.markdown("**🏷️ Context**")
    store    = st.selectbox("Store ID",    stores,   index=stores.index(P["store"])   if P["store"]    in stores   else 0)
    product  = st.selectbox("Product ID",  products, index=products.index(P["product"]) if P["product"] in products else 0)
    category = st.selectbox("Category",    cats,     index=cats.index(P["category"])   if P["category"] in cats     else 0)
    region   = st.selectbox("Region",      regions,  index=regions.index(P["region"])   if P["region"]   in regions  else 0)
    weather  = st.selectbox("Weather",     weathers, index=weathers.index(P["weather"]) if P["weather"]  in weathers else 0)
    season   = st.selectbox("Seasonality", seasons,  index=seasons.index(P["seasonality"]) if P["seasonality"] in seasons else 0)
    holiday  = st.toggle("Holiday / Promotion active", value=P["holiday"])

with col_inv:
    st.markdown("**📦 Inventory & Sales**")
    inventory        = st.slider("Inventory Level",      0,   500, int(P["inventory"]),       step=1)
    units_sold       = st.slider("Units Sold today",     0,   300, int(P["units_sold"]),      step=1)
    demand_forecast  = st.number_input("Demand Forecast",0.0, 500.0, float(P["demand_forecast"]), step=0.5)
    price            = st.number_input("Price (€)",      0.0,2000.0, float(P["price"]),           step=0.5)
    competitor_price = st.number_input("Competitor Price (€)", 0.0, 2000.0, float(P["competitor_pricing"]), step=0.5)
    discount         = st.slider("Discount (%)",         0,   20,  int(P["discount"]),        step=1)

with col_eng:
    st.markdown("**📈 Derived inputs** *(estimated from above)*")
    st.caption("These are pre-filled with sensible estimates. Adjust if you have more precise values.")

    # Sensible auto-estimates
    inv_lag1          = st.number_input("Inventory Lag1 (yesterday)", 0.0, 1000.0,
                                         float(inventory * 1.4), step=1.0)
    units_sold_lag1   = st.number_input("Units Sold Lag1 (yesterday)", 0.0, 500.0,
                                         float(units_sold * 0.9), step=1.0)
    rolling7_inv      = st.number_input("7-day rolling avg inventory", 0.0, 1000.0,
                                         float(inventory * 1.2), step=1.0)
    inv_change        = st.number_input("Inventory Change (day-over-day)", -500.0, 500.0,
                                         float(inventory - inv_lag1), step=1.0)
    inv_change_pct    = round(inv_change / inv_lag1, 4) if inv_lag1 else 0.0
    st.markdown(f"**Inventory Change %:** `{inv_change_pct:.3f}` *(auto-computed)*")

    days_of_stock     = st.number_input("Days of Stock",     0.0, 200.0,
                                         round(inventory / units_sold, 2) if units_sold else 0.0, step=0.1)
    inv_vs_rolling7   = round(float(inventory) - float(rolling7_inv), 2)
    st.markdown(f"**Inventory vs Rolling7:** `{inv_vs_rolling7:.2f}` *(auto-computed)*")
    sales_velocity    = st.number_input("Sales Velocity",    0.0,   5.0,
                                         round(units_sold / max(demand_forecast, 1), 3), step=0.01, format="%.3f")
    coverage_ratio    = st.number_input("Coverage Ratio",    0.0, 50.0,
                                         round(inventory / max(demand_forecast, 1), 3), step=0.01, format="%.3f")
    forecast_error    = st.number_input("Forecast Error",  -200.0, 200.0,
                                         round(units_sold - demand_forecast, 2), step=0.1)
    order_to_inv      = st.number_input("Order-to-Inventory Ratio", 0.0, 20.0,
                                         round(demand_forecast / max(inventory, 1), 3), step=0.01, format="%.3f")

st.markdown("")
predict_btn = st.button("🚀  Predict Risk", type="primary", use_container_width=True)

# ── prediction logic ──────────────────────────────────────────────────────────
def call_api(payload: dict, base_url: str):
    r = requests.post(f"{base_url}/predict", json=payload, timeout=8)
    r.raise_for_status()
    return r.json()

def call_local(payload: dict, arts: dict):
    import pandas as pd
    FIELD_TO_FEATURE = {
        "inventory_reconstructed": "Inventory_Reconstructed",
        "units_sold":              "Units Sold",
        "demand_forecast":         "Demand Forecast",
        "price":                   "Price",
        "discount":                "Discount",
        "competitor_pricing":      "Competitor Pricing",
        "holiday_promotion":       "Holiday/Promotion",
        "inventory_change":        "Inventory_Change",
        "inventory_change_pct":    "Inventory_Change_Pct",
        "days_of_stock":           "Days_of_Stock",
        "inventory_vs_rolling7":   "Inventory_vs_Rolling7",
        "sales_velocity":          "Sales_Velocity",
        "inventory_lag1":          "Inventory_Lag1",
        "units_sold_lag1":         "Units_Sold_Lag1",
        "rolling7_inventory":      "Rolling7_Inventory",
        "coverage_ratio":          "Coverage_Ratio",
        "forecast_error":          "Forecast_Error",
        "order_to_inventory":      "Order_to_Inventory",
    }
    CAT_FIELD_TO_COL = {
        "category":         "Category",
        "region":           "Region",
        "weather_condition":"Weather Condition",
        "seasonality":      "Seasonality",
    }
    model   = arts["model"]
    le      = arts["label_encoder"]
    feats   = arts["features"]
    cat_enc = arts["categorical_encoders"]

    row = {}
    for api_f, col in FIELD_TO_FEATURE.items():
        row[col] = payload.get(api_f, 0)
    for api_f, col in CAT_FIELD_TO_COL.items():
        enc_col = col + "_enc"
        val = payload.get(api_f, "")
        if col in cat_enc:
            enc = cat_enc[col]
            row[enc_col] = int(enc.transform([val])[0]) if val in enc.classes_ else 0
        else:
            row[enc_col] = 0

    X = pd.DataFrame([row])[feats]
    pred_enc = model.predict(X)[0]
    label    = le.inverse_transform([pred_enc])[0]
    proba    = model.predict_proba(X)[0]
    classes  = le.inverse_transform(range(len(proba)))
    prob_dict = {n: round(float(p), 4) for n, p in zip(classes, proba)}
    return {"risk_label": label, "confidence": round(float(proba.max()),4), "probabilities": prob_dict}


if predict_btn:
    payload = dict(
        inventory_reconstructed = float(inventory),
        units_sold              = int(units_sold),
        demand_forecast         = float(demand_forecast),
        price                   = float(price),
        discount                = int(discount),
        competitor_pricing      = float(competitor_price),
        holiday_promotion       = int(holiday),
        inventory_change        = float(inv_change),
        inventory_change_pct    = float(inv_change_pct),
        days_of_stock           = float(days_of_stock),
        inventory_vs_rolling7   = float(inv_vs_rolling7),
        sales_velocity          = float(sales_velocity),
        inventory_lag1          = float(inv_lag1),
        units_sold_lag1         = float(units_sold_lag1),
        rolling7_inventory      = float(rolling7_inv),
        coverage_ratio          = float(coverage_ratio),
        forecast_error          = float(forecast_error),
        order_to_inventory      = float(order_to_inv),
        category                = category,
        region                  = region,
        weather_condition       = weather,
        seasonality             = season,
    )

    result = None
    source = ""
    with st.spinner("Running prediction…"):
        if api_ok:
            try:
                result = call_api(payload, api_url)
                source = "FastAPI"
            except Exception as e:
                st.warning(f"API call failed ({e}). Falling back to local model.")
        if result is None and _LOCAL_ARTIFACTS:
            try:
                result = call_local(payload, _LOCAL_ARTIFACTS)
                source = "local model"
            except Exception as e:
                st.error(f"Local prediction failed: {e}")

    if result:
        st.markdown("---")
        label      = result["risk_label"]
        confidence = result["confidence"]
        probs      = result["probabilities"]

        RISK_CONFIG = {
            "Stockout Risk":  ("risk-stockout",  "🔴", "#b91c1c", "Place an emergency reorder or initiate a stock transfer immediately."),
            "Overstock Risk": ("risk-overstock",  "🟠", "#b45309", "Review the purchasing plan. Consider markdowns or redistribution to other stores."),
            "Safe Zone":      ("risk-safe",       "🟢", "#15803d", "No action required — inventory is in a healthy state."),
        }
        css_cls, emoji, fg, action = RISK_CONFIG.get(label, ("risk-safe","🟢","#15803d","No action needed."))

        res_col, prob_col = st.columns([1.1, 1.9])

        with res_col:
            st.markdown(f"""
            <div class="risk-result {css_cls}">
                <div class="risk-emoji">{emoji}</div>
                <div class="risk-label" style="color:{fg}">{label}</div>
                <div class="risk-conf">Confidence: <strong>{confidence:.1%}</strong></div>
                <hr style="border-color:rgba(0,0,0,0.08);margin:0.75rem 0">
                <div class="risk-action">{action}</div>
                <div style="font-size:0.72rem;color:#94a3b8;margin-top:0.75rem">
                    Source: {source} · {store} · {product}
                </div>
            </div>""", unsafe_allow_html=True)

        with prob_col:
            st.markdown("**Prediction probabilities**")
            sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
            class_names  = [c for c, _ in sorted_probs]
            values       = [v for _, v in sorted_probs]

            COLOR_MAP = {"Stockout Risk":"#ef4444","Overstock Risk":"#f59e0b","Safe Zone":"#22c55e"}
            bar_colors = [COLOR_MAP.get(c,"#94a3b8") for c in class_names]

            fig_p, ax_p = plt.subplots(figsize=(7, 2.8))
            y = np.arange(len(class_names))
            bars_p = ax_p.barh(y, values, color=bar_colors, edgecolor="white", height=0.5)
            ax_p.set_yticks(y)
            ax_p.set_yticklabels(class_names, fontsize=10)
            ax_p.set_xlim(0, 1.05)
            ax_p.set_xlabel("Probability", fontsize=9)
            ax_p.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
            ax_p.spines[["top","right","left"]].set_visible(False)
            ax_p.tick_params(left=False, labelsize=9)
            for bar, val in zip(bars_p, values):
                ax_p.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                          f"{val:.1%}", va="center", fontsize=9, color="#475569")
            plt.tight_layout()
            st.pyplot(fig_p, use_container_width=True)
            plt.close()

            st.markdown(f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
                        padding:0.75rem 1rem;font-size:0.83rem;color:#475569;margin-top:0.5rem">
                ℹ️ Probabilities show the model's confidence across all three classes.
                The predicted class is the one with the highest probability.
            </div>""", unsafe_allow_html=True)

        with st.expander("🔍 Raw API payload"):
            st.json(payload)
