from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Model Performance · IRP", page_icon="📈", layout="wide")

# ── hard-coded data ───────────────────────────────────────────────────────────

CLASS_LABELS = ["Overstock Risk", "Safe Zone", "Stockout Risk"]

CLASSIFICATION_REPORTS = {
    "Logistic Regression": {
        "Overstock Risk": {"Precision": 0.12, "Recall": 0.52, "F1": 0.19},
        "Safe Zone":      {"Precision": 0.67, "Recall": 0.25, "F1": 0.37},
        "Stockout Risk":  {"Precision": 0.80, "Recall": 0.85, "F1": 0.82},
        "_summary": {"Accuracy": 0.60, "Macro F1": 0.46, "Weighted F1": 0.61},
    },
    "Random Forest": {
        "Overstock Risk": {"Precision": 0.12, "Recall": 0.41, "F1": 0.19},
        "Safe Zone":      {"Precision": 0.65, "Recall": 0.45, "F1": 0.53},
        "Stockout Risk":  {"Precision": 0.86, "Recall": 0.83, "F1": 0.84},
        "_summary": {"Accuracy": 0.66, "Macro F1": 0.52, "Weighted F1": 0.69},
    },
    "XGBoost": {
        "Overstock Risk": {"Precision": 0.13, "Recall": 0.28, "F1": 0.17},
        "Safe Zone":      {"Precision": 0.66, "Recall": 0.61, "F1": 0.64},
        "Stockout Risk":  {"Precision": 0.88, "Recall": 0.81, "F1": 0.84},
        "_summary": {"Accuracy": 0.70, "Macro F1": 0.55, "Weighted F1": 0.72},
    },
}

CONFUSION_MATRICES = {
    "Logistic Regression": pd.DataFrame(
        [[381, 159, 186],
         [2258, 1200, 1295],
         [591, 436, 5794]],
        index=CLASS_LABELS,
        columns=CLASS_LABELS,
    ),
    "Random Forest": pd.DataFrame(
        [[296, 354, 76],
         [1760, 2152, 841],
         [366, 803, 5652]],
        index=CLASS_LABELS,
        columns=CLASS_LABELS,
    ),
    "XGBoost": pd.DataFrame(
        [[201, 465, 60],
         [1135, 2920, 698],
         [265, 1034, 5522]],
        index=CLASS_LABELS,
        columns=CLASS_LABELS,
    ),
}

# ── helpers ───────────────────────────────────────────────────────────────────

def format_metric(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{float(value):.3f}"


def build_results_df() -> pd.DataFrame:
    rows = []
    for model_name, report in CLASSIFICATION_REPORTS.items():
        summary = report["_summary"]
        row = {
            "Model": model_name,
            "Macro F1": summary["Macro F1"],
            "Accuracy": summary["Accuracy"],
        }
        for class_label in CLASS_LABELS:
            for metric_name in ["Precision", "Recall", "F1"]:
                row[f"{class_label} {metric_name}"] = report[class_label][metric_name]
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(by="Macro F1", ascending=False).reset_index(drop=True)
    return df


def highlight_selected_model(row: pd.Series):
    model_name = str(row.get("Model", "")).strip().lower()
    if model_name == "xgboost":
        return ["background-color: #d1fae5; font-weight: 700;" for _ in row]
    return ["" for _ in row]


def plot_bar_chart(df, value_column, title, ylabel, highlight_model="XGBoost", figsize=(3.4, 4.0)):
    chart_df = df[["Model", value_column]].dropna().copy()
    if chart_df.empty:
        return
    chart_df = chart_df.sort_values(by=value_column, ascending=False)

    colors = [
        "#16a34a" if str(m).strip().lower() == highlight_model.strip().lower() else "#94a3b8"
        for m in chart_df["Model"]
    ]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(chart_df["Model"], chart_df[value_column], color=colors)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel("")
    ymax = chart_df[value_column].max()
    ax.set_ylim(0, min(1.0, ymax * 1.18 if ymax > 0 else 1.0))
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(range(len(chart_df["Model"])))
    ax.set_xticklabels([str(m).replace(" ", "\n") for m in chart_df["Model"]])

    for bar, value in zip(bars, chart_df[value_column]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_heatmap(matrix_df: pd.DataFrame, title: str, normalize: bool = False):
    data = matrix_df.astype(float)
    if normalize:
        row_sums = data.sum(axis=1).replace(0, np.nan)
        data = data.div(row_sums, axis=0).fillna(0) * 100

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    image = ax.imshow(data.values, aspect="auto", cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)

    max_val = data.values.max()
    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data.iat[row_idx, col_idx]
            label = f"{value:.1f}%" if normalize else f"{int(round(value))}"
            color = "white" if value > max_val * 0.5 else "black"
            ax.text(col_idx, row_idx, label, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def generate_confusion_matrix_insights(model_name: str) -> str:
    name = model_name.strip().lower()
    if name == "xgboost":
        return """
**Key insights**
- **Best Safe Zone recall** of all models: 2920/4753 correct (61.5%) — a clear improvement over the other two models
- **Overstock Risk is poorly detected**: only 201/726 correct (27.7% recall) — the worst Overstock performance across all models; 465 cases bleed into Safe Zone
- **Stockout Risk is strong**: 5522/6821 correct (80.9% recall), though slightly lower than Logistic Regression
- **Balanced error distribution**: misclassifications are spread more evenly, avoiding the extreme Safe Zone collapse seen in Logistic Regression

**Takeaway:** Best overall model — it improves Safe Zone detection significantly while keeping Stockout Risk recall high. The main weakness is Overstock Risk, which is heavily confused with Safe Zone.
"""
    elif name == "random forest":
        return """
**Key insights**
- **Overstock Risk recall is low**: 296/726 correct (40.8%) — most misclassified as Safe Zone (354), making it unreliable for catching overstock situations
- **Safe Zone is mediocre**: 2152/4753 correct (45.3% recall) — large bleed into Overstock Risk (1760 cases misclassified), the highest cross-class leak of any model
- **Stockout Risk is solid**: 5652/6821 correct (82.9% recall), second best across models
- **Systematic bias toward Overstock Risk predictions**: the Overstock column receives the most off-diagonal spill (1760 Safe Zone + 366 Stockout), inflating false positives for that class

**Takeaway:** Middle-ground model — decent Stockout detection, but Safe Zone and Overstock Risk performance is unreliable. The large Safe Zone → Overstock leak would generate excessive false overstock alerts in production.
"""
    elif name == "logistic regression":
        return """
**Key insights**
- **Worst Safe Zone detection**: only 1200/4753 correct (25.2% recall) — 2258 Safe Zone cases are wrongly predicted as Overstock Risk, by far the largest single misclassification across all models
- **Best Stockout Risk recall**: 5794/6821 correct (84.9%) — the highest of all three models, but at the cost of everything else
- **Overstock Risk is partially detected**: 381/726 correct (52.5% recall) — best Overstock recall of all models, but largely a side effect of the model over-predicting Overstock across the board
- **Heavily biased toward Overstock Risk and Stockout Risk**: the model struggles to distinguish Safe Zone, treating most samples as one of the two risk classes

**Takeaway:** Worst overall model — its high Stockout recall is misleading, as it comes from a systematic over-prediction of risk classes. It completely fails at Safe Zone detection, which would flood operations with false alarms.
"""
    return "No insights available."


# ── page header ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 14px; margin-top: 6px;">
        <span style="font-size: 42px; line-height: 1;">📊</span>
        <h1 style="
            margin: 0;
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.1;
            letter-spacing: -0.5px;
        ">
            Model Performance
        </h1>
    </div>
    <p style="
        color: #64748b;
        font-size: 1.1rem;
        margin-top: 10px;
        margin-bottom: 0;
    ">
        Experiment results · confusion matrices · model selection rationale.
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown("<hr style='margin-top: 18px; margin-bottom: 22px;'>", unsafe_allow_html=True)

# ── summary metrics ───────────────────────────────────────────────────────────

summary_col1, summary_col2, summary_col3 = st.columns(3)
summary_col1.metric("Experiments", 1)
summary_col2.metric("Runs loaded", 3)
summary_col3.metric("Best model", "XGBoost")

st.markdown("---")

# ── results table ─────────────────────────────────────────────────────────────

results_df = build_results_df()

st.markdown('<p style="font-size:1.2rem; font-weight:700; letter-spacing:0.02em;">All models</p>', unsafe_allow_html=True)

preferred_columns = ["Model", "Macro F1", "Accuracy"]
class_metric_columns = [
    col for col in results_df.columns
    if col not in {"Model", "Macro F1", "Accuracy"} and col.endswith(("Precision", "Recall", "F1"))
]
visible_columns = preferred_columns + sorted(class_metric_columns)

styled_df = (
    results_df[visible_columns]
    .style
    .format({col: format_metric for col in visible_columns if col != "Model"})
    .apply(highlight_selected_model, axis=1)
)

st.dataframe(styled_df, use_container_width=True, hide_index=True)
st.caption("Rows are sorted by Macro F1. XGBoost is highlighted in green because it was selected.")

# ── comparison charts ─────────────────────────────────────────────────────────

st.markdown("---")
st.markdown('<p style="font-size:1.2rem; font-weight:700; letter-spacing:0.02em;">Model comparison charts</p>', unsafe_allow_html=True)
st.markdown(
    "<p style='letter-spacing: 0.18em; text-transform: uppercase; color: #94a3b8; font-weight: 700; margin-bottom: 0.5rem;'>"
    "Four-metric comparison (top models)"
    "</p>",
    unsafe_allow_html=True,
)

chart_metrics = ["Macro F1", "Accuracy", "Stockout Risk F1", "Stockout Risk Recall"]

chart_columns = st.columns(len(chart_metrics))
for col, metric in zip(chart_columns, chart_metrics):
    with col:
        plot_bar_chart(results_df, metric, metric, metric, highlight_model="XGBoost", figsize=(3.4, 4.0))

st.caption("XGBoost is highlighted in green.")

# ── confusion matrices ────────────────────────────────────────────────────────

st.markdown(
    """
    <div style="margin-top: 10px;">
        <h2 style="margin-bottom: 5px;">📊 Model Confusion Matrices</h2>
        <p style="color: #64748b; font-size: 15px; margin-bottom: 0;">
            These matrices show how each model performs across risk categories.
            The diagonal represents correct predictions, while off-diagonal values highlight misclassifications.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

for model_name, matrix_df in CONFUSION_MATRICES.items():
    st.markdown("---")
    st.markdown(f"### {model_name}")

    left, right = st.columns(2)

    with left:
        plot_heatmap(matrix_df, f"{model_name} confusion matrix", normalize=False)

    with right:
        insights = generate_confusion_matrix_insights(model_name)
        st.markdown("#### 📊 Model insights")
        st.markdown(insights)
