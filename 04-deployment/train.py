# train.py
# The purpose of this file is to load train/val/test parquet files
# and rebuild preprocessing pipeline
# and train the chosen model
# and evaluate it
# and log to MLflow
# and save the run_id and run_id.txt

import os
import json
import datetime
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, SMOTENC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")




BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "inventory-risk-stage3")
DATA_DIR = os.getenv(
    "DATA_DIR",
    os.path.abspath(os.path.join(BASE_DIR, "../02-features-modelling/data"))
)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", BASE_DIR)

TARGET = "Risk_Label"
DROP_COLS = [
    "Risk_Label",
    "Risk_Label_Current",
    "Store ID",
    "Product ID",
    "Date",
    "Demand Forecast",
    "Demand_Forecast_Clean",
]

NUMERICAL_FEATURES = [
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Price",
    "Discount",
    "Units_Sold_Lag1",
    "Inventory_Change_Pct",
    "Days_of_Stock",
    "Sales_Velocity",
    "Coverage_Ratio",
    "Forecast_Error",
    "Order_to_Inventory",
]

CATEGORICAL_FEATURES = [
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def load_data():
    train = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    val = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
    test = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))

    X_train = train.drop(columns=DROP_COLS).copy()[FEATURE_COLUMNS]
    X_val = val.drop(columns=DROP_COLS).copy()[FEATURE_COLUMNS]
    X_test = test.drop(columns=DROP_COLS).copy()[FEATURE_COLUMNS]

    le = LabelEncoder()
    y_train = le.fit_transform(train[TARGET].copy())
    y_val = le.transform(val[TARGET].copy())
    y_test = le.transform(test[TARGET].copy())

    return X_train, X_val, X_test, y_train, y_val, y_test, le


def build_models(X_train, y_train):
    preprocessor_logit = ColumnTransformer([
        ("num", StandardScaler(), NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    logit_pipeline = ImbPipeline([
        ("preprocessor", preprocessor_logit),
        ("smote", SMOTE(random_state=42)),
        (
            "model",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
        ),
    ])

    X_train_tree = X_train.copy()
    for col in CATEGORICAL_FEATURES:
        X_train_tree[col] = X_train_tree[col].astype("object")

    cat_indices = [X_train_tree.columns.get_loc(col) for col in CATEGORICAL_FEATURES]
    smote_nc = SMOTENC(categorical_features=cat_indices, random_state=42)

    preprocessor_tree = ColumnTransformer([
        ("num", "passthrough", NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    rf_pipeline = ImbPipeline([
        ("smote", smote_nc),
        ("preprocessor", preprocessor_tree),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    xgb_pipeline = SkPipeline([
        ("preprocessor", preprocessor_tree),
        (
            "model",
            XGBClassifier(
                objective="multi:softmax",
                num_class=3,
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="mlogloss",
            ),
        ),
    ])

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    sample_weights = np.array([class_weight_dict[y] for y in y_train])

    models = [
        ("Logistic Regression", logit_pipeline),
        ("Random Forest", rf_pipeline),
        ("XGBoost", xgb_pipeline),
    ]

    return models, sample_weights


def evaluate_split(model, X, y):
    y_pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
    }, y_pred


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X_train, X_val, X_test, y_train, y_val, y_test, le = load_data()

    # Tree pipelines expect categorical columns as object dtype
    for df in [X_train, X_val, X_test]:
        for col in CATEGORICAL_FEATURES:
            df[col] = df[col].astype("object")

    models, sample_weights = build_models(X_train, y_train)

    results = []
    all_labels = list(range(len(le.classes_)))
    class_names = list(le.classes_)

    for model_name, model in models:
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("run_time", datetime.datetime.now().isoformat())
            mlflow.set_tag("description", f"{model_name} on inventory risk data (aligned with Stage 2)")

            mlflow.log_param("sklearn_version", sklearn.__version__)
            mlflow.log_param("numerical_features", ", ".join(NUMERICAL_FEATURES))
            mlflow.log_param("categorical_features", ", ".join(CATEGORICAL_FEATURES))
            mlflow.log_param("n_classes", len(np.unique(y_train)))

            if model_name == "XGBoost":
                model.fit(X_train, y_train, model__sample_weight=sample_weights)
            else:
                model.fit(X_train, y_train)

            split_data = {
                "train": (X_train, y_train),
                "val": (X_val, y_val),
                "test": (X_test, y_test),
            }

            row = {"model_name": model_name, "run_id": run.info.run_id}
            val_pred = None

            for split_name, (X_split, y_split) in split_data.items():
                split_metrics, y_pred = evaluate_split(model, X_split, y_split)
                prefixed = {f"{split_name}_{k}": v for k, v in split_metrics.items()}
                mlflow.log_metrics(prefixed)
                row.update(prefixed)
                if split_name == "val":
                    val_pred = y_pred

            cm = confusion_matrix(y_val, val_pred, labels=all_labels)
            cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
            cm_csv = os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.csv")
            cm_png = os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
            report_txt = os.path.join(OUTPUT_DIR, f"classification_report_{model_name.lower().replace(' ', '_')}.txt")
            classes_json = os.path.join(OUTPUT_DIR, f"classes_{model_name.lower().replace(' ', '_')}.json")

            cm_df.to_csv(cm_csv)
            mlflow.log_artifact(cm_csv)

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Validation Confusion Matrix ({model_name})")
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            plt.savefig(cm_png)
            plt.close()
            mlflow.log_artifact(cm_png)

            report = classification_report(
                y_val,
                val_pred,
                labels=all_labels,
                target_names=class_names,
                zero_division=0,
            )
            with open(report_txt, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_txt)

            with open(classes_json, "w") as f:
                json.dump({"class_names": class_names, "feature_columns": FEATURE_COLUMNS}, f, indent=2)
            mlflow.log_artifact(classes_json)

            input_example = X_train.iloc[[0]].copy()
            mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example)

            print(
                f"Logged run for {model_name}: "
                f"val_f1={row['val_f1_macro']:.4f}, "
                f"val_recall={row['val_recall_macro']:.4f}, "
                f"val_precision={row['val_precision_macro']:.4f}, "
                f"val_accuracy={row['val_accuracy']:.4f}"
            )

            results.append(row)

    results_df = pd.DataFrame(results).sort_values(
        ["val_f1_macro", "val_recall_macro", "val_precision_macro", "val_accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    results_path = os.path.join(OUTPUT_DIR, "results_df.csv")
    results_df.to_csv(results_path, index=False)

    best_run = results_df.iloc[0]
    best_model_name = best_run["model_name"]
    best_run_id = best_run["run_id"]
    best_model_uri = f"runs:/{best_run_id}/model"

    with open(os.path.join(OUTPUT_DIR, "run_id.txt"), "w") as f:
        f.write(best_run_id)
    with open(os.path.join(OUTPUT_DIR, "best_model_uri.txt"), "w") as f:
        f.write(best_model_uri)
    with open(os.path.join(OUTPUT_DIR, "label_classes.json"), "w") as f:
        json.dump({"class_names": list(le.classes_)}, f, indent=2)

    print("\nBest model:", best_model_name)
    print("Best run_id:", best_run_id)
    print("Best model URI:", best_model_uri)
    print("\nResults summary:")
    print(results_df)


if __name__ == "__main__":
    main()




