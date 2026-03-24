# app.py
# This version of app.py will be modified to log input data, predictions and timestamp

import os
import json
import datetime
from typing import List, Union

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///../04-deployment/mlflow.db"
)
DEFAULT_RUN_ID_FILE = os.getenv(
    "RUN_ID_FILE",
    "../04-deployment/run_id.txt"
)
MODEL_URI = os.getenv("MODEL_URI")
PORT = int(os.getenv("PORT", 5001))

FEATURE_COLUMNS = [
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
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality",
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


class PredictionInput(BaseModel):
    Inventory_Level: float
    Units_Sold: float
    Units_Ordered: float
    Price: float
    Discount: float
    Units_Sold_Lag1: float
    Inventory_Change_Pct: float
    Days_of_Stock: float
    Sales_Velocity: float
    Coverage_Ratio: float
    Forecast_Error: float
    Order_to_Inventory: float
    Category: str
    Region: str
    Weather_Condition: str
    Seasonality: str

    def to_model_dict(self) -> dict:
        return {
            "Inventory Level": self.Inventory_Level,
            "Units Sold": self.Units_Sold,
            "Units Ordered": self.Units_Ordered,
            "Price": self.Price,
            "Discount": self.Discount,
            "Units_Sold_Lag1": self.Units_Sold_Lag1,
            "Inventory_Change_Pct": self.Inventory_Change_Pct,
            "Days_of_Stock": self.Days_of_Stock,
            "Sales_Velocity": self.Sales_Velocity,
            "Coverage_Ratio": self.Coverage_Ratio,
            "Forecast_Error": self.Forecast_Error,
            "Order_to_Inventory": self.Order_to_Inventory,
            "Category": self.Category,
            "Region": self.Region,
            "Weather Condition": self.Weather_Condition,
            "Seasonality": self.Seasonality,
        }


def load_labels():
    labels_path = "../04-deployment/label_classes.json"
    if os.path.exists(labels_path):
        with open(labels_path, "r") as f:
            data = json.load(f)
        return data.get("class_names", [])
    return ["Low Risk", "Medium Risk", "High Risk"]


LABELS = load_labels()


def resolve_model_uri():
    if MODEL_URI:
        return MODEL_URI
    if os.path.exists(DEFAULT_RUN_ID_FILE):
        with open(DEFAULT_RUN_ID_FILE, "r") as f:
            run_id = f.read().strip()
        if run_id:
            return f"runs:/{run_id}/model"
    raise FileNotFoundError(
        "No model URI found. Set MODEL_URI or make sure ../04-deployment/run_id.txt exists."
    )


def prepare_input(payload: Union[dict, List[dict]]) -> pd.DataFrame:
    if isinstance(payload, dict):
        df = pd.DataFrame([payload])
    elif isinstance(payload, list):
        df = pd.DataFrame(payload)
    else:
        raise ValueError("Input must be a dictionary or list of dictionaries.")

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    df = df[FEATURE_COLUMNS].copy()

    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("object")

    if df[NUMERICAL_FEATURES].isnull().any().any():
        bad_cols = df[NUMERICAL_FEATURES].columns[
            df[NUMERICAL_FEATURES].isnull().any()
        ].tolist()
        raise ValueError(
            f"These numerical columns contain invalid or missing values: {bad_cols}"
        )

    return df


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.sklearn.load_model(resolve_model_uri())

app = FastAPI(title="Inventory Risk Monitoring API", version="1.0")


@app.get("/")
def home():
    return {
        "message": "Inventory risk monitoring API is running.",
        "endpoint": "/predict",
        "logging": "enabled",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: Union[PredictionInput, List[PredictionInput]]):
    try:
        if isinstance(payload, list):
            records = [item.to_model_dict() for item in payload]
            original_payload = [item.model_dump() for item in payload]
        else:
            records = payload.to_model_dict()
            original_payload = payload.model_dump()

        X = prepare_input(records)
        preds = model.predict(X)

        decoded = []
        for p in preds:
            try:
                decoded.append(LABELS[int(p)])
            except Exception:
                decoded.append(str(p))

        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "input": original_payload,
            "prediction": [int(p) for p in preds],
            "prediction_label": decoded,
        }

        with open("prediction_logs.json", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return {
            "predictions_encoded": [int(p) for p in preds],
            "predictions_label": decoded,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))