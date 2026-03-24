# app.py
# The purpose of this file is to read best_model_uri.txt / run_id.txt
# and load the trained model from MLflow
# and define FastAPI app
# and define input schema
# and expose /, /health, /predict

import os
import json
from typing import List, Union

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}",
)
DEFAULT_MODEL_URI_FILE = os.getenv(
    "MODEL_URI_FILE",
    os.path.join(BASE_DIR, "best_model_uri.txt"),
)
DEFAULT_RUN_ID_FILE = os.getenv(
    "RUN_ID_FILE",
    os.path.join(BASE_DIR, "run_id.txt"),
)
MODEL_URI = os.getenv("MODEL_URI")

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

INTEGER_FEATURES = [
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Discount",
]

FLOAT_FEATURES = [
    "Price",
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
    Inventory_Level: int
    Units_Sold: int
    Units_Ordered: int
    Price: float
    Discount: int
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


def load_labels() -> List[str]:
    label_path = os.path.join(BASE_DIR, "label_classes.json")
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            data = json.load(f)
        return data.get("class_names", [])
    return ["Low Risk", "Medium Risk", "High Risk"]


LABELS = load_labels()


def resolve_model_uri() -> str:
    if MODEL_URI:
        return MODEL_URI

    if os.path.exists(DEFAULT_MODEL_URI_FILE):
        with open(DEFAULT_MODEL_URI_FILE, "r") as f:
            model_uri = f.read().strip()
        if model_uri:
            return model_uri

    if os.path.exists(DEFAULT_RUN_ID_FILE):
        with open(DEFAULT_RUN_ID_FILE, "r") as f:
            run_id = f.read().strip()
        if run_id:
            return f"runs:/{run_id}/model"

    raise FileNotFoundError(
        "No model URI found. Set MODEL_URI env var or place best_model_uri.txt "
        "or run_id.txt next to app.py."
    )


def prepare_input(payload: Union[dict, List[dict]]) -> pd.DataFrame:
    if isinstance(payload, dict):
        df = pd.DataFrame([payload])
    elif isinstance(payload, list):
        df = pd.DataFrame(payload)
    else:
        raise ValueError("Input must be a dictionary or a list of dictionaries.")

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    df = df[FEATURE_COLUMNS].copy()

    for col in INTEGER_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in FLOAT_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_int_cols = df[INTEGER_FEATURES].columns[df[INTEGER_FEATURES].isnull().any()].tolist()
    bad_float_cols = df[FLOAT_FEATURES].columns[df[FLOAT_FEATURES].isnull().any()].tolist()

    if bad_int_cols or bad_float_cols:
        raise ValueError(
            f"These numerical columns contain invalid or missing values: "
            f"{bad_int_cols + bad_float_cols}"
        )

    for col in INTEGER_FEATURES:
        df[col] = df[col].astype("int64")

    for col in FLOAT_FEATURES:
        df[col] = df[col].astype("float64")

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("object")

    return df


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model(resolve_model_uri())

app = FastAPI(title="Inventory Risk API", version="1.0")


@app.get("/")
def home():
    return {
        "message": "Inventory risk model is running.",
        "endpoint": "/predict",
        "required_features": [
            "Inventory_Level",
            "Units_Sold",
            "Units_Ordered",
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
            "Weather_Condition",
            "Seasonality",
        ],
        "note": "Discount is expected as an integer with the current saved MLflow schema.",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: Union[PredictionInput, List[PredictionInput]]):
    try:
        if isinstance(payload, list):
            records = [item.to_model_dict() for item in payload]
        else:
            records = payload.to_model_dict()

        X = prepare_input(records)
        preds = model.predict(X)

        decoded = []
        encoded = []

        for p in preds:
            try:
                p_int = int(p)
                encoded.append(p_int)
                decoded.append(LABELS[p_int] if 0 <= p_int < len(LABELS) else str(p))
            except Exception:
                encoded.append(str(p))
                decoded.append(str(p))

        return {
            "predictions_encoded": encoded,
            "predictions_label": decoded,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))