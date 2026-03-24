# test_api.py

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def sample_payload():
    return {
        "Inventory_Level": 120,
        "Units_Sold": 35,
        "Units_Ordered": 40,
        "Price": 19.99,
        "Discount": 0.10,
        "Units_Sold_Lag1": 30,
        "Inventory_Change_Pct": 0.08,
        "Days_of_Stock": 12,
        "Sales_Velocity": 2.9,
        "Coverage_Ratio": 1.4,
        "Forecast_Error": 3.5,
        "Order_to_Inventory": 0.33,
        "Category": "Electronics",
        "Region": "North",
        "Weather_Condition": "Sunny",
        "Seasonality": "Summer",
    }


def test_home_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200

    data = resp.json()
    assert "message" in data
    assert "endpoints" in data


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200

    data = resp.json()
    assert data.get("status") == "ok"


def test_predict_endpoint():
    payload = sample_payload()

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert "predictions_encoded" in data
    assert "predictions_label" in data
    assert isinstance(data["predictions_encoded"], list)
    assert isinstance(data["predictions_label"], list)
    assert len(data["predictions_encoded"]) == 1
    assert len(data["predictions_label"]) == 1


def test_predict_missing_field():
    payload = sample_payload()
    payload.pop("Price")

    resp = client.post("/predict", json=payload)
    assert resp.status_code in [400, 422]