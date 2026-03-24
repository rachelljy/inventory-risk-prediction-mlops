# test_api.py

import requests

BASE_URL = "http://127.0.0.1:5001"


def test_health_endpoint():
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code}"
    
    data = resp.json()
    assert data.get("status") == "ok"


def test_predict_endpoint():
    payload = {
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
        "Seasonality": "Summer"
    }

    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code}"

    data = resp.json()

    # structure checks
    assert "predictions_encoded" in data
    assert "predictions_label" in data

    # sanity checks
    assert isinstance(data["predictions_encoded"], list)
    assert len(data["predictions_encoded"]) > 0