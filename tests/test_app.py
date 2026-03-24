"""
Tests for the FastAPI serving endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


class TestHealthEndpoint:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status(self):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data

    def test_health_model_name(self):
        data = client.get("/health").json()
        assert "model_name" in data


class TestRootEndpoint:

    def test_root_redirects_to_docs(self):
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (301, 302, 307, 308)


class TestPredictEndpoint:

    VALID_PAYLOAD = {
        "inventory_reconstructed": 18.0,
        "units_sold": 127,
        "demand_forecast": 135.47,
        "price": 33.50,
        "discount": 20,
        "competitor_pricing": 29.69,
        "holiday_promotion": 0,
        "inventory_change": -72.0,
        "inventory_change_pct": -0.80,
        "days_of_stock": 0.14,
        "inventory_vs_rolling7": -33.5,
        "sales_velocity": 0.92,
        "inventory_lag1": 90.0,
        "units_sold_lag1": 115.0,
        "rolling7_inventory": 51.5,
        "coverage_ratio": 0.13,
        "forecast_error": -8.47,
        "order_to_inventory": 3.06,
        "category": "Groceries",
        "region": "North",
        "weather_condition": "Rainy",
        "seasonality": "Autumn",
    }

    def test_predict_valid_input(self):
        """If model is loaded, should return 200; if not, 503."""
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        assert response.status_code in (200, 503)

    def test_predict_returns_label(self):
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        if response.status_code == 200:
            data = response.json()
            assert data["risk_label"] in ["Stockout Risk", "Overstock Risk", "Safe Zone"]
            assert "confidence" in data
            assert "probabilities" in data

    def test_predict_missing_field(self):
        """Omitting a required field should return 422."""
        bad_payload = {k: v for k, v in self.VALID_PAYLOAD.items() if k != "units_sold"}
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_predict_invalid_holiday(self):
        """Holiday/Promotion must be 0 or 1."""
        payload = {**self.VALID_PAYLOAD, "holiday_promotion": 5}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_schema_in_docs(self):
        """OpenAPI schema should be accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "PredictionInput" in str(schema)
