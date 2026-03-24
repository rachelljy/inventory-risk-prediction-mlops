"""
Pydantic schemas for the IRP prediction API.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictionInput(BaseModel):
    """
    Input schema for a single inventory risk prediction.

    The caller provides today's snapshot for one store-product pair.
    Pre-computed engineered features are required because lag and rolling
    values depend on historical context the API does not store.
    """

    # ── Original numeric features ────────────────────────────────────────
    inventory_reconstructed: float = Field(..., description="Reconstructed inventory level (running balance)")
    units_sold: float = Field(..., description="Units sold today")
    demand_forecast: float = Field(..., description="Demand forecast for today")
    price: float = Field(..., description="Product price")
    discount: float = Field(..., description="Current discount (%)")
    competitor_pricing: float = Field(..., description="Competitor price for same product")
    holiday_promotion: int = Field(..., ge=0, le=1, description="1 if holiday/promotion, else 0")

    # ── Engineered features ──────────────────────────────────────────────
    inventory_change: float = Field(..., description="Reconstructed inv today - yesterday")
    inventory_change_pct: float = Field(..., description="Inventory change as % of yesterday")
    days_of_stock: float = Field(..., description="Reconstructed inv / units sold")
    inventory_vs_rolling7: float = Field(..., description="Reconstructed inv - 7-day rolling mean")
    sales_velocity: float = Field(..., description="Units sold / reconstructed inv")
    inventory_lag1: float = Field(..., description="Yesterday's reconstructed inventory")
    units_sold_lag1: float = Field(..., description="Yesterday's units sold")
    rolling7_inventory: float = Field(..., description="7-day rolling mean of reconstructed inv")
    coverage_ratio: float = Field(..., description="Reconstructed inv / demand forecast")
    forecast_error: float = Field(..., description="Units sold - demand forecast")
    order_to_inventory: float = Field(..., description="Units ordered / reconstructed inv")

    # ── Categorical features ─────────────────────────────────────────────
    category: str = Field(..., description="Product category (e.g. Groceries, Toys)")
    region: str = Field(..., description="Store region (North, South, East, West)")
    weather_condition: str = Field(..., description="Weather (Rainy, Sunny, Cloudy, Snowy)")
    seasonality: str = Field(..., description="Season (Spring, Summer, Autumn, Winter)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class PredictionOutput(BaseModel):
    """Response from the /predict endpoint."""
    risk_label: str = Field(..., description="Predicted risk: Stockout Risk, Overstock Risk, or Safe Zone")
    confidence: Optional[float] = Field(None, description="Model confidence (max class probability)")
    probabilities: Optional[dict] = Field(None, description="Per-class probabilities")


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status: str
    model_loaded: bool
    model_name: str
