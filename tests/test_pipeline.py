"""
Tests for the IRP data pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    clean_data,
    reconstruct_inventory,
    engineer_features,
    apply_labels,
    temporal_split,
    build_feature_list,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal realistic dataframe for testing (1 store, 1 product, 20 days)."""
    np.random.seed(42)
    n = 20
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "Date": dates,
        "Store ID": "S001",
        "Product ID": "P0001",
        "Category": "Groceries",
        "Region": "North",
        "Inventory Level": np.random.randint(50, 300, n),
        "Units Sold": np.random.randint(10, 100, n),
        "Units Ordered": np.random.randint(20, 80, n),
        "Demand Forecast": np.random.uniform(30, 150, n).round(2),
        "Price": 50.0,
        "Discount": 10,
        "Weather Condition": "Sunny",
        "Holiday/Promotion": 0,
        "Competitor Pricing": 48.0,
        "Seasonality": "Winter",
    })


# ── Cleaning tests ───────────────────────────────────────────────────────────

class TestCleanData:

    def test_schema_validation_passes(self, sample_df):
        result = clean_data(sample_df)
        assert len(result) == len(sample_df)

    def test_schema_validation_fails_on_missing_column(self, sample_df):
        bad_df = sample_df.drop(columns=["Units Sold"])
        with pytest.raises(ValueError, match="Missing required columns"):
            clean_data(bad_df)

    def test_negative_demand_preserved(self, sample_df):
        """clean_data documents negative demand but does NOT mutate the column.
        Actual fix happens in engineer_features via Demand_Forecast_Clean."""
        sample_df.loc[0, "Demand Forecast"] = -5.0
        sample_df.loc[1, "Demand Forecast"] = -9.99
        result = clean_data(sample_df)
        # Raw column should be untouched
        assert result.loc[0, "Demand Forecast"] == -5.0
        assert result.loc[1, "Demand Forecast"] == -9.99

    def test_duplicates_removed(self, sample_df):
        df_with_dupes = pd.concat([sample_df, sample_df.iloc[[0]]]).reset_index(drop=True)
        result = clean_data(df_with_dupes)
        assert len(result) == len(sample_df)

    def test_output_shape_preserved(self, sample_df):
        result = clean_data(sample_df)
        assert set(result.columns) == set(sample_df.columns)


@pytest.fixture
def sample_cfg():
    return {
        "data": {"date_column": "Date", "store_column": "Store ID", "product_column": "Product ID"},
        "labels": {"theta_low": 1.0, "theta_high": 1.5, "sales_velocity": 0.5},
        "features": {
            "numeric": ["Inventory Level", "Units Sold", "Demand Forecast",
                        "Price", "Discount", "Competitor Pricing", "Holiday/Promotion"],
            "engineered": ["Inventory_Change", "Inventory_Change_Pct", "Days_of_Stock",
                           "Inventory_vs_Rolling7", "Sales_Velocity",
                           "Inventory_Lag1", "Units_Sold_Lag1", "Rolling7_Inventory"],
            "categorical": ["Category", "Region", "Weather Condition", "Seasonality"],
        },
    }


# ── Reconstruction tests ─────────────────────────────────────────────────────

class TestReconstructInventory:

    def test_first_row_unchanged(self, sample_df):
        result = reconstruct_inventory(sample_df)
        assert result["Inventory_Reconstructed"].iloc[0] == sample_df["Inventory Level"].iloc[0]

    def test_no_negative_values(self, sample_df):
        result = reconstruct_inventory(sample_df)
        assert (result["Inventory_Reconstructed"] >= 0).all()

    def test_depletion_logic(self, sample_df):
        result = reconstruct_inventory(sample_df)
        # Second row should be: first_inv - first_sold + first_ordered, floored at 0
        expected = max(
            sample_df["Inventory Level"].iloc[0]
            - sample_df["Units Sold"].iloc[0]
            + sample_df["Units Ordered"].iloc[0],
            0,
        )
        assert result["Inventory_Reconstructed"].iloc[1] == expected

    def test_output_shape(self, sample_df):
        result = reconstruct_inventory(sample_df)
        assert len(result) == len(sample_df)
        assert "Inventory_Reconstructed" in result.columns


# ── Feature engineering tests ─────────────────────────────────────────────────

class TestEngineerFeatures:

    def test_required_columns_created(self, sample_df):
        df = reconstruct_inventory(sample_df)
        result = engineer_features(df)
        expected_cols = [
            "Inventory_Lag1", "Units_Sold_Lag1", "Rolling7_Inventory",
            "Inventory_Change", "Inventory_Change_Pct", "Days_of_Stock",
            "Inventory_vs_Rolling7", "Sales_Velocity",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nans_in_output(self, sample_df):
        df = reconstruct_inventory(sample_df)
        result = engineer_features(df)
        for col in ["Inventory_Lag1", "Rolling7_Inventory", "Days_of_Stock", "Sales_Velocity"]:
            assert result[col].isna().sum() == 0, f"NaNs in {col}"

    def test_rows_dropped(self, sample_df):
        df = reconstruct_inventory(sample_df)
        result = engineer_features(df)
        # First 7 days should be dropped (rolling window warmup)
        assert len(result) < len(df)


# ── Labelling tests ──────────────────────────────────────────────────────────

class TestApplyLabels:

    def test_three_classes(self, sample_df):
        df = reconstruct_inventory(sample_df)
        df = engineer_features(df)
        # Need Demand_Forecast_Clean for labelling (created in engineer_features)
        result = apply_labels(df)
        valid_labels = {"Stockout Risk", "Overstock Risk", "Safe Zone"}
        assert set(result["Risk_Label"].unique()).issubset(valid_labels)

    def test_no_nan_labels(self, sample_df):
        df = reconstruct_inventory(sample_df)
        df = engineer_features(df)
        result = apply_labels(df)
        assert result["Risk_Label"].isna().sum() == 0

    def test_stockout_precedence(self):
        """When inventory is very low, stockout should take precedence."""
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-08", periods=3),
            "Store ID": "S001",
            "Product ID": "P0001",
            "Inventory_Reconstructed": [5, 5, 5],
            "Demand_Forecast_Clean": [100, 100, 100],
            "Units Sold": [2, 2, 2],
        })
        result = apply_labels(df, theta_low=1.0, theta_high=1.5, sales_vel=0.5)
        # All should be stockout (5 < 100 * 1.0)
        current_labels = result["Risk_Label_Current"].unique()
        assert "Stockout Risk" in current_labels


# ── Temporal split tests ─────────────────────────────────────────────────────

class TestTemporalSplit:

    def test_no_overlap(self, sample_df):
        train, val, test = temporal_split(sample_df, "2023-01-10", "2023-01-15")
        if len(train) > 0 and len(val) > 0:
            assert train["Date"].max() < val["Date"].min()
        if len(val) > 0 and len(test) > 0:
            assert val["Date"].max() < test["Date"].min()

    def test_all_rows_accounted(self, sample_df):
        train, val, test = temporal_split(sample_df, "2023-01-10", "2023-01-15")
        assert len(train) + len(val) + len(test) == len(sample_df)


# ── Feature list tests ───────────────────────────────────────────────────────

class TestBuildFeatureList:

    def test_correct_count(self, sample_cfg):
        features = build_feature_list(sample_cfg)
        expected = 7 + 8 + 4  # numeric + engineered + categorical_enc
        assert len(features) == expected

    def test_encoded_suffix(self, sample_cfg):
        features = build_feature_list(sample_cfg)
        enc_features = [f for f in features if f.endswith("_enc")]
        assert len(enc_features) == 4
