# monitor.py
# The purpose of this file is to read prediction_logs.json
# and count predictions
# and show prediction distribution
# and check missing values
# and summarise numeric features
# and save a flattened file as monitoring_summary.csv

import json
import os
import pandas as pd

LOG_FILE = "prediction_logs.json"
SIM_FILE = "predictions.csv"


def load_json_logs(log_file=LOG_FILE):
    if not os.path.exists(log_file):
        return pd.DataFrame()

    rows = []
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    flattened = []
    for row in rows:
        ts = row.get("timestamp")
        preds = row.get("prediction", [])
        labels = row.get("prediction_label", [])
        inputs = row.get("input", {})

        if isinstance(inputs, list):
            for i, record in enumerate(inputs):
                flat = {"timestamp": ts}
                flat.update(record)
                flat["prediction"] = preds[i] if i < len(preds) else None
                flat["prediction_label"] = labels[i] if i < len(labels) else None
                flattened.append(flat)
        else:
            flat = {"timestamp": ts}
            flat.update(inputs)
            flat["prediction"] = preds[0] if preds else None
            flat["prediction_label"] = labels[0] if labels else None
            flattened.append(flat)

    return pd.DataFrame(flattened)


def main():
    logs_df = load_json_logs()
    sim_df = pd.read_csv(SIM_FILE) if os.path.exists(SIM_FILE) else pd.DataFrame()

    print("\n=== Monitoring Summary ===")
    print(f"Logged API predictions: {len(logs_df)}")
    print(f"Simulated predictions file rows: {len(sim_df)}")

    if not logs_df.empty:
        print("\n=== Logged Prediction Distribution ===")
        print(logs_df["prediction"].value_counts(dropna=False))

        print("\n=== Logged Prediction Label Distribution ===")
        if "prediction_label" in logs_df.columns:
            print(logs_df["prediction_label"].value_counts(dropna=False))

        print("\n=== Missing Values Check (Logged Data) ===")
        print(logs_df.isnull().sum())

    if not sim_df.empty:
        print("\n=== Simulated Prediction Distribution ===")
        print(sim_df["prediction"].value_counts(dropna=False))

        if "prediction_label" in sim_df.columns:
            print("\n=== Simulated Prediction Label Distribution ===")
            print(sim_df["prediction_label"].value_counts(dropna=False))

        print("\n=== Missing Values Check (Simulation File) ===")
        print(sim_df.isnull().sum())

        numeric_cols = [
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
        ]

        available_numeric = [c for c in numeric_cols if c in sim_df.columns]
        if available_numeric:
            print("\n=== Numerical Feature Summary (Simulation File) ===")
            print(sim_df[available_numeric].describe().T)

    if not logs_df.empty:
        logs_df.to_csv("monitoring_logs_flattened.csv", index=False)
    if not sim_df.empty:
        sim_df.to_csv("monitoring_summary.csv", index=False)

    print("\nMonitoring artifacts saved.")


if __name__ == "__main__":
    main()