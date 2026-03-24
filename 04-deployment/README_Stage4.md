# Stage 4 – Model Deployment (FastAPI)

## Overview

In this stage, the selected machine learning model is deployed as a production-ready API using **FastAPI**. The deployment includes model training, experiment tracking with MLflow, API serving, and automated endpoint testing.

---

## Objectives

* Train and select the best model from Stage 3
* Log models and metrics using MLflow
* Deploy the model via a REST API
* Enable real-time predictions
* Validate API functionality using automated tests

---

## Project Structure

```
04-deployment/
│
├── train.py                # Model training and MLflow logging
├── app.py                  # FastAPI application
├── test_api.py             # API testing using pytest
├── run_id.txt              # Best MLflow run ID
├── best_model_uri.txt      # Model URI for deployment
├── label_classes.json      # Class labels
├── results_df.csv          # Model comparison results
└── README.md               # Documentation
```

---

## Model Training (`train.py`)

### Description

* Loads processed datasets (train, validation, test)
* Applies preprocessing pipelines:

  * StandardScaler (numerical)
  * OneHotEncoder (categorical)
* Handles class imbalance:

  * SMOTE / SMOTENC
* Trains:

  * Logistic Regression
  * Random Forest
  * XGBoost

### Evaluation Metrics

* Accuracy
* Precision (macro)
* Recall (macro)
* F1-score (macro)

### Model Selection

Models are ranked based on:

1. Validation F1-score
2. Validation Recall
3. Validation Precision
4. Validation Accuracy

---

## MLflow Integration

MLflow is used to:

* Track experiments
* Log parameters and metrics
* Store trained models
* Enable reproducibility

### Start MLflow

```bash
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./mlruns \
--host 127.0.0.1 \
--port 5001
```

Access UI:

```
http://127.0.0.1:5001
```

---

## API Deployment (`app.py`)

### Description

* Built using **FastAPI**
* Loads the best model from MLflow
* Performs input validation using Pydantic
* Returns predictions in structured JSON format

---

## Endpoints

### 1. Health Check

```
GET /health
```

Response:

```json
{"status": "ok"}
```

---

### 2. Prediction Endpoint

```
POST /predict
```

### Example Request

```json
{
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
```

### Example Response

```json
{
  "predictions_encoded": [2],
  "predictions_label": ["Stockout Risk"]
}
```

---

## API Testing (`test_api.py`)

### Description

Automated tests are implemented using **pytest** to validate API functionality.

### Tests Included

* Health endpoint availability
* Prediction endpoint response structure
* Basic sanity checks on outputs

### Run Tests

```bash
pip install pytest requests
pytest test_api.py
```

### Expected Output

```
2 passed in X.XXs
```

---

## How to Run

### 1. Train Model

```bash
python train.py
```

### 2. Start API

```bash
uvicorn app:app --host 0.0.0.0 --port 5001 --reload
```

### 3. Run Tests

```bash
pytest test_api.py
```

---

## Key Features

* End-to-end ML pipeline from training to deployment
* MLflow experiment tracking
* FastAPI-based scalable deployment
* Input validation with Pydantic
* Automated API testing using pytest

---

## Conclusion

This stage successfully transforms a trained model into a deployable and testable microservice. The use of FastAPI and automated testing ensures robustness, scalability, and readiness for real-world deployment scenarios.
