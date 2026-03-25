# IRP Stage 3 — Experiment Tracking & Model Evaluation

**Inventory Risk Predictor (IRP) | MLOps Group Project**  
*Section 1, Group 5*

---

## Overview

Stage 3 focuses on **experiment tracking, model comparison, and model selection** for the Inventory Risk Predictor (IRP). Building on the engineered datasets from Stage 2, this stage recreates the same feature setup, trains three classification models, logs each run in **MLflow**, and selects the best-performing model for downstream deployment work in Stage 4.

This stage answers a key project question:

**Which model performs best when tracked and evaluated consistently across training, validation, and test sets?**

---

## Project Context

The Inventory Risk Predictor is designed to help retailers identify products that are likely to face:

- **Stockout Risk**
- **Overstock Risk**
- **Safe Zone**

In business terms, the system supports store managers and supply chain teams by improving visibility into future inventory imbalances and enabling earlier, more informed action.

---

## Stage 3 Goals

The goals of Stage 3 are to:

1. reload the Stage 2 train, validation, and test datasets  
2. recreate the same modelling feature setup  
3. train the same three benchmark models under a consistent pipeline  
4. handle class imbalance appropriately for each model family  
5. log parameters, metrics, and artifacts with **MLflow**  
6. compare model performance using macro classification metrics  
7. save the best run for use in Stage 4  

---

## Scenarios

This stage is run in two notebook variants:

- **Scenario 1**
- **Scenario 2**

Both notebooks follow the same Stage 3 structure:
- recreate the Stage 2 feature setup
- build the same three models
- evaluate them on train, validation, and test data
- log runs into MLflow
- save the best model run ID for Stage 4

In the current notebooks, both scenarios produce the same final model ranking and metrics.

---

## Input Data

Stage 3 uses the processed datasets produced in Stage 2:

- `train`
- `val`
- `test`

### Target Variable
- `Risk_Label`

### Leakage / non-model columns removed
The notebooks explicitly remove columns that should not be used as predictors:

- `Risk_Label`
- `Risk_Label_Current`
- `Store ID`
- `Product ID`
- `Date`
- `Demand Forecast`
- `Demand_Forecast_Clean`

This keeps the model aligned with a realistic prediction setting.

---

## Feature Setup

Stage 3 recreates the Stage 2 feature structure.

### Numerical features
- Inventory Level
- Units Sold
- Units Ordered
- Price
- Discount
- Competitor Pricing
- Inventory_Reconstructed
- Inventory_Lag1
- Units_Sold_Lag1
- Rolling7_Inventory
- Inventory_Change
- Inventory_Change_Pct
- Days_of_Stock
- Inventory_vs_Rolling7
- Sales_Velocity
- Coverage_Ratio
- Forecast_Error
- Order_to_Inventory

### Categorical features
- Category
- Region
- Weather Condition
- Seasonality

This ensures Stage 3 evaluates the models using the same engineered data foundation established in Stage 2.

---

## Models Implemented

Three models are trained and compared:

### 1. Logistic Regression
- standardizes numerical variables
- one-hot encodes categorical variables
- uses `class_weight="balanced"`
- serves as the interpretable linear baseline

### 2. Random Forest
- tree-based ensemble model
- captures non-linear relationships and feature interactions
- uses encoded categorical features and imbalance handling suited to mixed data

### 3. XGBoost
- gradient boosting model
- strongest predictive model in this stage
- uses class-weighted learning through computed sample weights

---

## Handling Class Imbalance

Because inventory risk prediction is imbalanced, Stage 3 uses model-specific balancing methods:

- **SMOTE** for Logistic Regression
- **SMOTENC** for Random Forest and XGBoost pipelines
- **class weights** computed for XGBoost

This is important because minority classes such as **Overstock Risk** can otherwise be under-predicted.

---

## Preprocessing Pipelines

### Logistic Regression pipeline
- `StandardScaler` for numerical features
- `OneHotEncoder(drop="first", handle_unknown="ignore")` for categorical features
- `SMOTE(random_state=42)`
- `LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)`

### Tree-based model pipelines
- categorical encoding with one-hot encoding
- no standard scaling required for tree models
- imbalance handling using `SMOTENC`
- fitted consistently across train, validation, and test evaluation

---

## Experiment Tracking with MLflow

A major addition in Stage 3 is **MLflow-based experiment tracking**.

For each model run, the notebooks log:

- model name
- evaluation metrics
- run ID
- confusion matrix artifacts
- classification report artifacts
- ranking summary

This makes the model comparison process reproducible and easier to connect to deployment and model management later in the project.

The notebook also saves the **best run ID** to:

```txt
run_id.txt
