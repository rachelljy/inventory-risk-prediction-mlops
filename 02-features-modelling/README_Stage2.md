# IRP Stage 2 — Feature Engineering & Baseline Modelling

**Inventory Risk Predictor (IRP) | MLOps Group Project**  
*Section 1, Group 5*

---

## Overview

Stage 2 expands the Inventory Risk Predictor from a simple baseline pipeline into a more structured modelling workflow through feature engineering, inventory reconstruction, scenario-based labelling, and stronger benchmark models.

The purpose of this stage is to improve how the system captures inventory behaviour before moving into experiment tracking, deployment, monitoring, and dashboard integration in later stages.

This stage answers a key project question:

**Can engineered inventory features and stronger machine learning models improve inventory risk prediction beyond the Stage 1 baselines?**

---

## Project Context

The Inventory Risk Predictor is designed to support retail decision-making by identifying products at risk of:

- **Stockout Risk**
- **Overstock Risk**
- **Safe Zone**

In business terms, this helps store managers and supply chain teams respond earlier to inventory imbalances, reduce missed sales, lower excess holding costs, and improve planning using data-driven risk signals.

---

## Dataset

This stage continues to use the synthetic retail inventory dataset from Kaggle.

| Property | Value |
|---|---|
| Source | Kaggle — Retail Store Inventory dataset |
| Initial rows | 73,100 |
| Rows after feature engineering | 72,400 |
| Date range | Jan 2022 to Jan 2024 |
| Granularity | Daily per Store × Product |

During preprocessing, the notebook clips negative demand forecast values and drops the first few rows per series where lag and rolling features cannot yet be computed. :contentReference[oaicite:1]{index=1}

---

## Stage 2 Goals

The goals of Stage 2 are to:

1. improve the raw dataset through feature engineering  
2. reconstruct inventory to create more consistent inventory dynamics  
3. compare multiple threshold scenarios for business-driven labelling  
4. build a proper next-period prediction target  
5. prepare train, validation, and test sets without temporal leakage  
6. benchmark stronger baseline models  
7. save model-ready datasets for later stages  

---

## Notebook Structure

| Section | Description |
|---|---|
| **1. Data Loading & Configuration** | Loads the dataset and defines shared thresholds, cutoffs, and constants |
| **2. Data Cleaning & Inventory Reconstruction** | Fixes negative demand values and reconstructs inventory-related signals |
| **3. Feature Engineering** | Creates lag, rolling, ratio, and forecast-based features |
| **4. Scenario Analysis & Labelling** | Compares alternative threshold settings and builds the risk target |
| **5. Temporal Split & Data Preparation** | Splits the data chronologically and prepares feature matrices |
| **6. Logistic Regression** | Tests an interpretable linear baseline |
| **7. Random Forest** | Evaluates a non-linear ensemble baseline |
| **8. XGBoost** | Evaluates a boosting-based model |
| **9. Model Comparison & Data Export** | Compares all models and saves train/val/test datasets |

---

## Data Cleaning and Reconstruction

Stage 2 introduces more disciplined preprocessing to support realistic modelling.

The notebook:

- clips negative `Demand Forecast` values to valid minimum levels
- reconstructs inventory into `Inventory_Reconstructed`
- sorts observations chronologically by product and store
- removes rows where lagged or rolling statistics are not available yet

The reconstructed inventory variable is a key design decision in this stage because it is reused consistently across feature engineering, labelling, and modelling. :contentReference[oaicite:2]{index=2}

---

## Engineered Features

Stage 2 engineers richer signals to capture demand behaviour, stock movement, and inventory position.

### Core engineered features include

**Lag features**
- `Inventory_Lag1`
- `Units_Sold_Lag1`

**Rolling and trend features**
- `Rolling7_Inventory`
- `Inventory_Change`
- `Inventory_Change_Pct`
- `Inventory_vs_Rolling7`

**Inventory and demand relationship features**
- `Days_of_Stock`
- `Sales_Velocity`
- `Coverage_Ratio`
- `Order_to_Inventory`

**Forecast-based features**
- `Forecast_Error`

### Final feature groups

**Numerical features**
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

**Categorical features**
- Category
- Region
- Weather Condition
- Seasonality

These additions make Stage 2 more behaviour-aware than Stage 1, which relied on a simpler baseline feature set. :contentReference[oaicite:3]{index=3}

---

## Scenario Analysis and Business Logic

A major addition in Stage 2 is **scenario-based labelling**. Instead of relying on only one fixed threshold configuration, the notebook compares three business logic scenarios.

| Scenario | `theta_low` | `theta_high` | `sales_vel` | Business meaning |
|---|---:|---:|---:|---|
| Scenario 1 | 1.2 | 4.5 | 0.8 | More conservative overstock flagging |
| Scenario 2 | 1.0 | 1.5 | 0.5 | More sensitive / stricter detection |
| Scenario 3 | 1.6 | 2.4 | 0.7 | Balanced compromise |

The class distributions produced by these settings differ meaningfully:

| Scenario | Overstock Risk | Safe Zone | Stockout Risk |
|---|---:|---:|---:|
| Scenario 1 | 6.04% | 39.88% | 54.08% |
| Scenario 2 | 2.57% | 47.04% | 50.39% |
| Scenario 3 | 5.14% | 34.63% | 60.23% |

This comparison helps the team choose a labelling setup that reflects the desired tradeoff between earlier risk detection and more conservative alerting. :contentReference[oaicite:4]{index=4}

---

## Target Construction

Stage 2 creates two related labels:

- **`Risk_Label_Current`** for the current-period inventory condition
- **`Risk_Label`** for the next-period prediction target

The target is shifted one step forward within each Store ID × Product ID series so the model predicts **tomorrow’s risk using today’s features**, which better matches real-world inventory decision-making. Rows without a valid future label are removed. :contentReference[oaicite:5]{index=5}

---

## Temporal Split

To avoid leakage, the data is split chronologically.

| Split | Date range | Rows |
|---|---|---:|
| Train | 2022-01-08 to 2023-06-30 | 53,900 |
| Validation | 2023-07-01 to 2023-10-31 | 12,300 |
| Test | 2023-11-01 to 2023-12-30 | 6,000 |

This preserves the forecasting structure of the task and ensures future data is not used to predict the past. :contentReference[oaicite:6]{index=6}

---

## Class Distribution

The split reveals a clear class imbalance.

### Train
- Stockout Risk: 28,916 (53.6%)
- Safe Zone: 21,693 (40.2%)
- Overstock Risk: 3,291 (6.1%)

### Validation
- Stockout Risk: 6,821 (55.5%)
- Safe Zone: 4,753 (38.6%)
- Overstock Risk: 726 (5.9%)

### Test
- Stockout Risk: 3,321 (55.4%)
- Safe Zone: 2,335 (38.9%)
- Overstock Risk: 344 (5.7%)

This imbalance is one reason Stage 2 introduces model-level handling such as class weighting and stronger ensemble baselines. :contentReference[oaicite:7]{index=7}

---

## Data Preparation

Before modelling, the pipeline:

- removes leakage-prone columns
- excludes target labels from predictors
- uses one-hot encoding for categorical variables
- scales numerical variables where appropriate
- preserves a consistent preprocessing pipeline across splits

This makes the stage more robust and easier to connect to later experiment tracking and deployment work. :contentReference[oaicite:8]{index=8}

---

## Models Evaluated

Stage 2 benchmarks three models:

### 1. Logistic Regression
A simple, interpretable linear baseline used to test whether the feature space is linearly separable.

### 2. Random Forest
A tree-based ensemble baseline used to capture non-linear interactions between inventory, demand, and sales features.

### 3. XGBoost
A boosting model used as the strongest benchmark in this stage.

The notebook compares all three models on train, validation, and test sets using accuracy and macro F1 as the main summary metrics. :contentReference[oaicite:9]{index=9}

---

## Model Results

### Validation performance

| Model | Validation Accuracy | Validation Macro F1 |
|---|---:|---:|
| Logistic Regression | 0.6465 | 0.5088 |
| Random Forest | 0.6629 | 0.5243 |
| XGBoost | 0.6957 | 0.5427 |

### Test performance

| Model | Test Accuracy | Test Macro F1 |
|---|---:|---:|
| Logistic Regression | 0.6387 | 0.4965 |
| Random Forest | 0.6578 | 0.5231 |
| XGBoost | 0.7003 | 0.5451 |

Among the three models, **XGBoost performs best overall** on both validation and test, making it the strongest Stage 2 candidate going into later project stages. :contentReference[oaicite:10]{index=10}

---

## Key Insights

- **Feature engineering improves the modelling foundation.**  
  Stage 2 adds lag, rolling, ratio, and forecast-based signals that capture inventory behaviour more realistically than the Stage 1 baseline. :contentReference[oaicite:11]{index=11}

- **Scenario design matters.**  
  Different threshold settings produce meaningfully different class balances, which affects both business interpretation and model learning. :contentReference[oaicite:12]{index=12}

- **The problem remains imbalanced.**  
  Overstock Risk is still the minority class, so achieving good performance across all classes remains challenging. :contentReference[oaicite:13]{index=13}

- **XGBoost is the strongest Stage 2 model.**  
  It achieves the best validation and test macro F1, outperforming both Logistic Regression and Random Forest. :contentReference[oaicite:14]{index=14}

- **There is still room to improve risk-class performance.**  
  Even with XGBoost, the project still needs stronger tuning and downstream integration to better support high-value business decisions around stockout and overstock detection. :contentReference[oaicite:15]{index=15}

---

## Outputs

Stage 2 produces:

- engineered train / validation / test datasets
- model-ready feature matrices
- scenario comparison outputs
- baseline model metrics
- saved split datasets for downstream stages

The notebook explicitly saves the processed train, validation, and test datasets for later use. :contentReference[oaicite:16]{index=16}

---

## Stage 2 Conclusion

Stage 2 establishes a stronger modelling foundation for the Inventory Risk Predictor by moving beyond simple baseline logic into engineered inventory features, scenario-based labelling, and non-linear ensemble models.

It shows that:
- the project can construct a richer and more realistic feature space
- business-driven labelling choices materially affect the target distribution
- stronger tree-based models outperform the linear baseline
- XGBoost is the best overall Stage 2 model and the most promising model to carry forward

This stage directly prepares the pipeline for **experiment tracking, model comparison, dashboard linkage, and deployment-oriented stages**.

---

## Notes on Synthetic Data

Because the dataset is synthetic, the results should be interpreted as a benchmark for **pipeline quality and modelling methodology**, not as production-ready retail performance.

The synthetic structure may simplify some demand and inventory relationships, but Stage 2 still provides an important validation of:
- feature engineering choices
- scenario-driven labelling logic
- temporal evaluation discipline
- baseline model comparison

---

## Requirements

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imblearn
jupyter
