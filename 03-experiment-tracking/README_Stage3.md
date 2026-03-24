# Stage 3 – Modeling & Evaluation (MLOps)

## Overview

This stage focuses on building, comparing, and selecting machine learning models to predict inventory risk levels. Multiple classification models are trained using a consistent preprocessing pipeline, and their performance is evaluated using robust metrics beyond accuracy.

---

## Objectives

* Build end-to-end modeling pipelines
* Handle class imbalance effectively
* Train multiple classification models
* Evaluate models using appropriate metrics
* Select the best model for deployment (Stage 4)

---

## Dataset

The dataset used in this stage comes from Stage 2 (Feature Engineering), and includes:

* Train set
* Validation set
* Test set

### Target Variable

* `Risk_Label` (multi-class classification)

### Feature Types

**Numerical Features**

* Inventory Level
* Units Sold
* Units Ordered
* Price
* Discount
* Units_Sold_Lag1
* Inventory_Change_Pct
* Days_of_Stock
* Sales_Velocity
* Coverage_Ratio
* Forecast_Error
* Order_to_Inventory

**Categorical Features**

* Category
* Region
* Weather Condition
* Seasonality

---

## Preprocessing Pipeline

### Logistic Regression

* StandardScaler for numerical features
* OneHotEncoder (drop="first") for categorical features
* SMOTE applied to handle class imbalance

### Tree-Based Models (Random Forest, XGBoost)

* No scaling required for numerical features
* OneHotEncoder for categorical features
* SMOTENC used to handle mixed feature types

---

## Models Implemented

1. **Logistic Regression**

   * Class-weight balanced
   * Regularized linear classifier
   * Interpretable baseline model

2. **Random Forest**

   * Ensemble of decision trees
   * Robust to non-linear relationships
   * Handles feature interactions well

3. **XGBoost**

   * Gradient boosting model
   * Strong predictive performance
   * Uses class-weighted training via sample weights

---

## Handling Class Imbalance

* SMOTE used for Logistic Regression
* SMOTENC used for tree-based models
* Class weights computed for XGBoost

---

## Evaluation Metrics

Models are evaluated using:

* Accuracy
* Precision (macro)
* Recall (macro)
* F1-score (macro)

### Why Macro Metrics?

Macro-averaging ensures that all classes are treated equally, which is critical in imbalanced classification problems such as inventory risk prediction.

---

## Model Evaluation Process

Each model is evaluated on:

* Training set
* Validation set
* Test set

Metrics are computed consistently across all splits to:

* Detect overfitting
* Compare generalization performance

---

## Model Selection Criteria

Models are ranked based on:

1. Validation F1-score (primary)
2. Validation Recall
3. Validation Precision
4. Validation Accuracy

The best-performing model is selected for deployment in Stage 4.

---

## Results

A summary table (`results_df`) is generated containing:

* Model name
* Train / validation / test metrics
* Ranking based on validation performance

Example columns:

```id="4yqjtr"
model_name | val_f1_macro | val_recall_macro | val_precision_macro | val_accuracy
```

---

## Key Insights

* Accuracy alone is insufficient for imbalanced classification
* F1-score and recall are more informative for risk prediction
* Tree-based models typically outperform linear models in capturing complex relationships
* Proper handling of class imbalance significantly improves model performance

---

## Output

* Trained pipelines for each model
* Evaluation metrics for all models
* `results_df` for comparison
* Selected best model for deployment

---

## Conclusion

Stage 3 establishes a robust modeling framework by comparing multiple algorithms using consistent preprocessing and evaluation strategies. The selected model balances predictive performance and generalization, making it suitable for deployment in Stage 4.
