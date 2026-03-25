# IRP Stage 1 — Baseline Pipeline

**Inventory Risk Predictor (IRP) | MLOps Group Project**  
*Section 1, Group 5*

---

## Overview

Stage 1 establishes the **baseline pipeline** for the Inventory Risk Predictor (IRP), a multiclass classification system designed to predict next-day inventory risk for retail products.

The objective of this stage is to:
- build the first end-to-end version of the data pipeline
- define the inventory risk labelling strategy
- apply a proper temporal train/validation/test split
- benchmark simple baseline approaches before moving to feature engineering, experiment tracking, deployment, and monitoring in later stages

This stage answers a core project question:

**Can a basic rules-based or linear model provide a reasonable starting point for predicting Stockout Risk, Overstock Risk, and Safe Zone?**

---

## Project Context

The Inventory Risk Predictor supports retailers by identifying products that may soon experience:
- **Stockout Risk**
- **Overstock Risk**
- **Safe Zone**

In business terms, the system is intended to help store managers and supply chain teams reduce missed sales from stockouts, avoid excess holding and markdown costs from overstocking, and improve inventory planning through earlier and clearer risk signals.

---

## Dataset

This stage uses a **synthetic retail inventory dataset** sourced from Kaggle.

| Property | Value |
|---|---|
| Source | Kaggle — Retail Store Inventory dataset |
| Rows | 73,100 |
| Date range | Jan 2022 to Dec 2023 |
| Granularity | Daily per Store × Product |

The dataset includes inventory, sales, demand, pricing, discounting, and contextual retail variables used to construct the baseline prediction pipeline.

---

## Stage 1 Goals

The goals of Stage 1 are to:

1. load and clean the retail inventory dataset  
2. perform initial exploratory data analysis (EDA)  
3. define a business-informed labelling strategy for inventory risk  
4. split the data chronologically to avoid look-ahead bias  
5. implement a rule-based baseline  
6. implement a Logistic Regression baseline  
7. compare both models using validation metrics  

---

## Notebook Structure

| Section | Description |
|---|---|
| **1. Data Loading and Cleaning** | Loads the CSV, standardizes data types, sorts by product/store/date, and fixes invalid values |
| **2. Exploratory Data Analysis (EDA)** | Examines distributions, category counts, and numerical summaries to understand the dataset |
| **3. Labelling Strategy** | Creates three risk classes based on inventory and demand relationships |
| **4. Temporal Train / Validation / Test Split** | Splits the data chronologically to preserve the forecasting setup |
| **5. Baseline 1 — Rule-Based Heuristic** | Uses threshold rules as direct predictions |
| **6. Baseline 2 — Logistic Regression** | Trains a multinomial logistic regression model with class balancing |
| **7. Summary of Baselines** | Compares validation results across both baselines |
| **8. Stage 1 Conclusion** | Summarizes findings and motivates improvements in later stages |

---

## Data Cleaning

The notebook performs basic preprocessing to make the dataset suitable for modelling:

- loads the raw retail inventory CSV
- converts the `Date` column to datetime format
- sorts records chronologically by store, product, and date
- checks data types and missingness
- clips negative `Demand Forecast` values where necessary
- prepares the cleaned dataset for labelling and splitting

This ensures the baseline pipeline starts from a reproducible and logically ordered dataset.

---

## Labelling Strategy

Stage 1 defines three inventory risk classes:

- **Stockout Risk**
- **Overstock Risk**
- **Safe Zone**

The labels are created using a business-style rule based on the relationship between:
- current inventory level
- expected demand
- observed sales behavior

To make the problem predictive rather than descriptive, labels are based on the **next-day risk state (`t+1`)**, while the model uses **today’s features (`t`)** as inputs.

### Risk Logic

| Label | Condition |
|---|---|
| **Stockout Risk** | Inventory Level < Demand Forecast × θ_low |
| **Overstock Risk** | Inventory Level > Demand Forecast × θ_high and Units Sold < Demand Forecast × sales_velocity |
| **Safe Zone** | Neither condition is met |

### Default Thresholds

| Parameter | Value | Purpose |
|---|---|---|
| `θ_low` | 1.2 | Flags low inventory relative to demand |
| `θ_high` | 4.5 | Flags potentially excessive inventory |
| `sales_velocity` | 0.8 | Helps confirm true overstock cases |

**Priority rule:** if both conditions could apply, **Stockout Risk takes precedence**, reflecting its higher business urgency.

---

## Temporal Split

To avoid data leakage, the dataset is split chronologically rather than randomly.

| Split | Period | Rows |
|---|---|---|
| **Train** | Jan 2022 to Jun 2023 | 54,600 |
| **Validation** | Jul 2023 to Oct 2023 | 12,300 |
| **Test** | Nov 2023 to Dec 2023 | 6,100 |

This setup reflects a realistic forecasting workflow where past data is used to predict future inventory risk.

---

## Baseline Models

### 1. Rule-Based Heuristic

The first baseline directly applies the labelling thresholds as prediction rules.

This serves as the minimum benchmark because:
- it is simple
- it is easy to interpret
- it reflects business logic without machine learning

### 2. Logistic Regression

The second baseline uses **multinomial Logistic Regression** with:
- encoded categorical features
- numerical retail features
- `class_weight="balanced"` to partially address class imbalance

This baseline tests whether a simple linear classifier can learn useful boundaries between the three inventory risk classes.

---

## Features Used in Logistic Regression

The Logistic Regression model uses 11 features:

### Numerical Features
- Inventory Level
- Units Sold
- Demand Forecast
- Price
- Discount
- Competitor Pricing
- Holiday/Promotion

### Encoded Categorical Features
- Category
- Region
- Weather Condition
- Seasonality

Categorical columns are label-encoded using training-set mappings.

---

## Validation Results

### Rule-Based Heuristic

- **Weighted F1:** 0.5906

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Overstock Risk | 0.08 | 0.08 | 0.08 |
| Safe Zone | 0.74 | 0.74 | 0.74 |
| Stockout Risk | 0.20 | 0.20 | 0.20 |

### Logistic Regression

- **Weighted F1:** 0.2947

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Overstock Risk | 0.07 | 0.43 | 0.12 |
| Safe Zone | 0.74 | 0.21 | 0.33 |
| Stockout Risk | 0.18 | 0.35 | 0.24 |

### Baseline Comparison

| Model | Weighted F1 | Key Pattern |
|---|---:|---|
| Rule-Based Heuristic | 0.5906 | Stronger overall score, but heavily benefits from Safe Zone dominance |
| Logistic Regression | 0.2947 | Better minority-class recall, but much weaker overall classification balance |

---

## Key Insights

- The **rule-based baseline** produces the strongest weighted F1, mainly because it aligns closely with the labelling logic and performs well on the dominant **Safe Zone** class.
- **Logistic Regression** improves recall for minority classes, especially **Overstock Risk**, but suffers from poor overall balance and low weighted F1.
- These results suggest that the data is **not well separated by a simple linear boundary**.
- The baseline confirms that more advanced models will be needed to improve overall performance while also capturing the business-critical risk classes.

---

## Stage 1 Conclusion

Stage 1 successfully establishes the baseline for the Inventory Risk Predictor.

It demonstrates that:
- the pipeline can move from raw retail data to labelled multiclass risk prediction
- a temporal split can be applied correctly for forecasting
- simple baselines provide a meaningful reference point for future model improvements

Among the two baselines, the **rule-based heuristic** performs better overall, while **Logistic Regression** reveals the difficulty of balancing minority-class detection with overall predictive quality.

This creates a clear motivation for later stages, which will explore stronger feature engineering and more advanced machine learning models.

---

## Notes on Synthetic Data

This project uses a synthetic dataset, so the results should be interpreted as a **pipeline and modelling benchmark**, not as production-ready retail performance.

Synthetic data may:
- reduce noise compared with real retail operations
- make some feature relationships cleaner than in practice
- simplify demand and inventory dynamics

As a result, Stage 1 is best understood as validating the **methodology**, including:
- the labelling approach
- the temporal split design
- the baseline comparison framework

---

## Requirements

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
