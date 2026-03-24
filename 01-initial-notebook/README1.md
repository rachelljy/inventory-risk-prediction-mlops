# IRP Stage 1 — Baseline Pipeline

**MLOps Group Project | Section 1, Group 5**  
*Maria-Irina Popa · Jia Yi Rachel Lee · Thomas Christian Matenco · Enzo Jerez · Roberto Cummings*

---

## Overview

This notebook implements the **baseline pipeline** for the Inventory Risk Predictor (IRP) — a multiclass classification system that predicts next-day inventory risk for retail products. The goal is to establish the minimum performance bar before introducing feature engineering and ensemble models in later stages.

---

## Dataset

| Property | Value |
|---|---|
| Source | Kaggle — Retail Store Inventory dataset |
| Rows | 73,100 |
| Date range | Jan 2022 – Jan 2024 |
| Granularity | Daily per Store × Product |

---

## Notebook Structure

| Section | Description |
|---|---|
| **1. Data Loading & Cleaning** | Loads the CSV, sorts by Store/Product/Date, fixes dtypes, clips negative Demand Forecast values |
| **2. EDA** | Bar charts for categorical variables, summary statistics and boxplots for key numerical columns |
| **3. Labelling Strategy** | Derives three-class risk labels using inventory-to-demand ratio thresholds |
| **4. Temporal Split** | Chronological train/val/test split to prevent lookahead bias |
| **5. Baseline 1 — Rule-Based Heuristic** | Applies thresholds directly as predictions; sets the minimum bar |
| **6. Baseline 2 — Logistic Regression** | Multinomial LR with class weighting; assesses linear separability |
| **7. Baseline Comparison** | Side-by-side weighted F1 and recall comparison on the validation set |
| **8. Conclusion** | Summary of findings and motivation for Stages 2 and 3 |

---

## Labelling Logic

Labels are derived from the **t+1 shift** of the current-day risk label, making the model predict *tomorrow's* risk from *today's* features.

| Label | Condition |
|---|---|
| **Stockout Risk** | `Inventory Level < Demand Forecast × θ_low` |
| **Overstock Risk** | `Inventory Level > Demand Forecast × θ_high` AND `Units Sold < Demand Forecast × sales_velocity` |
| **Safe Zone** | Neither condition met |

> Stockout Risk takes precedence when both conditions are met (higher business cost).

**Default thresholds** (tunable — sensitivity analysis in Stage 2):

| Parameter | Value | Rationale |
|---|---|---|
| `θ_low` | 1.2 | ~20th percentile of inventory-to-demand ratio |
| `θ_high` | 4.5 | ~80th percentile of inventory-to-demand ratio |
| `sales_velocity` | 0.8 | Complements θ_high to filter genuine overstock |

---

## Temporal Split

| Split | Period | Approx. Size |
|---|---|---|
| Train | Jan 2022 – Jun 2023 | ~75% |
| Validation | Jul 2023 – Oct 2023 | ~17% |
| Test | Nov 2023 – Jan 2024 | ~8% (held out) |

---

## Baseline Results (Validation Set)

| Model | Weighted F1 | Stockout Recall | Overstock Recall |
|---|---|---|---|
| Rule-Based Heuristic | 0.59 | 0.20 | 0.08 |
| Logistic Regression | 0.29 | 0.35 | 0.43 |

**Key takeaway:** The rule-based model inflates its weighted F1 by defaulting to Safe Zone. Logistic Regression improves minority class recall significantly but suffers precision collapse under `class_weight='balanced'`. Both results confirm that **Stage 3 ensemble models must achieve ≥ F1 0.64** (5 pp above the rule-based bar) to justify the use of ML.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## How to Run

1. Place the dataset at `../data/retail_store_inventory.csv` relative to the notebook.
2. Run all cells top to bottom — no external dependencies beyond the packages above.
3. `RANDOM_STATE = 42` is set globally for reproducibility.

---

## Notes on Synthetic Data

The F1 scores reported here are inflated by the synthetic nature of the dataset. Real retail data would exhibit noisier demand signals, greater label ambiguity, and lower class separability. These results establish a **methodological baseline** — validating the pipeline structure, labelling logic, and temporal split discipline — not a performance benchmark. Stages 2 and 3 partially address this by introducing reconstructed inventory dynamics, but the synthetic origin remains a known limitation throughout the project.

---
