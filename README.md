# Inventory Risk Predictor (IRP)

**MLOps Group Project — Section 1, Group 5**
Maria-Irina Popa · Enzo Jerez · Roberto Cummings · Jia Yi Rachel Lee · Thomas Christian Matenco

An end-to-end ML system that predicts **inventory risk** (Stockout Risk, Overstock Risk, Safe Zone) one day in advance for retail operations. Built with XGBoost, FastAPI, MLflow, Docker, and CI/CD automation.

---

## Business Problem

Retail stockouts create lost-sales risk and service-level problems. Inventory planners need early warning to decide where to focus attention among hundreds of store-product combinations. This system provides a daily risk score for every product in every store, enabling proactive reordering, stock transfers, and markdown decisions.

**Key design principle:** Missing a real stockout is 5–10× more costly than generating an extra alert. The system is tuned for **high recall on Stockout Risk**.

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Raw CSV    │────▶│   train.py   │────▶│   MLflow     │     │  FastAPI     │
│   Dataset    │     │  (pipeline)  │     │  (tracking)  │     │  (app.py)    │
└──────────────┘     └──────┬───────┘     └──────────────┘     └──────┬───────┘
                            │                                         │
                            ▼                                         ▼
                     ┌──────────────┐                          ┌──────────────┐
                     │ models/      │─────────────────────────▶│  /predict    │
                     │ model.joblib │                          │  /health     │
                     └──────────────┘                          └──────────────┘
                                                                      │
                                                               ┌──────┴───────┐
                                                               │   Docker     │
                                                               │   Render     │
                                                               └──────────────┘
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/christianthomas25/inventory-risk-prediction-mlops.git
cd inventory-risk-prediction-mlops
pip install -r requirements.txt
```

### 2. Add the dataset

Place `retail_store_inventory.csv` in the `data/` folder.

### 3. Train the model

```bash
python train.py
```

This will:
- Run the full data pipeline (reconstruction, feature engineering, labelling)
- Train an XGBoost classifier with SMOTE
- Log parameters, metrics, and artifacts to MLflow
- Save the model to `models/model.joblib`

### 4. Start the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API.

### 5. Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "seasonality": "Autumn"
  }'
```

**Response:**
```json
{
  "risk_label": "Stockout Risk",
  "confidence": 0.9731,
  "probabilities": {
    "Overstock Risk": 0.0082,
    "Safe Zone": 0.0187,
    "Stockout Risk": 0.9731
  }
}
```

---

## Project Structure

```
inventory-risk-prediction-mlops/
├── config.yaml            # Single source of truth for all parameters
├── train.py               # Model training with MLflow tracking
├── app.py                 # FastAPI serving endpoint
├── Dockerfile             # Containerized service
├── render.yaml            # Render.com deployment manifest
├── requirements.txt       # Python dependencies
├── .github/
│   └── workflows/
│       └── ci-cd.yml      # Lint → Test → Build → Deploy
├── src/
│   ├── __init__.py
│   ├── pipeline.py        # Data loading, reconstruction, feature engineering
│   └── schemas.py         # Pydantic request/response models
├── tests/
│   ├── test_pipeline.py   # Unit tests for data pipeline
│   └── test_app.py        # Unit tests for API endpoints
├── models/                # Saved serving artifacts (model + encoders)
├── data/                  # Dataset (git-ignored)
└── notebooks/             # Exploratory notebooks (Stages 1–3)
```

---

## Configuration

All parameters are centralized in `config.yaml`:

| Section | Key parameters |
|---------|---------------|
| `labels` | `theta_low` (stockout threshold), `theta_high` (overstock threshold), `sales_velocity` |
| `split` | `cutoff_val`, `cutoff_test` (temporal split dates) |
| `model.xgboost` | `n_estimators`, `max_depth`, `learning_rate`, etc. |
| `serving` | `host`, `port`, `model_path` |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Redirects to interactive docs |
| `GET` | `/health` | Service and model status |
| `POST` | `/predict` | Predict inventory risk for one store-product snapshot |
| `GET` | `/docs` | Swagger UI (auto-generated) |

---

## MLflow Experiment Tracking

After training, view experiment history:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open [http://localhost:5000](http://localhost:5000) to compare runs, metrics, and artifacts.

---

## Docker

```bash
# Build
docker build -t irp .

# Run
docker run -p 8000:8000 irp

# Test
curl http://localhost:8000/health
```

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) runs on every push to `main`:

1. **Lint** — flake8 on all Python files
2. **Test** — pytest on `tests/`
3. **Build** — Docker image build
4. **Deploy** — Trigger Render.com deployment (main branch only)

---

## Testing

```bash
pytest tests/ -v
```

Tests cover:
- Inventory reconstruction logic (depletion, floor at zero)
- Feature engineering (column creation, NaN handling)
- Label application (three-class logic, precedence rules)
- Temporal splitting (no overlap, all rows accounted)
- API endpoints (health, predict, validation, error handling)

---

## Label Definitions

Labels are **business-rule proxies** applied to reconstructed inventory:

| Label | Condition | Business meaning |
|-------|-----------|-----------------|
| **Stockout Risk** | `Inventory < Demand × 1.0` | Stock insufficient to cover 1 day of forecast demand |
| **Overstock Risk** | `Inventory > Demand × 1.5` AND `Sales < Demand × 0.5` | Excess stock with low sales velocity |
| **Safe Zone** | Neither condition | Healthy inventory balance |

Labels are shifted by t+1: the model predicts **tomorrow's** risk from **today's** features.

---

## Known Limitations

1. **Proxy labels** — derived from threshold rules, not observed stockout events.
2. **Feature-label circularity** — engineered features overlap with label construction variables; the t+1 shift partially mitigates this.
3. **Overstock instability** — only 2.6% of data, with 4% day-to-day persistence; model performance on this class is limited.
4. **Synthetic dataset** — findings may not generalize to real retail operations.

---

## Future Work

- Validate against actual stockout events (zero-on-hand, lost sales logs)
- Multi-day forecast horizon (3–7 days) for more lead time
- Evidently AI drift monitoring in production
- Regression alternative (predict inventory level, derive risk post-hoc)

---

## License

Academic project — IE University, MBDS 2026.
