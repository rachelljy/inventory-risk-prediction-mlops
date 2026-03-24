# Stage 5 – Monitoring & Drift Detection (FastAPI + Evidently AI)

## Overview

This stage extends the deployed machine learning model into a monitored production system. The API is enhanced to log predictions, simulate real-world traffic, and detect potential data drift using statistical analysis and Evidently AI.

---

## Objectives

* Log real-time predictions from the deployed API
* Simulate production traffic using automated requests
* Analyze model inputs and outputs over time
* Detect data drift and monitor model behavior
* Generate drift reports using Evidently AI

---

## Project Structure

```
05-monitoring/
│
├── app.py                      # FastAPI app with logging
├── simulate.py                 # Simulates API traffic (100+ requests)
├── monitor.py                  # Monitoring analysis script
├── evidently_report.py         # Drift detection using Evidently AI
├── test_api.py                 # API testing with pytest
├── prediction_logs.json        # Logged API predictions
├── predictions.csv             # Simulated dataset
├── monitoring_summary.csv      # Monitoring output
├── monitoring_logs_flattened.csv
└── README.md
```

---

## Monitoring Pipeline

The monitoring system consists of four main components:

### 1. API Logging (`app.py`)

* FastAPI-based service
* Logs:

  * input features
  * predictions
  * timestamps
* Stored in `prediction_logs.json`

---

### 2. Simulation (`simulate.py`)

* Sends ~100 requests to `/predict`
* Generates realistic production-like data
* Saves outputs to `predictions.csv`

Run:

```bash
python simulate.py
```

---

### 3. Monitoring Analysis (`monitor.py`)

Analyzes both:

* API logs
* simulated dataset

Metrics computed:

* Prediction distribution
* Missing values check
* Feature statistics (mean, quartiles, max)

Run:

```bash
python monitor.py
```

---

### 4. Drift Detection (`evidently_report.py`)

Evidently AI is used to detect changes between:

* **Reference data** (first 50 rows)
* **Current data** (last 50 rows)

Run:

```bash
python evidently_report.py
```

Output:

```
drift_report.html
```

Open:

```bash
open drift_report.html
```

---

## Evidently AI Integration

Evidently provides:

* Statistical drift detection
* Feature distribution comparisons
* Visual reports for monitoring

Drift is detected by comparing distributions between reference and current datasets.

---

## API Testing (`test_api.py`)

Automated tests validate:

* `/health` endpoint
* `/predict` endpoint
* response structure

Run:

```bash
pytest test_api.py
```

---

## Key Results

* 100+ predictions successfully simulated
* No missing values detected in input data
* Stable numerical feature distributions observed
* Prediction distribution tracked across classes
* Drift report generated using Evidently AI

---

## Interpretation

* Input features remain within expected ranges
* No immediate signs of severe data drift
* Prediction distribution highlights model behavior patterns
* Monitoring pipeline provides early detection capabilities

---

## End-to-End Flow

1. Train model (`train.py`)
2. Deploy API (`app.py`)
3. Validate API (`test_api.py`)
4. Simulate traffic (`simulate.py`)
5. Monitor outputs (`monitor.py`)
6. Detect drift (`evidently_report.py`)

---

## Conclusion

Stage 5 introduces a complete monitoring system for the deployed model. By combining logging, simulation, statistical analysis, and Evidently AI, the system enables continuous tracking of model performance and early detection of data drift, aligning with best practices in production MLOps.
