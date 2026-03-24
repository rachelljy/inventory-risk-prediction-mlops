# IRP Streamlit Dashboard

Five-page Streamlit interface for the Inventory Risk Predictor.

## File structure

```
streamlit_app/
├── Home.py                        ← entry point (run this)
├── style.py                       ← shared CSS + sidebar helper
├── requirements_streamlit.txt     ← pip install these
├── .streamlit/
│   └── config.toml                ← theme config
└── pages/
    ├── 1_Data_Explorer.py         ← Page 2
    ├── 2_Model_Performance.py     ← Page 3
    ├── 3_Risk_Predictor.py        ← Page 4
    └── 4_About.py                 ← Page 5
```

## Setup

```bash
# 1. Install dependencies (in addition to main requirements.txt)
pip install -r streamlit_app/requirements_streamlit.txt

# 2. Place the dataset at:
#    data/retail_store_inventory.csv  (repo root)

# 3. (Optional) Train the model so local fallback works:
python train.py

# 4. (Optional) Start the FastAPI server for live predictions:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# 5. Run the dashboard:
streamlit run streamlit_app/Home.py
```

## Pages

| # | File | Content |
|---|------|---------|
| 1 | `Home.py` | Business value, KPI cards, $740B callout, flow diagram |
| 2 | `pages/1_Data_Explorer.py` | Dataset stats, interactive filter, class distribution, label formulas |
| 3 | `pages/2_Model_Performance.py` | Model comparison table, 4-panel chart, confusion matrices (tabbed), rationale |
| 4 | `pages/3_Risk_Predictor.py` | Live demo — sliders + dropdowns → risk badge + probability bar chart |
| 5 | `pages/4_About.py` | Team cards, 5-stage methodology, dataset limitations, links |

## Risk Predictor — prediction backends

The Risk Predictor tries backends in this order:

1. **FastAPI** (`http://localhost:8000/predict`) — if the server is running and the model is loaded
2. **Local joblib** (`models/model.joblib`) — if the API is unreachable, uses the saved artefact directly
3. **Error** — if neither is available, a clear message explains what to run

The API endpoint URL is configurable in the sidebar.

## Dataset path

The Data Explorer expects the CSV at `data/retail_store_inventory.csv` relative to the repo root.
Place it there before running.
