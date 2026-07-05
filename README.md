# CLV Decision System

A streamlined Customer Lifetime Value and churn prediction project using the
UCI Online Retail II dataset.

Dataset source: https://archive.ics.uci.edu/dataset/502/online+retail+ii

The dashboard is designed as one end-user decision surface. It blends executive
KPIs, customer opportunities, retention planning, product focus, model quality,
and downloadable predictions without exposing internal department categories as
separate tabs.

Business dashboard additions:

- Euro-formatted KPIs and tables
- Product category filter in the sidebar
- CLV by product category
- CLV by country
- Roomier charts and wider customer tables for presentation use
- Priority customers, retention planner, product focus, and diagnostics in one flow

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- streamlit
- plotly
- joblib

The main streamlined workflow does not require DuckDB, XGBoost, SHAP, Power BI,
or any paid API.

## Quick Start

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\clv\train_churn.py
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

By default, training expects the full UCI workbook in `data/raw/`. A generated
demo dataset is available only when you intentionally pass `--demo`.

## Use The Real UCI Data

Option A: automatic download

```powershell
.\.venv\Scripts\python.exe scripts\download_uci_data.py
```

Option B: manual download

1. Download `online_retail_II.xlsx` from the UCI page:
   https://archive.ics.uci.edu/dataset/502/online+retail+ii
2. Place it in `data/raw/`.
3. Run:

```powershell
.\.venv\Scripts\python.exe src\clv\train_churn.py
```

Demo-only fallback:

```powershell
.\.venv\Scripts\python.exe src\clv\train_churn.py --demo
```

## Outputs

- `data/processed/clean_transactions.csv`
- `data/processed/customer_predictions.csv`
- `data/processed/model_metrics.csv`
- `data/processed/feature_importance.csv`
- `data/processed/segment_summary.csv`
- `artifacts/models/churn_model.joblib`
- `artifacts/models/clv_model.joblib`

## Model Summary

The pipeline uses historical transactions before a cutoff date to create
customer-level RFM and behavior features. It then labels whether each customer
purchased in the next 90 days and how much revenue they generated.

Models:

- RandomForestClassifier for churn probability
- RandomForestRegressor for 90-day predicted CLV

Business output:

- `formula_clv_12m`
- `predicted_clv_90d`
- `churn_probability`
- `revenue_at_risk`
- `segment`
- `favorite_category`
- `recommended_action`
- `recommended_product_focus`
