from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
UCI_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/502/online+retail+ii"

RAW_FILE_CANDIDATES = [
    RAW_DIR / "online_retail_II.xlsx",
    RAW_DIR / "online_retail_ii.xlsx",
    RAW_DIR / "online_retail_II.csv",
    RAW_DIR / "online_retail_ii.csv",
]

DEMO_DATA_FILE = RAW_DIR / "online_retail_ii_demo.csv"
CLEAN_TRANSACTIONS_FILE = PROCESSED_DIR / "clean_transactions.csv"
PREDICTIONS_FILE = PROCESSED_DIR / "customer_predictions.csv"
METRICS_FILE = PROCESSED_DIR / "model_metrics.csv"
FEATURE_IMPORTANCE_FILE = PROCESSED_DIR / "feature_importance.csv"
SEGMENT_SUMMARY_FILE = PROCESSED_DIR / "segment_summary.csv"
DATA_QUALITY_FILE = PROCESSED_DIR / "data_quality_summary.csv"

CHURN_MODEL_FILE = MODEL_DIR / "churn_model.joblib"
CLV_MODEL_FILE = MODEL_DIR / "clv_model.joblib"
METADATA_FILE = MODEL_DIR / "training_metadata.joblib"

RANDOM_STATE = 42
PREDICTION_DAYS = 90
GROSS_MARGIN = 0.35
CLV_MONTHS = 12

NUMERIC_FEATURES = [
    "recency_days",
    "tenure_days",
    "customer_age_days",
    "order_count",
    "line_count",
    "total_quantity",
    "unique_products",
    "gross_revenue",
    "return_revenue",
    "net_revenue",
    "avg_order_value",
    "avg_items_per_order",
    "avg_unit_price",
    "purchase_frequency_monthly",
    "return_rate",
    "formula_clv_12m",
    "rfm_score",
]

CATEGORICAL_FEATURES = ["country"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
