from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clv.config import (  # noqa: E402
    CHURN_MODEL_FILE,
    CLEAN_TRANSACTIONS_FILE,
    CLV_MODEL_FILE,
    CATEGORICAL_FEATURES,
    DATA_QUALITY_FILE,
    FEATURE_IMPORTANCE_FILE,
    METADATA_FILE,
    METRICS_FILE,
    MODEL_DIR,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    PREDICTIONS_FILE,
    PROCESSED_DIR,
    RANDOM_STATE,
    SEGMENT_SUMMARY_FILE,
)
from clv.customer_features import (  # noqa: E402
    build_customer_snapshot,
    build_data_quality_summary,
    build_training_frame,
)
from clv.recommendations import add_segments_and_actions, segment_summary  # noqa: E402
from clv.retail_data import find_raw_data_file, load_transactions  # noqa: E402


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def train_churn_model(allow_demo_data: bool = False) -> tuple[Pipeline, Pipeline, pd.DataFrame]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw_file = find_raw_data_file()
    if raw_file is None or "demo" in raw_file.name.lower():
        if not allow_demo_data:
            raise FileNotFoundError(
                "Full UCI Online Retail II data was not found in data/raw. "
                "Run scripts/download_uci_data.py or place online_retail_II.xlsx in data/raw. "
                "Use --demo only when you intentionally want generated sample data."
            )
        print("Using generated demo data with Online Retail II style columns.")
        raw_file = None
    else:
        print(f"Using full raw data file: {raw_file}")

    transactions = load_transactions(data_file=raw_file, allow_demo_data=allow_demo_data)
    transactions.to_csv(CLEAN_TRANSACTIONS_FILE, index=False)
    build_data_quality_summary(transactions).to_csv(DATA_QUALITY_FILE, index=False)

    training, cutoff = build_training_frame(transactions)
    X = training[MODEL_FEATURES].copy()
    y_churn = training["churn_label"].astype(int)
    y_clv = training["future_revenue_90d"].astype(float)

    stratify = y_churn if y_churn.nunique() == 2 and y_churn.value_counts().min() >= 2 else None
    X_train, X_test, y_churn_train, y_churn_test, y_clv_train, y_clv_test = train_test_split(
        X,
        y_churn,
        y_clv,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    churn_model = Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=160,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    clv_model = Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=160,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training churn model...")
    churn_model.fit(X_train, y_churn_train)
    print("Training CLV model...")
    clv_model.fit(X_train, y_clv_train)

    churn_prob_test = positive_class_probability(churn_model, X_test)
    churn_pred_test = (churn_prob_test >= 0.5).astype(int)
    clv_pred_test = np.maximum(0, clv_model.predict(X_test))

    metrics = {
        "training_customers": int(len(training)),
        "cutoff_date": str(pd.Timestamp(cutoff).date()),
        "churn_accuracy": float(accuracy_score(y_churn_test, churn_pred_test)),
        "churn_pr_auc": float(average_precision_score(y_churn_test, churn_prob_test)),
        "clv_mae": float(mean_absolute_error(y_clv_test, clv_pred_test)),
        "clv_rmse": float(np.sqrt(mean_squared_error(y_clv_test, clv_pred_test))),
        "clv_r2": float(r2_score(y_clv_test, clv_pred_test)),
    }
    if y_churn_test.nunique() == 2:
        metrics["churn_roc_auc"] = float(roc_auc_score(y_churn_test, churn_prob_test))
    else:
        metrics["churn_roc_auc"] = float("nan")

    current_features = build_customer_snapshot(transactions)
    scored = score_customers(current_features, churn_model, clv_model)
    scored = add_segments_and_actions(scored)

    scored.to_csv(PREDICTIONS_FILE, index=False)
    pd.DataFrame([metrics]).to_csv(METRICS_FILE, index=False)
    feature_importance(churn_model).to_csv(FEATURE_IMPORTANCE_FILE, index=False)
    segment_summary(scored).to_csv(SEGMENT_SUMMARY_FILE, index=False)

    joblib.dump(churn_model, CHURN_MODEL_FILE)
    joblib.dump(clv_model, CLV_MODEL_FILE)
    joblib.dump(
        {
            "model_features": MODEL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "cutoff_date": str(pd.Timestamp(cutoff).date()),
            "metrics": metrics,
        },
        METADATA_FILE,
    )

    print("\nModel metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    print(f"\nSaved predictions: {PREDICTIONS_FILE}")
    print(f"Saved models: {MODEL_DIR}")
    return churn_model, clv_model, scored


def positive_class_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = list(model.named_steps["model"].classes_)
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    return probabilities[:, -1]


def score_customers(customer_features: pd.DataFrame, churn_model: Pipeline, clv_model: Pipeline) -> pd.DataFrame:
    scored = customer_features.copy()
    X = scored[MODEL_FEATURES].copy()
    scored["churn_probability"] = positive_class_probability(churn_model, X).clip(0, 1)
    scored["predicted_clv_90d"] = np.maximum(0, clv_model.predict(X))
    return scored


def feature_importance(churn_model: Pipeline) -> pd.DataFrame:
    model = churn_model.named_steps["model"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])

    preprocessor = churn_model.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()
    clean_names = [name.replace("numeric__", "").replace("categorical__", "") for name in feature_names]
    return (
        pd.DataFrame({"feature": clean_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLV and churn models for Online Retail II.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use generated demo data only if the full UCI file is missing.",
    )
    args = parser.parse_args()
    train_churn_model(allow_demo_data=args.demo)
