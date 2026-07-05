from __future__ import annotations

import numpy as np
import pandas as pd

from clv.config import CLV_MONTHS, GROSS_MARGIN, PREDICTION_DAYS


def percentile_score(series: pd.Series, high_is_good: bool = True) -> pd.Series:
    values = series.astype(float).fillna(0)
    ranks = values.rank(pct=True, method="average", ascending=high_is_good)
    return np.ceil(ranks * 5).clip(1, 5).astype(int)


def favorite_product(transactions: pd.DataFrame) -> pd.DataFrame:
    positive = transactions[transactions["Quantity"] > 0].copy()
    if positive.empty:
        return pd.DataFrame(columns=["CustomerID", "favorite_product", "favorite_category"])

    product_counts = (
        positive.groupby(["CustomerID", "Description", "product_category"], as_index=False)["Quantity"]
        .sum()
        .sort_values(["CustomerID", "Quantity"], ascending=[True, False])
    )
    return (
        product_counts.drop_duplicates("CustomerID")[["CustomerID", "Description", "product_category"]]
        .rename(columns={"Description": "favorite_product", "product_category": "favorite_category"})
    )


def build_customer_snapshot(
    transactions: pd.DataFrame,
    as_of_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if transactions.empty:
        raise ValueError("No transactions available for feature building.")

    data = transactions.copy()
    if as_of_date is None:
        as_of_date = data["InvoiceDate"].max() + pd.Timedelta(days=1)
    else:
        as_of_date = pd.Timestamp(as_of_date)

    positive = data[data["Quantity"] > 0].copy()
    if positive.empty:
        positive = data.copy()

    invoice_level = (
        positive.groupby(["CustomerID", "InvoiceNo"], as_index=False)
        .agg(
            invoice_date=("InvoiceDate", "min"),
            order_revenue=("gross_revenue", "sum"),
            order_quantity=("Quantity", "sum"),
            country=("Country", "last"),
        )
        .sort_values("invoice_date")
    )

    customer = (
        invoice_level.groupby("CustomerID", as_index=False)
        .agg(
            first_purchase=("invoice_date", "min"),
            last_purchase=("invoice_date", "max"),
            order_count=("InvoiceNo", "nunique"),
            gross_revenue=("order_revenue", "sum"),
            total_quantity=("order_quantity", "sum"),
            country=("country", "last"),
        )
    )

    line_features = (
        data.groupby("CustomerID", as_index=False)
        .agg(
            line_count=("InvoiceNo", "size"),
            unique_products=("StockCode", "nunique"),
            return_revenue=("return_revenue", "sum"),
            net_revenue=("net_revenue", "sum"),
        )
    )
    customer = customer.merge(line_features, on="CustomerID", how="left")
    customer = customer.merge(favorite_product(data), on="CustomerID", how="left")

    customer["recency_days"] = (as_of_date - customer["last_purchase"]).dt.days.clip(lower=0)
    customer["tenure_days"] = (customer["last_purchase"] - customer["first_purchase"]).dt.days.clip(lower=0)
    customer["customer_age_days"] = (as_of_date - customer["first_purchase"]).dt.days.clip(lower=1)
    customer["avg_order_value"] = customer["gross_revenue"] / customer["order_count"].clip(lower=1)
    customer["avg_items_per_order"] = customer["total_quantity"] / customer["order_count"].clip(lower=1)
    customer["avg_unit_price"] = customer["gross_revenue"] / customer["total_quantity"].clip(lower=1)
    customer["purchase_frequency_monthly"] = customer["order_count"] / (customer["customer_age_days"] / 30.0)
    customer["return_rate"] = customer["return_revenue"] / customer["gross_revenue"].replace(0, np.nan)
    customer["return_rate"] = customer["return_rate"].replace([np.inf, -np.inf], np.nan).fillna(0)
    customer["formula_clv_12m"] = (
        customer["avg_order_value"]
        * customer["purchase_frequency_monthly"]
        * CLV_MONTHS
        * GROSS_MARGIN
    )

    customer["recency_score"] = percentile_score(customer["recency_days"], high_is_good=False)
    customer["frequency_score"] = percentile_score(customer["order_count"], high_is_good=True)
    customer["monetary_score"] = percentile_score(customer["gross_revenue"], high_is_good=True)
    customer["rfm_score"] = (
        customer["recency_score"] + customer["frequency_score"] + customer["monetary_score"]
    )

    numeric_columns = customer.select_dtypes(include=[np.number]).columns
    customer[numeric_columns] = customer[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    customer["favorite_product"] = customer["favorite_product"].fillna("General giftware")
    customer["favorite_category"] = customer["favorite_category"].fillna("General Merchandise")

    return customer


def build_training_frame(
    transactions: pd.DataFrame,
    prediction_days: int = PREDICTION_DAYS,
    cutoff_fraction: float = 0.78,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    min_date = transactions["InvoiceDate"].min()
    max_date = transactions["InvoiceDate"].max()
    cutoff = min_date + (max_date - min_date) * cutoff_fraction
    prediction_end = cutoff + pd.Timedelta(days=prediction_days)

    history = transactions[transactions["InvoiceDate"] <= cutoff].copy()
    future = transactions[
        (transactions["InvoiceDate"] > cutoff) & (transactions["InvoiceDate"] <= prediction_end)
    ].copy()

    features = build_customer_snapshot(history, as_of_date=cutoff + pd.Timedelta(days=1))
    future_positive = future[future["Quantity"] > 0].copy()
    future_revenue = future_positive.groupby("CustomerID")["gross_revenue"].sum().rename("future_revenue_90d")
    future_orders = future_positive.groupby("CustomerID")["InvoiceNo"].nunique().rename("future_orders_90d")

    training = features.merge(future_revenue, on="CustomerID", how="left")
    training = training.merge(future_orders, on="CustomerID", how="left")
    training["future_revenue_90d"] = training["future_revenue_90d"].fillna(0)
    training["future_orders_90d"] = training["future_orders_90d"].fillna(0)
    training["churn_label"] = (training["future_orders_90d"] == 0).astype(int)

    return training, cutoff


def build_data_quality_summary(transactions: pd.DataFrame, raw_rows: int | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "clean_transaction_rows", "value": len(transactions)},
            {"metric": "raw_rows_loaded", "value": raw_rows if raw_rows is not None else len(transactions)},
            {"metric": "customers", "value": transactions["CustomerID"].nunique()},
            {"metric": "date_min", "value": str(transactions["InvoiceDate"].min().date())},
            {"metric": "date_max", "value": str(transactions["InvoiceDate"].max().date())},
            {"metric": "cancelled_or_return_lines", "value": int(transactions["is_cancelled"].sum())},
            {"metric": "countries", "value": transactions["Country"].nunique()},
            {"metric": "product_categories", "value": transactions["product_category"].nunique()},
        ]
    )
