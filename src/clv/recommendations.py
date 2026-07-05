from __future__ import annotations

import numpy as np
import pandas as pd


def add_segments_and_actions(scored: pd.DataFrame) -> pd.DataFrame:
    data = scored.copy()
    high_clv = data["predicted_clv_90d"].quantile(0.75)
    high_value = data["gross_revenue"].quantile(0.75)

    conditions = [
        (data["rfm_score"] >= 13) & (data["churn_probability"] < 0.40),
        (data["predicted_clv_90d"] >= high_clv) & (data["churn_probability"] >= 0.55),
        (data["gross_revenue"] >= high_value) & (data["churn_probability"] < 0.55),
        (data["frequency_score"] >= 4) & (data["churn_probability"] < 0.55),
        (data["customer_age_days"] <= 90) & (data["recency_days"] <= 45),
        (data["churn_probability"] >= 0.65) | (data["recency_days"] >= 180),
    ]
    choices = [
        "Champions",
        "High Value At Risk",
        "Big Spenders",
        "Loyal Customers",
        "New Customers",
        "At Risk",
    ]
    data["segment"] = np.select(conditions, choices, default="Need Attention")
    data["recommended_action"] = data.apply(recommend_action, axis=1)
    data["recommended_product_focus"] = data["favorite_product"].fillna("General giftware")
    data["recommended_category_focus"] = data["favorite_category"].fillna("General Merchandise")
    data["revenue_at_risk"] = data["churn_probability"] * data["predicted_clv_90d"]
    data["priority_score"] = data["revenue_at_risk"] + (0.25 * data["predicted_clv_90d"])
    data["churn_risk_label"] = pd.cut(
        data["churn_probability"],
        bins=[-0.001, 0.35, 0.65, 1.0],
        labels=["Low", "Medium", "High"],
    )
    return data.sort_values("priority_score", ascending=False).reset_index(drop=True)


def recommend_action(row: pd.Series) -> str:
    segment = row["segment"]
    if segment == "High Value At Risk":
        return "VIP win-back offer with personal account follow-up"
    if segment == "Champions":
        return "Early-access loyalty reward and premium cross-sell"
    if segment == "Big Spenders":
        return "Account-managed bundle or volume discount"
    if segment == "Loyal Customers":
        return "Personalized cross-sell based on favorite product"
    if segment == "New Customers":
        return "Welcome journey and second-order incentive"
    if segment == "At Risk":
        return "Reactivation campaign with limited-time incentive"
    return "Review behavior and test a low-cost recommendation"


def segment_summary(scored: pd.DataFrame) -> pd.DataFrame:
    return (
        scored.groupby("segment", as_index=False)
        .agg(
            customers=("CustomerID", "nunique"),
            revenue=("gross_revenue", "sum"),
            predicted_clv_90d=("predicted_clv_90d", "sum"),
            revenue_at_risk=("revenue_at_risk", "sum"),
            avg_churn_probability=("churn_probability", "mean"),
        )
        .sort_values("revenue_at_risk", ascending=False)
    )
