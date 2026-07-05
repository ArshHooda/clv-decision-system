from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clv.config import (  # noqa: E402
    DATA_QUALITY_FILE,
    FEATURE_IMPORTANCE_FILE,
    METRICS_FILE,
    PREDICTIONS_FILE,
    SEGMENT_SUMMARY_FILE,
)


st.set_page_config(page_title="CLV Decision Dashboard", layout="wide")

EURO = "\u20ac"

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        div[data-testid="stMetric"] {
            padding: 0.85rem 0.95rem;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            background: #ffffff;
        }
        div[data-testid="stDataFrame"] {
            font-size: 0.94rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def money(value: float) -> str:
    return f"{EURO}{value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.1%}"


def filtered_predictions(data: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    if "favorite_category" not in data.columns:
        data = data.copy()
        data["favorite_category"] = "General Merchandise"

    segments = sorted(data["segment"].dropna().unique())
    countries = sorted(data["country"].dropna().unique())
    categories = sorted(data["favorite_category"].dropna().unique())

    selected_segments = st.sidebar.multiselect("Segment", segments, default=segments)
    selected_categories = st.sidebar.multiselect("Product category", categories, default=categories)
    selected_countries = st.sidebar.multiselect("Country", countries, default=countries)
    min_churn = st.sidebar.slider("Minimum churn probability", 0, 100, 0, step=5) / 100

    out = data[
        data["segment"].isin(selected_segments)
        & data["favorite_category"].isin(selected_categories)
        & data["country"].isin(selected_countries)
        & (data["churn_probability"] >= min_churn)
    ].copy()
    return out


def overview_section(data: pd.DataFrame, segment_summary: pd.DataFrame) -> None:
    segment_summary = (
        data.groupby("segment", as_index=False)
        .agg(
            customers=("CustomerID", "nunique"),
            revenue_at_risk=("revenue_at_risk", "sum"),
            predicted_clv_90d=("predicted_clv_90d", "sum"),
        )
        .sort_values("revenue_at_risk", ascending=False)
    )

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Customers", f"{data['CustomerID'].nunique():,}")
    kpi_cols[1].metric("Historical revenue", money(data["gross_revenue"].sum()))
    kpi_cols[2].metric("Predicted 90-day CLV", money(data["predicted_clv_90d"].sum()))
    kpi_cols[3].metric("Revenue at risk", money(data["revenue_at_risk"].sum()))
    kpi_cols[4].metric("Avg churn probability", pct(data["churn_probability"].mean()))

    left, right = st.columns((1.1, 0.9), gap="large")
    with left:
        fig = px.bar(
            segment_summary,
            x="segment",
            y="customers",
            color="revenue_at_risk",
            color_continuous_scale="Tealrose",
            title="Segment size and revenue at risk",
            labels={
                "segment": "Segment",
                "customers": "Customers",
                "revenue_at_risk": "Revenue at risk",
            },
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=60, b=80))
        st.plotly_chart(fig, width="stretch")
    with right:
        country = (
            data.groupby("country", as_index=False)
            .agg(revenue=("gross_revenue", "sum"), risk=("revenue_at_risk", "sum"))
            .sort_values("revenue", ascending=False)
            .head(10)
        )
        fig = px.bar(
            country,
            x="revenue",
            y="country",
            orientation="h",
            title="Top countries by revenue",
            labels={"revenue": "Revenue", "country": "Country"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        fig.update_xaxes(tickprefix=EURO)
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=60, b=40))
        st.plotly_chart(fig, width="stretch")

    st.subheader("CLV performance by category and country")
    left, right = st.columns((1, 1), gap="large")
    with left:
        category_clv = (
            data.groupby("favorite_category", as_index=False)
            .agg(
                customers=("CustomerID", "nunique"),
                predicted_clv=("predicted_clv_90d", "sum"),
                revenue_at_risk=("revenue_at_risk", "sum"),
                avg_churn=("churn_probability", "mean"),
            )
            .sort_values("predicted_clv", ascending=False)
        )
        fig = px.bar(
            category_clv,
            x="predicted_clv",
            y="favorite_category",
            orientation="h",
            color="revenue_at_risk",
            color_continuous_scale="Tealrose",
            title="Predicted CLV by product category",
            labels={
                "predicted_clv": "Predicted 90-day CLV",
                "favorite_category": "Product category",
                "revenue_at_risk": "Revenue at risk",
            },
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=470, margin=dict(l=10, r=10, t=60, b=40))
        fig.update_xaxes(tickprefix=EURO)
        st.plotly_chart(fig, width="stretch")
    with right:
        country_clv = (
            data.groupby("country", as_index=False)
            .agg(
                customers=("CustomerID", "nunique"),
                predicted_clv=("predicted_clv_90d", "sum"),
                revenue_at_risk=("revenue_at_risk", "sum"),
                avg_churn=("churn_probability", "mean"),
            )
            .sort_values("predicted_clv", ascending=False)
            .head(12)
        )
        fig = px.bar(
            country_clv,
            x="predicted_clv",
            y="country",
            orientation="h",
            color="avg_churn",
            color_continuous_scale="RdYlGn_r",
            title="Predicted CLV by country",
            labels={
                "predicted_clv": "Predicted 90-day CLV",
                "country": "Country",
                "avg_churn": "Avg churn probability",
            },
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=470, margin=dict(l=10, r=10, t=60, b=40))
        fig.update_xaxes(tickprefix=EURO)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Priority customer list")
    cols = [
        "CustomerID",
        "country",
        "segment",
        "favorite_category",
        "gross_revenue",
        "predicted_clv_90d",
        "churn_probability",
        "revenue_at_risk",
        "recommended_action",
    ]
    st.dataframe(
        data.sort_values("priority_score", ascending=False)[cols].head(30),
        width="stretch",
        height=520,
        hide_index=True,
        column_config=customer_column_config(),
    )


def customer_opportunities_section(data: pd.DataFrame) -> None:
    st.subheader("Customer opportunities")
    left, right = st.columns((1.1, 0.9), gap="large")
    with left:
        top_accounts = data.sort_values("predicted_clv_90d", ascending=False).head(20)
        st.dataframe(
            top_accounts[
                [
                    "CustomerID",
                    "country",
                    "segment",
                    "favorite_category",
                    "gross_revenue",
                    "predicted_clv_90d",
                    "favorite_product",
                    "recommended_action",
                ]
            ],
            width="stretch",
            height=520,
            hide_index=True,
            column_config=customer_column_config(),
        )
    with right:
        fig = px.scatter(
            data,
            x="gross_revenue",
            y="predicted_clv_90d",
            color="segment",
            size="order_count",
            hover_data=["CustomerID", "country", "favorite_category", "favorite_product"],
            title="Historical value vs predicted value",
            labels={
                "gross_revenue": "Historical revenue",
                "predicted_clv_90d": "Predicted 90-day CLV",
                "segment": "Segment",
            },
        )
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=40))
        fig.update_xaxes(tickprefix=EURO)
        fig.update_yaxes(tickprefix=EURO)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Product focus")
    products = (
        data.groupby(["favorite_category", "favorite_product"], as_index=False)
        .agg(customers=("CustomerID", "nunique"), predicted_value=("predicted_clv_90d", "sum"))
        .sort_values("predicted_value", ascending=False)
        .head(15)
    )
    fig = px.bar(
        products,
        x="predicted_value",
        y="favorite_product",
        color="favorite_category",
        orientation="h",
        title="Top product focus by predicted CLV",
        labels={
            "predicted_value": "Predicted 90-day CLV",
            "favorite_product": "Product",
            "favorite_category": "Category",
        },
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    fig.update_xaxes(tickprefix=EURO)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=60, b=40))
    st.plotly_chart(fig, width="stretch")


def retention_planner_section(data: pd.DataFrame) -> None:
    st.subheader("Retention and campaign planning")
    target_pct = st.slider("Target top churn-risk customers", 5, 50, 20, step=5) / 100
    save_rate = st.slider("Expected save rate", 1, 40, 15, step=1) / 100
    cost_per_customer = st.number_input("Campaign cost per customer", min_value=0.0, value=1.0, step=0.5)

    ranked = data.sort_values("revenue_at_risk", ascending=False)
    target_n = max(1, int(len(ranked) * target_pct))
    target = ranked.head(target_n)
    expected_saved = target["revenue_at_risk"].sum() * save_rate
    campaign_cost = target_n * cost_per_customer
    expected_roi = (expected_saved - campaign_cost) / campaign_cost if campaign_cost else 0

    cols = st.columns(4)
    cols[0].metric("Target customers", f"{target_n:,}")
    cols[1].metric("Revenue at risk targeted", money(target["revenue_at_risk"].sum()))
    cols[2].metric("Expected saved revenue", money(expected_saved))
    cols[3].metric("Expected ROI", f"{expected_roi:,.1f}x")

    campaign = (
        target.groupby(["segment", "favorite_category", "recommended_action"], as_index=False)
        .agg(customers=("CustomerID", "nunique"), revenue_at_risk=("revenue_at_risk", "sum"))
        .sort_values("revenue_at_risk", ascending=False)
    )
    st.dataframe(
        campaign,
        width="stretch",
        height=360,
        hide_index=True,
        column_config={
            "revenue_at_risk": st.column_config.NumberColumn("Revenue at risk", format=f"{EURO}%.0f"),
            "favorite_category": st.column_config.TextColumn("Category", width="medium"),
            "recommended_action": st.column_config.TextColumn("Recommended action", width="large"),
        },
    )

    st.subheader("Campaign contact list")
    st.dataframe(
        target[
            [
                "CustomerID",
                "segment",
                "favorite_category",
                "churn_probability",
                "revenue_at_risk",
                "recommended_action",
                "recommended_product_focus",
            ]
        ],
        width="stretch",
        height=520,
        hide_index=True,
        column_config=customer_column_config(),
    )


def diagnostics_section(data: pd.DataFrame, metrics: pd.DataFrame, feature_importance: pd.DataFrame, quality: pd.DataFrame) -> None:
    st.subheader("Model and data quality")
    left, right = st.columns((1, 1), gap="large")
    with left:
        st.caption("Model metrics")
        st.dataframe(metrics, width="stretch", hide_index=True)
        st.caption("Data quality")
        st.dataframe(quality, width="stretch", hide_index=True)
    with right:
        fig = px.bar(
            feature_importance.head(15),
            x="importance",
            y="feature",
            orientation="h",
            title="Churn model feature importance",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=60, b=40))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Prediction data")
    st.dataframe(data, width="stretch", height=560, hide_index=True, column_config=customer_column_config())
    st.download_button(
        "Download customer_predictions.csv",
        data=data.to_csv(index=False),
        file_name="customer_predictions.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("CLV, Churn, and Customer Decision Dashboard")
    st.caption("Built on UCI Online Retail II transaction data with the existing CLV project framework.")

    data = load_csv(PREDICTIONS_FILE)
    if data.empty:
        st.error("No predictions found. Run `python src/clv/train_churn.py` first.")
        st.stop()

    metrics = load_csv(METRICS_FILE)
    feature_importance = load_csv(FEATURE_IMPORTANCE_FILE)
    segment_summary_data = load_csv(SEGMENT_SUMMARY_FILE)
    quality = load_csv(DATA_QUALITY_FILE)
    data = filtered_predictions(data)

    if data.empty:
        st.warning("No customers match the current filters.")
        st.stop()

    overview_section(data, segment_summary_data)
    st.divider()
    customer_opportunities_section(data)
    st.divider()
    retention_planner_section(data)
    st.divider()
    with st.expander("Model and data quality", expanded=False):
        diagnostics_section(data, metrics, feature_importance, quality)


def customer_column_config() -> dict:
    return {
        "CustomerID": st.column_config.TextColumn("Customer ID", width="small"),
        "country": st.column_config.TextColumn("Country", width="medium"),
        "segment": st.column_config.TextColumn("Segment", width="medium"),
        "favorite_category": st.column_config.TextColumn("Category", width="medium"),
        "favorite_product": st.column_config.TextColumn("Favorite product", width="large"),
        "gross_revenue": st.column_config.NumberColumn("Historical revenue", format=f"{EURO}%.0f"),
        "predicted_clv_90d": st.column_config.NumberColumn("Predicted CLV", format=f"{EURO}%.0f"),
        "formula_clv_12m": st.column_config.NumberColumn("Formula CLV", format=f"{EURO}%.0f"),
        "churn_probability": st.column_config.ProgressColumn(
            "Churn probability",
            min_value=0,
            max_value=1,
            format="percent",
        ),
        "revenue_at_risk": st.column_config.NumberColumn("Revenue at risk", format=f"{EURO}%.0f"),
        "recommended_action": st.column_config.TextColumn("Recommended action", width="large"),
        "recommended_product_focus": st.column_config.TextColumn("Product focus", width="large"),
    }


if __name__ == "__main__":
    main()
