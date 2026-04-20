from __future__ import annotations

import streamlit as st

from pipeline import (
    add_calendar_features,
    add_lag_features,
    add_rain_features_historic,
    load_training_csv,
    make_feature_matrix,
    save_artifacts,
    select_best,
    train_and_score_models,
    train_best_on_all_data,
)


def main() -> None:
    st.set_page_config(page_title="Footfall - Model Selection", layout="wide")
    st.title("Footfall Model Selection (Training App)")

    st.markdown(
        """
This app uses the last 2 years of historical data to:
- Prepare/enrich data (calendar, rain, lag/rolling)
- Split chronologically to evaluate model efficiency
- Finalize one model based on metrics
- Retrain the best model on **all** available weekday data
- Save the trained model + `StandardScaler` + feature schema for the user-facing app
"""
    )

    with st.expander("Load + preview raw data", expanded=True):
        df = load_training_csv()
        st.write(f"Rows: {len(df)} | Date range: {df['date'].min().date()} → {df['date'].max().date()}")
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("Feature engineering (calendar + rain + lag/rolling)", expanded=True):
        df = add_calendar_features(df)
        df = add_rain_features_historic(df)
        df = add_lag_features(df)
        st.write("Engineered columns preview")
        st.dataframe(
            df[["date", "breakfast_footfall_pct", "day", "month", "is_weekend", "is_holiday", "is_long_weekend", "rain_category", "rolling_7", "lag_7"]].head(12),
            use_container_width=True,
        )

    with st.expander("Build model matrix (weekdays only + one-hot)", expanded=True):
        bundle = make_feature_matrix(df)
        st.write(f"Training rows (weekdays): {len(bundle.X)}")
        st.write(f"Number of features: {bundle.X.shape[1]}")
        st.write("Feature columns")
        st.code(", ".join(bundle.feature_columns))

    st.subheader("Model efficiency (chronological split)")
    train_frac = st.slider("Train split fraction", min_value=0.6, max_value=0.9, value=0.8, step=0.05)
    results = train_and_score_models(bundle, train_frac=train_frac)
    st.dataframe(results.sort_values(["R2", "RMSE", "MAE"], ascending=[False, True, True]), use_container_width=True, hide_index=True)

    best_name = select_best(results)
    st.success(f"Selected best model: {best_name}")

    st.subheader("Finalize + save artifacts (trained on ALL data)")
    st.info(
        "For long sequential forecasts, Linear Regression can go negative and get clamped to 0. "
        "If that happens in the user app, select Random Forest or KNN here before saving."
    )
    final_model_name = st.selectbox(
        "Model to save (trained on ALL data)",
        options=list(results["model"]),
        index=list(results["model"]).index(best_name),
    )

    if st.button("Train selected model on ALL data and save"):
        art = train_best_on_all_data(bundle, final_model_name)
        out_path = save_artifacts(art)
        st.success("Saved model artifacts.")
        st.write(f"Artifact path: {out_path}")
        st.json(art.meta)


if __name__ == "__main__":
    main()

