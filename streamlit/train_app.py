from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

from core.dataEnrichment import data_enrichment
from core.dataFilter import dataFilter
from core.dataSplitAndScale import dataSplitAndScale
from core.pipeline import (
    MultiModelArtifacts,
    candidate_models,
    evaluate_model,
    load_training_csv,
    save_all_artifacts,
    select_best
)


def main() -> None:
    st.set_page_config(page_title="Footfall - Model Selection", layout="wide")
    st.title("Footfall Model Selection (Training App)")

    st.markdown(
        """
This app uses the historical data to:
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
        df = data_enrichment(df)

    with st.expander("Build model matrix (weekdays only + one-hot)", expanded=True):
        X, Y = dataFilter(df)
    
    with st.expander("Train/test split + scaling (80/20  chronological)", expanded=True):
        X_train_scaled, X_test_scaled, y_train, y_test = dataSplitAndScale(X, Y)

    st.subheader("Model efficiency (chronological split)")
    rows = []
    for name, model in candidate_models().items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        m = evaluate_model(model, y_test.to_numpy(), pred)
        rows.append({"model": name, **m})

    results = pd.DataFrame(rows)
    st.dataframe(
        results.sort_values(["R2", "RMSE", "MAE"], ascending=[False, True, True]), 
        width = 'stretch',
        hide_index=True
    )
    
    best_name = select_best(results)
    st.success(f"Selected best model: {best_name}")

    st.subheader("Finalize + save artifacts (all models trained on ALL data)")
    st.info(
        "All candidate models will be retrained on the full dataset and saved together."
        "The user app can then switch between them at prediction time."
    )

    if st.button("Train all candidate models on ALL data and save artifacts"):
        scaler = StandardScaler()
        X_all_scaled = scaler.fit_transform(X)

        trained_models = {}
        for name, model in candidate_models().items():
            model.fit(X_all_scaled, Y)
            trained_models[name] = model
        
        metrics = {row["model"]: {k: float(row[k]) for k in ("MAE", "RMSE", "R2")} 
                   for _, row in results.iterrows()}
        
        art = MultiModelArtifacts(
            models=trained_models,
            scaler=scaler,
            feature_columns=list(X.columns),
            metrics=metrics,
            best_model=best_name,
            meta={
                "trained_on": "all_weekday_rows",
                "n_rows": int(len(X)),
                "date_min": str(df['date'].min().date()),
                "date_max": str(df['date'].max().date())
            }
        )
        out_path = save_all_artifacts(art)
        st.success(f"Saved {len(trained_models)} models.")
        st.write(f"Artifacts saved to: `{out_path}`")
        st.json({"best model": best_name, "models": list(trained_models), **art.meta})


if __name__ == "__main__":
    main()

