from __future__ import annotations

from datetime import date as Date, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from core.dataEnrichment import dataEnrichment
from core.dataFilter import dataFilter
from core.pipeline import (
    get_precip_for_dates,
    load_all_artifacts,
    load_training_csv,
)
from  core.utils import categorize_rain

def _forecast_sequence(
    model,
    scaler,
    feature_columns: List[str],
    history_df: pd.DataFrame,
    start_date: Date,
    end_date: Date,
    forecast_precip: Dict[Date, float],
) -> pd.DataFrame:
    
    # Base dataframe = history + placeholder rows for forecast horizon
    base = history_df[["date", "breakfast_footfall_pct"]].copy()
    base["date"] = pd.to_datetime(base["date"])

    horizon: List[Date] = []
    cur = start_date
    while cur <= end_date:
        horizon.append(cur)
        cur = cur + timedelta(days=1)

    placeholders = pd.DataFrame(
        {
            "date": [pd.Timestamp(d) for d in horizon],
            "breakfast_footfall_pct": [0.0] * len(horizon)
        }
    )
    running = pd.concat([base, placeholders], ignore_index=True)
    running = running.sort_values("date").reset_index(drop=True)

    out_rows = [] 
    for cur in horizon:
        ts = pd.Timestamp(cur)
        day_name = ts.day_name()
        is_weekend = int(ts.weekday() >= 5)
        precipitation = float(forecast_precip.get(cur, 0.0))

        if is_weekend == 1:
            running.loc[running["date"] == ts, "breakfast_footfall_pct"] = 0.0
            out_rows.append(
                {
                    "date": cur,
                    "day": day_name,
                    "is_weekend": is_weekend,
                    "is_holiday": 0,
                    "is_long_weekend": 0,
                    "precipitation_mm": precipitation,
                    "predicted_breakfast_footfall_pct": 0.0,
                }
            )
            continue
        
        enriched = dataEnrichment(running.copy(), verbose=False)

        target_mask = enriched["date"] == ts
        enriched.loc[target_mask, "precipitation_mm"] = precipitation
        enriched.loc[target_mask, "rain_category"] = categorize_rain(precipitation)
        
        target_row = enriched[target_mask].iloc[0]
        is_holiday = int(target_row["is_holiday"])
        is_long_weekend = int(target_row["is_long_weekend"])

        X, _Y = dataFilter(enriched, verbose=False)

        target_idx = enriched.index[target_mask][0]
        if target_idx not in X.index:
            pred_pct = 0.0
        else:
            X_target = X.loc[[target_idx]].reindex(columns=feature_columns, fill_value=0)
            X_scaled = scaler.transform(X_target)
            raw_pred = float(model.predict(X_scaled)[0])
            pred_pct = max(0.0, min(100.0, raw_pred))
        
        if is_holiday == 1:
            pred_pct = 0.0
        
        running.loc[running["date"] == ts, "breakfast_footfall_pct"] = pred_pct

        out_rows.append(
            {
                "date": cur,
                "day": day_name,
                "is_weekend": 0, # 1, already handled above
                "is_holiday": is_holiday,
                "is_long_weekend": is_long_weekend,
                "precipitation_mm": precipitation,
                "predicted_breakfast_footfall_pct": round(pred_pct, 2),
            }
        )
    
    return pd.DataFrame(out_rows)

def main() -> None:
    st.set_page_config(page_title="Footfall - Forecast", layout="wide")
    st.title("Footfall Forecast")

    art = load_all_artifacts()
    hist_df = load_training_csv()
    last_hist_date = hist_df["date"].max().date()
    start_date = last_hist_date + timedelta(days=1)
    default_end = max(Date.today(), start_date)

    model_names = list(art.models)

    st.caption(
        f"Training data ends: **{last_hist_date}** | Forcasting from **{start_date}** to **{default_end}**"
        f"Best model (training): **{art.best_model}**"
    )

    st.markdown("**Models to forecast**")
    cb_cols = st.columns(min(len(model_names), 4) or 1)

    selected_models: list[str] = []
    for i, name in enumerate(model_names):
        default_checked = (name == art.best_model)
        if cb_cols[i % len(cb_cols)].checkbox(name, value=default_checked, key=f"model_cb_{name}"):
            selected_models.append(name)

    with st.expander("Model accuracy (test split)", expanded=False):
        if art.metrics:
            metrics_df = (
                pd.DataFrame(art.metrics).T.reset_index().rename(columns={"index": "model"})
            )
            shown = (
                metrics_df[metrics_df["model"].isin(selected_models)]
                if selected_models else metrics_df
            )
            st.dataframe(
                shown.sort_values(["R2" , "RMSE", "MAE"], ascending=[False, True, True]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.write("No metrics available.")
    
    c1,c2 = st.columns(2)

    with c1:
        end_date = st.date_input("Forecast until", value=default_end, min_value=start_date)
    with c2:
        total_employees = st.number_input("Total employees", min_value=0, value=1000, step=10)
    
    if end_date < start_date:
        st.error(f"End date must be on/after {start_date}.")
        return
    
    if not selected_models:
        st.error("Please select at least one model to forecast.")
        return
    
    if st.button("Run forecast"):
        horizon = []
        cur = start_date
        while cur <= end_date:
            horizon.append(cur)
            cur = cur + timedelta(days=1)
        
        with st.spinner("Fetching precipitation forecast..."):
            precip = get_precip_for_dates(horizon, today=Date.today())
        
        per_model_frames = {}
        for name in selected_models:
            with st.spinner(f"Forecasting {len(horizon)} days with {name}..."):
                fdf = _forecast_sequence(
                    model=art.models[name],
                    scaler=art.scaler,
                    feature_columns=art.feature_columns,
                    history_df=hist_df,
                    start_date=start_date,
                    end_date=end_date,
                    forecast_precip=precip,
                )
            fdf = fdf[fdf["is_weekend"] == 0].copy()
            per_model_frames[name] = fdf
        
        chart_df = pd.DataFrame(
            {name: df.set_index("date")["predicted_breakfast_footfall_pct"] 
             for name, df in per_model_frames.items()}
        )
        chart_df.index = pd.to_datetime(chart_df.index)

        st.subheader(f"Forcast: {start_date} ➡️ {end_date} | Total Employees: {total_employees}")

        avaiable_dates = [d.date() for d in chart_df.index]
        if avaiable_dates:
            default_date = avaiable_dates[-1]
            sel_date = st.date_input(
                "Show prediction for date",
                value=default_date,
                min_value=avaiable_dates[0],
                max_value=avaiable_dates[-1],
                key="widget_date",
            )

            sel_ts = pd.Timestamp(sel_date)
            if sel_ts in chart_df.index:
                row = chart_df.loc[sel_ts]
                avg_pct = float(row.mean())
                st.markdown(f"### 🎯 Prediction for {sel_date}")

                item = list(row.items()) + [("Average (selected models)", avg_pct)]
                cols = st.columns(len(item))
                for col, (name, pct_val) in zip(cols, item):
                    pct = float(pct_val)
                    is_avg = name.startswith("Average")
                    people = int(round(pct / 100.0 * total_employees)) if total_employees > 0 else None
                    count_color = "#B8860B" if is_avg else "#0E8E50"
                    bg = "#FFF8E1" if is_avg else "#FAFAFA"
                    border = "#E0B84D" if is_avg else "#E0E0E0"
                    count_html = (
                        f"<div style='font-size: 34px;font-weight: 700;color: {count_color};'>"
                        f"{people} <span style='font-size: 16px;color:#555:'>people</span></div>"
                        if people is not None else ""
                    )
                    pct_html = (
                        f"<div style='font-size:24px;font-weight:700;color:#1F77B4;'>"
                        f"{pct:.2f}% <span style='font-size:14px;color:#555;'>footFall</span></div>"
                    )
                    col.markdown(
                        f"""
<div style="border:1px solid {border}; border-radius:10px;padding:14px;background:{bg};">
    <div style="font-size:13px;color:#666;margin-bottom:6px;">{name}</div>
    {count_html}
    {pct_html}
</div>
""",
                        unsafe_allow_html=True,
                    )
            else:
                st.warning(f"{sel_date} is a weekend / not in the forcast range.")

            st.markdown("### 📈 Forcast trend")
            if total_employees > 0:
                people_df = (chart_df / 100.0 * total_employees).round().astype(int)
                st.line_chart(people_df, x_label="Date", y_label="Predicted Breakfast Footfall (people)")
            else:
                st.info("Total employees not specified, showing percentage values only.")

            st.markdown("### 📊 Forcast details")
            table_df = chart_df.copy()
            table_df.index = table_df.index.date
            table_df.index.name = "date"
            pct_table = table_df.round(2).add_suffix(" (%)")
            if total_employees > 0:
                people_table = (table_df / 100.0 * total_employees).round().astype(int).add_suffix(" people")
                details_df = pd.concat([pct_table, people_table], axis=1)
            else:
                details_df = pct_table
            st.dataframe(details_df, use_container_width=True)


if __name__ == "__main__":
    main()

