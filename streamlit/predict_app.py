from __future__ import annotations

from datetime import date as Date, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from pipeline import (
    add_calendar_features,
    add_lag_features,
    add_rain_features_historic,
    build_feature_row_for_date,
    get_precip_for_dates,
    load_artifacts,
    load_training_csv,
)


def _forecast_sequence(
    model,
    scaler,
    feature_columns: List[str],
    history_df: pd.DataFrame,
    start_date: Date,
    end_date: Date,
    forecast_precip: Dict[Date, float],
) -> pd.DataFrame:
    """
    Forecast day-by-day so lag/rolling use prior predictions.
    We keep the full daily history (including weekends) because the provided CSV contains weekend 0%,
    and the lag/rolling logic in this project is based on that full series.
    """
    hist = history_df.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date").reset_index(drop=True)

    # Build an in-memory series to compute rolling/lag as we append predictions
    series_dates = list(hist["date"].dt.date)
    series_vals = list(hist["breakfast_footfall_pct"].astype(float))

    out_rows = []
    cur = start_date
    while cur <= end_date:
        ts = pd.Timestamp(cur)
        is_weekend = int(ts.weekday() >= 5)

        # lag/rolling based on *previous* days (including weekends)
        if len(series_vals) >= 7:
            rolling_7 = float(np.mean(series_vals[-7:]))
            lag_7 = float(series_vals[-7])
        else:
            # fallback for very small histories
            rolling_7 = float(np.mean(series_vals)) if series_vals else 0.0
            lag_7 = float(series_vals[0]) if series_vals else 0.0

        precipitation = float(forecast_precip.get(cur, 0.0))

        if is_weekend == 1:
            raw_pred = 0.0
            pred_pct = 0.0
            debug = {"note": "Weekend -> forced 0.0%"}
        else:
            X_one, debug = build_feature_row_for_date(
                feature_columns=feature_columns,
                target_date=cur,
                precipitation_mm=precipitation,
                rolling_7=rolling_7,
                lag_7=lag_7,
            )
            X_scaled = scaler.transform(X_one)
            raw_pred = float(model.predict(X_scaled)[0])
            pred_pct = max(0.0, min(100.0, raw_pred))

        out_rows.append(
            {
                "date": cur,
                "is_weekend": is_weekend,
                "precipitation_mm": precipitation,
                "raw_pred_pct": raw_pred,
                "predicted_breakfast_footfall_pct": pred_pct,
                "rolling_7_used": rolling_7,
                "lag_7_used": lag_7,
                **debug,
            }
        )

        # Append to history for next day lag/rolling
        series_dates.append(cur)
        series_vals.append(float(pred_pct))
        cur = cur + timedelta(days=1)

    return pd.DataFrame(out_rows)


def main() -> None:
    st.set_page_config(page_title="Footfall - Forecast", layout="wide")
    st.title("Footfall Forecast (User App)")

    st.markdown(
        """
This app is user-facing:
- Loads the saved **best model + scaler**
- Forecasts **sequentially** from the end of training data up to today (or a selected end date)
- Computes `lag_7` and `rolling_7` using history + prior predictions
- Pulls **precipitation forecast** from Open‑Meteo for the forecast horizon
"""
    )

    art = load_artifacts()
    st.caption(f"Loaded model: {art.model_name}")

    with st.expander("Load + enrich training history (for lag/rolling base)", expanded=False):
        df = load_training_csv()
        df = add_calendar_features(df)
        df = add_rain_features_historic(df)
        df = add_lag_features(df)
        st.write(f"History rows: {len(df)} | last date: {df['date'].max().date()}")
        st.dataframe(df.tail(10), use_container_width=True)

    last_hist_date = load_training_csv()["date"].max().date()
    default_end = Date.today()

    c1, c2, c3 = st.columns(3)
    with c1:
        end_date = st.date_input("Forecast until", value=default_end, min_value=Date(2026, 1, 1))
    with c2:
        total_employees = st.number_input("Total employees", min_value=0, value=1000, step=10)
    with c3:
        include_weekends = st.checkbox("Show weekends (predicted as 0%)", value=True)

    # We forecast sequentially from 01/01/2026 to avoid stale lag/rolling issues.
    start_date = Date(2026, 1, 1)
    if end_date < start_date:
        st.error("End date must be on/after 01/01/2026.")
        return
    if st.button("Run forecast"):
        horizon = []
        cur = start_date
        while cur <= end_date:
            horizon.append(cur)
            cur = cur + timedelta(days=1)

        with st.spinner("Fetching precipitation forecast..."):
            precip = get_precip_for_dates(horizon, today=Date.today())

        with st.spinner("Forecasting sequentially..."):
            hist_df = load_training_csv()
            forecast_df = _forecast_sequence(
                model=art.model,
                scaler=art.scaler,
                feature_columns=art.feature_columns,
                history_df=hist_df,
                start_date=start_date,
                end_date=end_date,
                forecast_precip=precip,
            )

        with st.expander("Debug (why 0?)", expanded=False):
            st.write(
                {
                    "forecast_start": str(start_date),
                    "forecast_end": str(end_date),
                    "n_days": int(len(forecast_df)),
                    "min_raw_pred": float(forecast_df["raw_pred_pct"].min()),
                    "max_raw_pred": float(forecast_df["raw_pred_pct"].max()),
                    "min_pred_clamped": float(forecast_df["predicted_breakfast_footfall_pct"].min()),
                    "max_pred_clamped": float(forecast_df["predicted_breakfast_footfall_pct"].max()),
                    "rain_zero_days": int((forecast_df["precipitation_mm"] == 0).sum()),
                }
            )
            st.dataframe(forecast_df.head(15), use_container_width=True, hide_index=True)

        display_df = forecast_df.copy()
        if not include_weekends:
            display_df = display_df[display_df["is_weekend"] == 0].copy()

        if total_employees > 0:
            display_df["predicted_people"] = (
                display_df["predicted_breakfast_footfall_pct"] / 100.0 * total_employees
            ).round().astype(int)

        st.subheader("Forecast output")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        todays_row = forecast_df[forecast_df["date"] == Date.today()]
        if len(todays_row) == 1:
            pct = float(todays_row["predicted_breakfast_footfall_pct"].iloc[0])
            st.metric("Today’s predicted footfall %", f"{pct:.2f}%")
            if total_employees > 0:
                st.metric("Today’s predicted people", f"{int(round(pct / 100.0 * total_employees))}")


if __name__ == "__main__":
    main()

