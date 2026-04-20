from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# Reuse project utilities (holidays + rain categorization + Open-Meteo archive fetch)
# NOTE: This is *your* local module at `ML_BKF_Pred/streamlit/utils.py` (not `streamlit.utils`).
import sys

_STREAMLIT_UTILS_DIR = str((Path(__file__).resolve().parent / "streamlit").as_posix())
if _STREAMLIT_UTILS_DIR not in sys.path:
    sys.path.insert(0, _STREAMLIT_UTILS_DIR)

from utils import categorize_rain, fetch_precipitation_data, pune_holidays_2024, pune_holidays_2025  # type: ignore


@dataclass(frozen=True)
class TrainedBundle:
    feature_columns: List[str]
    scaler: StandardScaler
    models: Dict[str, object]
    last_7day_mean: float
    last_7day_lag: float


def _project_paths() -> Tuple[Path, Path]:
    here = Path(__file__).resolve().parent
    csv_path = here / "canteen_breakfast_data.csv"
    return here, csv_path


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    all_holidays = pd.to_datetime(pune_holidays_2024 + pune_holidays_2025)
    df["day"] = df["date"].dt.day_name()
    df["month"] = df["date"].dt.month_name()
    df["is_weekend"] = (df["date"].dt.weekday >= 5).astype(int)
    df["is_holiday"] = df["date"].isin(all_holidays).astype(int)

    # Long-weekend heuristic used in the existing enrichment module
    df["is_long_weekend"] = 0
    for i in range(len(df)):
        dow = df.loc[i, "day"]
        hol = df.loc[i, "is_holiday"]
        if dow == "Thursday" and hol == 1 and i + 1 < len(df):
            df.loc[i + 1, "is_long_weekend"] = 1
        elif dow == "Tuesday" and hol == 1 and i - 1 >= 0:
            df.loc[i - 1, "is_long_weekend"] = 1

    return df


def _add_rain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Archive API is historical; keep it bounded to the data range
    start_date = df["date"].min().date().isoformat()
    end_date = df["date"].max().date().isoformat()
    rain = fetch_precipitation_data(start_date=start_date, end_date=end_date)
    if rain is None or rain.empty:
        df["precipitation_mm"] = np.nan
    else:
        rain = rain.copy()
        rain["date"] = pd.to_datetime(rain["date"])
        df = df.merge(rain[["date", "precipitation_mm"]], on="date", how="left")

    df["rain_category"] = df["precipitation_mm"].apply(categorize_rain)
    # In case of Unknown, treat as No Rain for stable one-hot encoding
    df.loc[df["rain_category"].isin(["Unknown"]), "rain_category"] = "No Rain"
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Match the original notebook logic: include weekends (0% entries exist in CSV)
    df["rolling_7"] = (
        df["breakfast_footfall_pct"].shift(1).rolling(window=7).mean().bfill()
    )
    df["lag_7"] = df["breakfast_footfall_pct"].shift(7).bfill()
    return df


def _make_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df_ml = df[df["is_weekend"] == 0].copy()

    categorical_cols = ["day", "rain_category", "month"]
    df_final = pd.get_dummies(
        df_ml, columns=categorical_cols, drop_first=True, prefix=["day", "rain", "mon"]
    )

    cols_to_drop = ["date", "precipitation_mm", "breakfast_footfall_pct"]
    X = df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns])
    y = df_final["breakfast_footfall_pct"]
    return X, y


@st.cache_resource(show_spinner=False)
def _train_models() -> TrainedBundle:
    _, csv_path = _project_paths()
    df = pd.read_csv(csv_path)

    df = _add_calendar_features(df)
    df = _add_rain_features(df)
    df = _add_lag_features(df)

    X, y = _make_feature_matrix(df)

    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models: Dict[str, object] = {
        "Linear Regression": LinearRegression(),
        "KNN (k=7)": KNeighborsRegressor(n_neighbors=7),
        "Random Forest (n=200)": RandomForestRegressor(
            n_estimators=200, random_state=42
        ),
    }
    for m in models.values():
        m.fit(X_train_scaled, y_train)

    # These are used to build a reasonable “today” feature row without needing future labels.
    last_7day_mean = float(df["breakfast_footfall_pct"].tail(7).mean())
    last_7day_lag = float(df["breakfast_footfall_pct"].iloc[-7])

    return TrainedBundle(
        feature_columns=list(X.columns),
        scaler=scaler,
        models=models,
        last_7day_mean=last_7day_mean,
        last_7day_lag=last_7day_lag,
    )


def _build_single_row_features(bundle: TrainedBundle, target_date: Date) -> pd.DataFrame:
    # Rain for the target day (historical archive). If unavailable, assume 0mm.
    try:
        rain = fetch_precipitation_data(
            start_date=target_date.isoformat(), end_date=target_date.isoformat()
        )
        precipitation_mm = float(rain["precipitation_mm"].iloc[0]) if rain is not None and len(rain) else 0.0
    except Exception:
        precipitation_mm = 0.0

    all_holidays = set(pd.to_datetime(pune_holidays_2024 + pune_holidays_2025).date)

    dt = pd.Timestamp(target_date)
    day_name = dt.day_name()
    month_name = dt.month_name()
    is_weekend = int(dt.weekday() >= 5)
    is_holiday = int(target_date in all_holidays)

    # Long-weekend heuristic based on the requested date only
    is_long_weekend = 0
    if day_name == "Friday":
        is_long_weekend = int((target_date.replace(day=target_date.day - 1) in all_holidays) if target_date.day > 1 else 0)
    elif day_name == "Monday":
        is_long_weekend = int((target_date.replace(day=target_date.day + 1) in all_holidays) if target_date.day < 28 else 0)

    base = pd.DataFrame(
        [
            {
                "date": dt,
                "breakfast_footfall_pct": np.nan,  # placeholder, not used as a feature
                "precipitation_mm": precipitation_mm,
                "rain_category": categorize_rain(precipitation_mm) or "No Rain",
                "day": day_name,
                "month": month_name,
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
                "is_long_weekend": is_long_weekend,
                "rolling_7": bundle.last_7day_mean,
                "lag_7": bundle.last_7day_lag,
            }
        ]
    )

    base.loc[base["rain_category"].isin(["Unknown"]), "rain_category"] = "No Rain"

    # Apply same one-hot as training, then align columns
    df_dum = pd.get_dummies(
        base, columns=["day", "rain_category", "month"], drop_first=True, prefix=["day", "rain", "mon"]
    )
    cols_to_drop = ["date", "precipitation_mm", "breakfast_footfall_pct"]
    X_one = df_dum.drop(columns=[c for c in cols_to_drop if c in df_dum.columns])
    X_one = X_one.reindex(columns=bundle.feature_columns, fill_value=0)
    return X_one


def main() -> None:
    st.set_page_config(page_title="Footfall Prediction", layout="wide")
    st.title("Breakfast Footfall Predictor")

    bundle = _train_models()

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            target_date = st.date_input("Today’s date", value=Date.today())
        with c2:
            total_employees = st.number_input(
                "Total employees",
                min_value=0,
                value=1000,
                step=10,
            )
        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    X_one = _build_single_row_features(bundle, target_date)
    X_one_scaled = bundle.scaler.transform(X_one)

    preds_pct: Dict[str, float] = {}
    for name, model in bundle.models.items():
        pred = float(model.predict(X_one_scaled)[0])
        preds_pct[name] = max(0.0, min(100.0, pred))

    avg_pct = float(np.mean(list(preds_pct.values()))) if preds_pct else 0.0
    avg_pct = max(0.0, min(100.0, avg_pct))

    st.subheader("Predictions")
    cols = st.columns(4)
    i = 0
    for name, pct in preds_pct.items():
        cols[i].metric(label=name, value=f"{pct:.2f}%")
        i += 1
    cols[3].metric(label="Average (3-model)", value=f"{avg_pct:.2f}%")

    st.subheader("Estimated breakfast headcount")
    if total_employees > 0:
        rows = []
        for name, pct in preds_pct.items():
            rows.append(
                {
                    "model": name,
                    "predicted_footfall_pct": pct,
                    "predicted_people": int(round((pct / 100.0) * total_employees)),
                }
            )
        rows.append(
            {
                "model": "Average (3-model)",
                "predicted_footfall_pct": avg_pct,
                "predicted_people": int(round((avg_pct / 100.0) * total_employees)),
            }
        )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Enter total employees > 0 to see predicted people count.")


if __name__ == "__main__":
    main()
