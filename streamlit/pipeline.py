from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from utils import (
    categorize_rain,
    fetch_precipitation_data,
    fetch_precipitation_forecast,
    pune_holidays_2024,
    pune_holidays_2025,
)


@dataclass(frozen=True)
class DatasetBundle:
    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    feature_columns: List[str]


@dataclass(frozen=True)
class ModelArtifacts:
    model_name: str
    model: object
    scaler: StandardScaler
    feature_columns: List[str]
    meta: Dict[str, object]


def get_repo_paths() -> Tuple[Path, Path, Path]:
    """
    Returns:
        streamlit_dir: .../ML_BKF_Pred/streamlit
        project_dir:   .../ML_BKF_Pred
        artifacts_dir: .../ML_BKF_Pred/artifacts
    """
    streamlit_dir = Path(__file__).resolve().parent
    project_dir = streamlit_dir.parent
    artifacts_dir = project_dir / "artifacts"
    return streamlit_dir, project_dir, artifacts_dir


def load_training_csv() -> pd.DataFrame:
    _, project_dir, _ = get_repo_paths()
    csv_path = project_dir / "canteen_breakfast_data.csv"
    df = pd.read_csv(csv_path)
    if "date" not in df.columns or "breakfast_footfall_pct" not in df.columns:
        raise ValueError("CSV must contain columns: date, breakfast_footfall_pct")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _holiday_month_day_set() -> set[tuple[int, int]]:
    """
    Temporary holiday handling.

    You will provide the real holiday list later. For now we approximate holidays for other
    years by reusing the (month, day) patterns observed in the last 2 years holiday lists.
    """
    s = set()
    for d in pd.to_datetime(pune_holidays_2024 + pune_holidays_2025):
        s.add((int(d.month), int(d.day)))
    return s


def _is_holiday(dt: Date) -> int:
    md = (int(dt.month), int(dt.day))
    return int(md in _holiday_month_day_set())


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["day"] = df["date"].dt.day_name()
    df["month"] = df["date"].dt.month_name()
    df["is_weekend"] = (df["date"].dt.weekday >= 5).astype(int)
    df["is_holiday"] = df["date"].dt.date.apply(_is_holiday).astype(int)

    df["is_long_weekend"] = 0
    for i in range(len(df)):
        dow = df.loc[i, "day"]
        hol = df.loc[i, "is_holiday"]
        if dow == "Thursday" and hol == 1 and i + 1 < len(df):
            df.loc[i + 1, "is_long_weekend"] = 1
        elif dow == "Tuesday" and hol == 1 and i - 1 >= 0:
            df.loc[i - 1, "is_long_weekend"] = 1

    return df


def add_rain_features_historic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    start_date = df["date"].min().date().isoformat()
    end_date = df["date"].max().date().isoformat()
    rain = fetch_precipitation_data(start_date=start_date, end_date=end_date)

    if rain is None or getattr(rain, "empty", True):
        df["precipitation_mm"] = np.nan
    else:
        rain = rain.copy()
        rain["date"] = pd.to_datetime(rain["date"])
        df = df.merge(rain[["date", "precipitation_mm"]], on="date", how="left")

    df["rain_category"] = df["precipitation_mm"].apply(categorize_rain)
    df.loc[df["rain_category"].isin(["Unknown"]), "rain_category"] = "No Rain"
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses the *full* series (including weekend zeros) to match the provided CSV behavior.
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["rolling_7"] = (
        df["breakfast_footfall_pct"].shift(1).rolling(window=7).mean().bfill()
    )
    df["lag_7"] = df["breakfast_footfall_pct"].shift(7).bfill()
    return df


def make_feature_matrix(df: pd.DataFrame) -> DatasetBundle:
    df = df.copy()
    # Train only on weekdays (matches existing implementation)
    df_ml = df[df["is_weekend"] == 0].copy()

    categorical_cols = ["day", "rain_category", "month"]
    df_final = pd.get_dummies(
        df_ml, columns=categorical_cols, drop_first=True, prefix=["day", "rain", "mon"]
    )

    cols_to_drop = ["date", "precipitation_mm", "breakfast_footfall_pct"]
    X = df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns])
    y = df_final["breakfast_footfall_pct"]

    return DatasetBundle(df=df, X=X, y=y, feature_columns=list(X.columns))


def time_split(X: pd.DataFrame, y: pd.Series, train_frac: float = 0.8):
    split_idx = int(len(X) * train_frac)
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def evaluate_model(model: object, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def candidate_models() -> Dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "KNN (k=7)": KNeighborsRegressor(n_neighbors=7),
        "Random Forest (n=200)": RandomForestRegressor(n_estimators=200, random_state=42),
    }


def select_best(results: pd.DataFrame) -> str:
    """
    Pick the best model based on:
    - Highest R2
    - Then lowest RMSE
    - Then lowest MAE
    """
    ranked = results.sort_values(["R2", "RMSE", "MAE"], ascending=[False, True, True])
    return str(ranked.iloc[0]["model"])


def train_and_score_models(bundle: DatasetBundle, train_frac: float = 0.8) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = time_split(bundle.X, bundle.y, train_frac=train_frac)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rows = []
    for name, model in candidate_models().items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        m = evaluate_model(model, y_test.to_numpy(), pred)
        rows.append({"model": name, **m})

    return pd.DataFrame(rows)


def train_best_on_all_data(bundle: DatasetBundle, best_model_name: str) -> ModelArtifacts:
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(bundle.X)

    model = candidate_models()[best_model_name]
    model.fit(X_all_scaled, bundle.y)

    meta: Dict[str, object] = {
        "trained_on": "all_weekday_rows",
        "n_rows": int(len(bundle.X)),
        "date_min": str(bundle.df["date"].min().date()),
        "date_max": str(bundle.df["date"].max().date()),
    }

    return ModelArtifacts(
        model_name=best_model_name,
        model=model,
        scaler=scaler,
        feature_columns=bundle.feature_columns,
        meta=meta,
    )


def save_artifacts(art: ModelArtifacts) -> Path:
    if joblib is None:
        raise RuntimeError("joblib is required to save model artifacts")

    _, _, artifacts_dir = get_repo_paths()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "footfall_model.joblib"
    joblib.dump(
        {
            "model_name": art.model_name,
            "model": art.model,
            "scaler": art.scaler,
            "feature_columns": art.feature_columns,
            "meta": art.meta,
        },
        out_path,
    )
    return out_path


def load_artifacts(path: Optional[Path] = None) -> ModelArtifacts:
    if joblib is None:
        raise RuntimeError("joblib is required to load model artifacts")

    _, _, artifacts_dir = get_repo_paths()
    p = path or (artifacts_dir / "footfall_model.joblib")
    payload = joblib.load(p)
    return ModelArtifacts(
        model_name=str(payload["model_name"]),
        model=payload["model"],
        scaler=payload["scaler"],
        feature_columns=list(payload["feature_columns"]),
        meta=dict(payload.get("meta", {})),
    )


def build_feature_row_for_date(
    feature_columns: List[str],
    target_date: Date,
    precipitation_mm: float,
    rolling_7: float,
    lag_7: float,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    ts = pd.Timestamp(target_date)

    day_name = ts.day_name()
    month_name = ts.month_name()
    is_weekend = int(ts.weekday() >= 5)
    is_holiday = _is_holiday(target_date)

    # Keep the same long-weekend heuristic, but in a date-local form
    is_long_weekend = 0
    if day_name == "Friday":
        prev = (pd.Timestamp(target_date) - pd.Timedelta(days=1)).date()
        is_long_weekend = _is_holiday(prev)  # type: ignore[arg-type]
    elif day_name == "Monday":
        nxt = (pd.Timestamp(target_date) + pd.Timedelta(days=1)).date()
        is_long_weekend = _is_holiday(nxt)  # type: ignore[arg-type]

    rain_cat = categorize_rain(precipitation_mm)
    if rain_cat in ["Unknown", None]:
        rain_cat = "No Rain"

    base = pd.DataFrame(
        [
            {
                "date": ts,
                "breakfast_footfall_pct": np.nan,
                "precipitation_mm": precipitation_mm,
                "rain_category": rain_cat,
                "day": day_name,
                "month": month_name,
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
                "is_long_weekend": is_long_weekend,
                "rolling_7": float(rolling_7),
                "lag_7": float(lag_7),
            }
        ]
    )

    df_dum = pd.get_dummies(
        base,
        columns=["day", "rain_category", "month"],
        drop_first=True,
        prefix=["day", "rain", "mon"],
    )
    cols_to_drop = ["date", "precipitation_mm", "breakfast_footfall_pct"]
    X_one = df_dum.drop(columns=[c for c in cols_to_drop if c in df_dum.columns])
    X_one = X_one.reindex(columns=feature_columns, fill_value=0)

    debug = {
        "day": day_name,
        "month": month_name,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "is_long_weekend": is_long_weekend,
        "precipitation_mm": precipitation_mm,
        "rain_category": rain_cat,
        "rolling_7": float(rolling_7),
        "lag_7": float(lag_7),
    }
    return X_one, debug


def _get_archive_precip_for_dates(dates: List[Date]) -> Dict[Date, float]:
    if not dates:
        return {}
    start = min(dates).isoformat()
    end = max(dates).isoformat()
    df = fetch_precipitation_data(start_date=start, end_date=end)
    if df is None or getattr(df, "empty", True):
        return {d: 0.0 for d in dates}
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    mapping = {d: float(mm) for d, mm in zip(df["date"], df["precipitation_mm"])}
    return {d: float(mapping.get(d, 0.0)) for d in dates}


def _get_forecast_precip_for_dates(dates: List[Date]) -> Dict[Date, float]:
    """
    Open-Meteo forecast endpoint only supports a limited horizon.
    We therefore only call it for short, future-facing windows.
    """
    if not dates:
        return {}
    start = min(dates).isoformat()
    end = max(dates).isoformat()
    df = fetch_precipitation_forecast(start_date=start, end_date=end)
    if df is None or getattr(df, "empty", True):
        return {d: 0.0 for d in dates}
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    mapping = {d: float(mm) for d, mm in zip(df["date"], df["precipitation_mm"])}
    return {d: float(mapping.get(d, 0.0)) for d in dates}


def get_precip_for_dates(dates: List[Date], today: Optional[Date] = None) -> Dict[Date, float]:
    """
    Returns precipitation_mm for each requested date.

    Strategy:
    - **Past dates**: use Open-Meteo archive API (actuals)
    - **Today and future**: use Open-Meteo forecast API (prediction)
    """
    if not dates:
        return {}

    ref = today or pd.Timestamp.today().date()
    past = [d for d in dates if d < ref]
    future = [d for d in dates if d >= ref]

    out: Dict[Date, float] = {}
    out.update(_get_archive_precip_for_dates(past))

    # Forecast endpoint has limited horizon; we only call it for the requested future dates range.
    out.update(_get_forecast_precip_for_dates(future))

    # Ensure every date has a value
    for d in dates:
        out.setdefault(d, 0.0)
    return out

