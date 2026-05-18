import streamlit as st
import pandas as pd
import numpy as np

from core.utils import fetch_precipitation_data, categorize_rain, pune_holidays_2024, pune_holidays_2025, pune_holidays_2026

def _noop(*args, **kwargs):
    return None

def data_enrichment(df, verbose: bool = True):
    write = st.write if verbose else _noop
    bar_chart = st.bar_chart if verbose else _noop
    
    # ensure df has datetime column named date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        raise KeyError("Dataframe must contain a 'date' column for enrichment")
    
    data_start = df['date'].min().date().isoformat()
    data_end = df['date'].max().date().isoformat()
    write(f"Fetching Precipitation data from Open-Meteo API ({data_start} \u2192 {data_end})...")
    df_rain = fetch_precipitation_data(start_date=data_start, end_date=data_end)

    if df_rain is not None:
        # merge into main df by matching date (both lowercase)
        df = df.merge(df_rain.rename(columns={'date': 'date'}), on='date', how='left')
        write(f"Merged precipitation data, resulting dataframe has {len(df)} records")
        write(df[['date','precipitation_mm']].head(10))
    else:
        # create placeholder column so later code doesn't break
        df['precipitation_mm'] = np.nan
        write("No precipitation data available; column filled with NaN.")


    # Apply rain categorization and prepare for feature engineering
    df['rain_category'] = df['precipitation_mm'].apply(categorize_rain)

    if verbose:
        raindf = df[df['breakfast_footfall_pct'] > 0].groupby('rain_category')['breakfast_footfall_pct'].mean()
        logical_order = ['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain']
        raindf = raindf.reindex(logical_order)
        write(raindf)
        bar_chart(raindf, sort=False, height=300, width=500, x_label="Rain Category", y_label="Average Breakfast Footfall %")
        # bar_chart(raindf, height=300, width=500, x_label="Rain Category", y_label="Average Breakfast Footfall %")
        write("Rain categories created:")
        write(df['rain_category'].value_counts())
        write(f"Dataframe shape before feature engineering: {df.shape}")
    
    df = df.sort_values('date').reset_index(drop=True)

    all_holidays = pd.to_datetime(pune_holidays_2024 + pune_holidays_2025 + pune_holidays_2026)

    # create flag
    if 'date' in df.columns:
        df['is_holiday'] = df['date'].isin(all_holidays).astype(int)
    else:
        df['is_holiday'] = 0  # fallback in case date column missing

    write(f"Holidays marked in dataframe: {df['is_holiday'].sum()} days")

    # Identify long weekends and add flag to dataframe
    # Case 1: Holiday on Thursday (people take leave on Friday)
    # Case 2: Holiday on Tuesday (people take leave on Monday)

    # ensure date is datetime and compute day-of-week for later flags
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['day'] = df['date'].dt.day_name()
        # optional weekend flag if not already present
        if 'is_weekend' not in df.columns:
            df['is_weekend'] = df['date'].dt.weekday >= 5
            df['is_weekend'] = df['is_weekend'].astype(int)
    else:
        raise KeyError("Dataframe does not contain a 'date' column")

    # ensure columns are lowercase
    for col in ['day', 'is_holiday']:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in df")

    # Adding month column as string for categorical encoding
    df['month'] = df['date'].dt.month_name()

    # initialize flag
    df['is_long_weekend'] = 0

    for i in range(len(df)):
        dow = df.loc[i, 'day']
        hol = df.loc[i, 'is_holiday']
        # Thursday holiday -> mark next day
        if dow == 'Thursday' and hol == 1:
            if i + 1 < len(df):
                df.loc[i + 1, 'is_long_weekend'] = 1
        # Tuesday holiday -> mark previous day
        elif dow == 'Tuesday' and hol == 1:
            if i - 1 >= 0:
                df.loc[i - 1, 'is_long_weekend'] = 1

    if verbose:
        # Summary statistics for long weekends
        long_weekend_mask = df['is_long_weekend'] == 1
        count_long_weekend = long_weekend_mask.sum()
        mean_footfall_long_weekend = df.loc[long_weekend_mask, 'breakfast_footfall_pct'].mean()
        write(f"Long-weekend days flagged: {count_long_weekend}")
        write(f"Mean footfall during long weekends: {mean_footfall_long_weekend:.2f}%")

        tempdf = df.copy()
        tempdf['is_long_weekend_label'] = df['is_long_weekend'].map({0: 'No Long Weekend', 1: 'Long Weekend'})
        tempdf.set_index('is_long_weekend_label', inplace=True)

        write(tempdf[tempdf['breakfast_footfall_pct'] > 0].groupby('is_long_weekend_label')['breakfast_footfall_pct'].mean())
        bar_chart(tempdf[tempdf['breakfast_footfall_pct'] > 0].groupby('is_long_weekend_label')['breakfast_footfall_pct'].mean(), height=300, width=500, y_label="Average Breakfast Footfall %")

        # Mean footfall by day of week
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_mean = (
            df[df['breakfast_footfall_pct'] > 0]
            .groupby('day')['breakfast_footfall_pct']
            .mean()
            .reindex(weekday_order)
        )
        write("Average breakfast footfall by day of week:")
        write(weekday_mean)
        bar_chart(weekday_mean, sort=False, height=300, width=500, x_label="Day of Week", y_label="Average Breakfast Footfall %")


    df["rolling_7"] = (
        df["breakfast_footfall_pct"]
        .shift(1)  # shift to avoid data leakage
        .rolling(window=7)
        .mean()
        .fillna(method="bfill")  # backfill for the first 7 days
    )

    # Last 7 day data
    df["lag_7"] = df["breakfast_footfall_pct"].shift(7).fillna(method="bfill")

    return df

