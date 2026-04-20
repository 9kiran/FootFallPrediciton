import streamlit as st
import pandas as pd

def dataFilter(df):
    # 1. Filter to weekdays only for modeling
    df_ml = df[df['is_weekend'] == 0].copy()
    st.write(f"After filtering weekdays: {len(df_ml)} records")

    # 2. Identify categorical features to encode
    categorical_cols = ['day', 'rain_category', 'month']

    # 3. Create one-hot encoding for categorical features
    df_final = pd.get_dummies(df_ml, columns=categorical_cols, drop_first=True, prefix=['day', 'rain', 'mon'])
    st.write(f"After get_dummies: {df_final.shape}")

    # 4. Define columns to drop (data leakage & non-features)
    cols_to_drop = ['date', 'precipitation_mm', 'breakfast_footfall_pct']

    # 5. Prepare X and Y
    X = df_final.drop(columns=[col for col in cols_to_drop if col in df_final.columns])
    Y = df_final['breakfast_footfall_pct']

    st.write(f"Feature matrix shape: {X.shape}")
    st.write(f"Target vector shape: {Y.shape}")
    st.write(f"\nFeatures included: {list(X.columns)}")
    st.write(X)

    return X, Y
