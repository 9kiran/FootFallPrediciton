import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from matplotlib import pyplot as plt
import streamlit as st

def get_model_param_number(model_class, param):
    if model_class.__name__ == 'KNeighborsRegressor':
        return param['n_neighbors']
    elif model_class.__name__ == 'RandomForestRegressor':
        return param['n_estimators']
    return -1  # default for unknown models


def evaluate_and_plot_variations(model_class, param_grid, model_name, X_train, y_train, X_test, y_test, figsize=(10,5), top_n=5):
    """Train multiple variations of a model and plot metrics.

    Args:
        model_class: estimator class (e.g., KNeighborsRegressor)
        param_grid: dict of hyperparameters for ParameterGrid
        model_name: str label for chart titles
        X_train, y_train, X_test, y_test: data arrays
        top_n: number of best configs to print

    Returns:
        DataFrame of results sorted by R2 desc
    """
    results = []
    for params in ParameterGrid(param_grid):
        model = model_class(**params)
        model.random_state = 42  # ensure reproducibility if applicable
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        results.append({
            'param_index': get_model_param_number(model_class, params),
            'model': model_name,
            'params': params,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae
        })

    results_df = pd.DataFrame(results).sort_values('R2', ascending=False).reset_index(drop=True)

    # display top configs
    st.write(f"\nTop {top_n} {model_name} configs by R²:")
    for i in range(min(top_n, len(results_df))):
        row = results_df.loc[i]
        st.write(f"{i+1}. {row['params']} -> R2={row['R2']:.4f}, RMSE={row['RMSE']:.2f}, MAE={row['MAE']:.2f}")

    # plot metric evolution
    df_plot = results_df.copy()
    df_plot['idx'] = df_plot.index + 1

    fig, ax = plt.subplots(1, 3, figsize=(figsize[0], figsize[1]))
    ax[0].plot(df_plot['idx'], df_plot['R2'], marker='o')
    ax[0].set_title(f'{model_name} R² Trend')
    ax[0].set_xlabel(model_name)
    ax[0].set_ylabel('R²')

    ax[1].plot(df_plot['idx'], df_plot['RMSE'], marker='o', color='orange')
    ax[1].set_title(f'{model_name} RMSE Trend')
    ax[1].set_xlabel(model_name)
    ax[1].set_ylabel('RMSE')

    ax[2].plot(df_plot['idx'], df_plot['MAE'], marker='o', color='green')
    ax[2].set_title(f'{model_name} MAE Trend')
    ax[2].set_xlabel(model_name)
    ax[2].set_ylabel('MAE')

    st.pyplot(fig)

    return results_df
