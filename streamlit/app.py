from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from dataEnrichment import dataEnrichment
from dataFilter import dataFilter
from dataSplitAndScale import dataSplitAndScale
from ModelOptimization import evaluate_and_plot_variations

st.set_page_config(layout="wide")

# Set random seed for reproducibility
np.random.seed(42)

st.title("Breakfast Footfall Prediction Dashboard", text_alignment="center")
st.header("Data Prepartion for Modeling", divider=True)

with st.expander("Loading and enriching data..."):
    df = pd.read_csv('..\canteen_breakfast_data.csv')
    st.write(f"Initial data loaded with {len(df)} records and columns: {list(df.columns)}")
    st.write(df.head())
    df = dataEnrichment(df)

with st.expander("Preparing data for modeling..."):
    X,Y = dataFilter(df)

with st.expander("Data splitting and scaling..."):
    X_train_scaled, X_test_scaled, y_train, y_test = dataSplitAndScale(X, Y)


st.header("Data Modeling", divider=True)

tabs = st.tabs(["Linear Regression", "KNN", "Random Forest"])

with tabs[0]:
    st.write("Training and Evaluating Linear Regression model...")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    y_pred_lr = lr.predict(X_test_scaled)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)

    # df_lr_results = pd.DataFrame({
    #     #'Matrix': ['MAE', 'R2 Score', 'RMSE'],
    #     'Value': [mae_lr, r2_lr, rmse_lr],
    #     'Significance': ['Lower is better', 'Higher is better', 'Lower is better']
    # }, index=['MAE', 'R2 Score', 'RMSE'])  

    # st.write(df_lr_results)
    st.write(f"MAE: {mae_lr:.2f}%")
    st.write(f"R2 Score: {r2_lr:.4f}")
    st.write(f"RMSE: {rmse_lr:.2f}%")

    residuals = y_test - y_pred_lr
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("Residual Analysis")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_pred_lr, residuals, alpha=0.5)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Predicted')
        st.pyplot(fig)
        st.write("This plot shows whether errors are randomly distributed around zero. In our breakfast footfall model, a random scatter indicates proper capture of trends from date, holiday, and weather features.")

    with col2:
        st.write("Actual vs Predicted")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.scatter(y_test, y_pred_lr, alpha=0.5)
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.set_title('Actual vs Predicted')
        st.pyplot(fig2)
        st.write("Points near the diagonal line mean accurate predictions; deviations reveal systematic bias. For this dataset, off-diagonal clusters would suggest further feature engineering (e.g., long weekend or rain categories) is needed.")

    with col3:
        st.write("Residual Distribution")
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.hist(residuals, bins=20, color='tab:blue', alpha=0.7)
        ax3.axvline(residuals.mean(), color='red', linestyle='--', label=f'Mean={residuals.mean():.2f}')
        ax3.set_xlabel('Residual')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Residual Histogram')
        ax3.legend()
        st.pyplot(fig3)
        st.write("A central, narrow distribution means low error variance. Wide skewed residuals mean the model may miss peak or low footfall patterns (e.g., holidays or extreme weather days).")

with tabs[1]:
    st.write("Training and evaluation KNN model...")
    # Example usage for KNN and Random Forest
    res = evaluate_and_plot_variations(
        KNeighborsRegressor,
        {
            'n_neighbors': range(1, 31)
        },
        'KNN',
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        figsize=(12, 4),
        top_n=3
    )
    st.write(res)
    st.write(f"\nTop KNN params (first row): {res.loc[0, 'params']}")

with tabs[2]:
    st.write("Training and evaluation Random Forest model...")
    res = evaluate_and_plot_variations(
        RandomForestRegressor,
        {
            'n_estimators': range(1, 25)
        },
        'RandomForest',
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        figsize=(12, 4),
        top_n=3
    )
    st.write(res)
    st.write(f"\nTop RandomForest params (first row): {res.loc[0, 'params']}")




# # Model Training
# lr = LinearRegression()
# lr.fit(X_train_scaled, y_train)

# # EVALUATION 
# y_pred = lr.predict(X_test_scaled)
# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"MAE: {mae:.2f}%")
# print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
# print(f"RMSE: {rmse:.2f}%")



# # Generic model variation + plot helper
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import ParameterGrid


# def evaluate_and_plot_variations(model_class, param_grid, model_name, X_train, y_train, X_test, y_test, figsize=(10,5), top_n=5):
#     """Train multiple variations of a model and plot metrics.

#     Args:
#         model_class: estimator class (e.g., KNeighborsRegressor)
#         param_grid: dict of hyperparameters for ParameterGrid
#         model_name: str label for chart titles
#         X_train, y_train, X_test, y_test: data arrays
#         top_n: number of best configs to print

#     Returns:
#         DataFrame of results sorted by R2 desc
#     """
#     results = []
#     for params in ParameterGrid(param_grid):
#         model = model_class(**params)
#         model.random_state = 42  # ensure reproducibility if applicable
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         r2 = r2_score(y_test, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         mae = mean_absolute_error(y_test, y_pred)

#         results.append({
#             'model': model_name,
#             'params': params,
#             'R2': r2,
#             'RMSE': rmse,
#             'MAE': mae
#         })

#     results_df = pd.DataFrame(results).sort_values('R2', ascending=False).reset_index(drop=True)

#     # display top configs
#     print(f"\nTop {top_n} {model_name} configs by R²:")
#     for i in range(min(top_n, len(results_df))):
#         row = results_df.loc[i]
#         print(f"{i+1}. {row['params']} -> R2={row['R2']:.4f}, RMSE={row['RMSE']:.2f}, MAE={row['MAE']:.2f}")

#     # plot metric evolution
#     df_plot = results_df.copy()
#     df_plot['idx'] = df_plot.index + 1

#     fig, ax = plt.subplots(1, 3, figsize=(figsize[0], figsize[1]))
#     ax[0].plot(df_plot['idx'], df_plot['R2'], marker='o')
#     ax[0].set_title(f'{model_name} R² Trend')
#     ax[0].set_xlabel(model_name)
#     ax[0].set_ylabel('R²')

#     ax[1].plot(df_plot['idx'], df_plot['RMSE'], marker='o', color='orange')
#     ax[1].set_title(f'{model_name} RMSE Trend')
#     ax[1].set_xlabel(model_name)
#     ax[1].set_ylabel('RMSE')

#     ax[2].plot(df_plot['idx'], df_plot['MAE'], marker='o', color='green')
#     ax[2].set_title(f'{model_name} MAE Trend')
#     ax[2].set_xlabel(model_name)
#     ax[2].set_ylabel('MAE')

#     plt.tight_layout()
#     plt.show()

#     return results_df


# # Example usage for KNN and Random Forest
# evaluate_and_plot_variations(
#     KNeighborsRegressor,
#     {
#         'n_neighbors': range(1, 31)
#     },
#     'KNN',
#     X_train_scaled,
#     y_train,
#     X_test_scaled,
#     y_test,
#     figsize=(12, 4),
#     top_n=3
# )

# rf_results_df = evaluate_and_plot_variations(
#     RandomForestRegressor,
#     {
#         'n_estimators': range(1, 250,3)
#     },
#     'RandomForest',
#     X_train_scaled,
#     y_train,
#     X_test_scaled,
#     y_test,
#     figsize=(12, 4),
#     top_n=3
# )

# print('\nTop RandomForest params (first row):', rf_results_df.loc[0, 'params'])