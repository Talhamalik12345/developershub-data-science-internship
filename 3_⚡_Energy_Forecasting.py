"""
Task 3 – Energy Consumption Time Series Forecasting
Objective: Forecast short-term household energy usage.
Dataset: Synthetic hourly energy data (one year).
Models: ARIMA, XGBoost, and Seasonal Decomposition (Prophet-style trend/seasonality).
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Task 3 – Energy Forecasting", page_icon="⚡", layout="wide")

st.title("⚡ Task 3 – Energy Consumption Time Series Forecasting")
st.markdown("""
**Objective:** Forecast short-term household energy consumption using historical patterns.  
**Models compared:** ARIMA · XGBoost · Seasonal Trend Decomposition  
**Metrics:** MAE and RMSE on a 7-day held-out test set.
""")
st.markdown("---")


# ─── Synthetic Data ───────────────────────────────────────────────────────────
@st.cache_data
def generate_energy_data(days=365, seed=42):
    rng = np.random.default_rng(seed)
    freq = "h"
    idx = pd.date_range("2024-01-01", periods=days * 24, freq=freq)

    t = np.arange(len(idx))
    # Yearly seasonality (kWh per hour)
    yearly = 1.2 * np.sin(2 * np.pi * t / (365 * 24))
    # Daily seasonality – peak mornings & evenings
    hour = idx.hour
    daily = (
        0.8 * np.sin(2 * np.pi * (hour - 7) / 24)
        + 0.5 * np.sin(2 * np.pi * (hour - 19) / 24)
    )
    # Weekday effect
    weekday_effect = np.where(idx.dayofweek < 5, 0.3, -0.2)
    # Trend (slight increase over year)
    trend = 0.0003 * t
    # Noise
    noise = rng.normal(0, 0.15, len(t))

    consumption = 2.5 + trend + yearly + daily + weekday_effect + noise
    consumption = np.clip(consumption, 0.1, 8.0)

    return pd.Series(consumption, index=idx, name="consumption_kwh")


series = generate_energy_data()

# ─── EDA ──────────────────────────────────────────────────────────────────────
st.subheader("1. Dataset Overview & EDA")

daily_mean = series.resample("D").mean()
weekly_mean = series.resample("W").mean()

col1, col2, col3 = st.columns(3)
col1.metric("Total Hours", f"{len(series):,}")
col2.metric("Avg Consumption (hourly)", f"{series.mean():.3f} kWh")
col3.metric("Peak Hour Avg", f"{series.max():.2f} kWh")

fig, axes = plt.subplots(3, 1, figsize=(13, 8))

axes[0].plot(weekly_mean.index, weekly_mean.values, color="#e74c3c", lw=1.5)
axes[0].set_title("Weekly Average Energy Consumption (kWh/h)")
axes[0].set_ylabel("kWh/h")
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

hourly_avg = series.groupby(series.index.hour).mean()
axes[1].bar(hourly_avg.index, hourly_avg.values, color="#3498db", edgecolor="white")
axes[1].set_title("Average Consumption by Hour of Day")
axes[1].set_xlabel("Hour")
axes[1].set_ylabel("Avg kWh/h")
axes[1].set_xticks(range(0, 24, 2))

day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
daily_dow = series.groupby(series.index.dayofweek).mean()
axes[2].bar(day_names, daily_dow.values, color="#2ecc71", edgecolor="white")
axes[2].set_title("Average Consumption by Day of Week")
axes[2].set_ylabel("Avg kWh/h")

plt.tight_layout()
st.pyplot(fig)
plt.close()

# ─── Seasonal Decomposition ───────────────────────────────────────────────────
st.subheader("2. Seasonal Decomposition")

daily_series = series.resample("D").mean()
decomp = seasonal_decompose(daily_series, model="additive", period=7)

fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
for ax, data, title, color in zip(
    axes,
    [daily_series, decomp.trend, decomp.seasonal, decomp.resid],
    ["Observed", "Trend", "Seasonality (Weekly)", "Residuals"],
    ["#3498db", "#e74c3c", "#2ecc71", "#95a5a6"]
):
    ax.plot(data, color=color, lw=1.2)
    ax.set_ylabel(title)
    ax.set_title(title)
plt.suptitle("Seasonal Decomposition – Daily Consumption (kWh/day)", y=1.01, fontsize=12)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ─── Train/Test Split ─────────────────────────────────────────────────────────
st.subheader("3. Train / Test Split (7-day forecast horizon)")

# Use daily resampled data for forecasting (more stable)
test_days = 7
train_daily = daily_series.iloc[:-test_days]
test_daily = daily_series.iloc[-test_days:]

st.markdown(f"- **Train:** `{train_daily.index[0].date()}` to `{train_daily.index[-1].date()}` ({len(train_daily)} days)")
st.markdown(f"- **Test:**  `{test_daily.index[0].date()}` to `{test_daily.index[-1].date()}` ({len(test_daily)} days)")


# ─── ARIMA ────────────────────────────────────────────────────────────────────
@st.cache_resource
def fit_arima(train):
    model = ARIMA(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7))
    return model.fit()

st.subheader("4a. ARIMA Model")

with st.spinner("Fitting ARIMA(2,1,2)(1,1,1)[7] …"):
    arima_result = fit_arima(train_daily)

arima_forecast = arima_result.forecast(steps=test_days)
arima_mae = mean_absolute_error(test_daily, arima_forecast)
arima_rmse = np.sqrt(mean_squared_error(test_daily, arima_forecast))

st.markdown(f"**ARIMA MAE:** `{arima_mae:.4f}` kWh  |  **RMSE:** `{arima_rmse:.4f}` kWh")

with st.expander("ARIMA Model Summary"):
    st.text(str(arima_result.summary()))


# ─── XGBoost Feature Engineering ──────────────────────────────────────────────
@st.cache_data
def make_xgb_features(s: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"consumption": s.values}, index=s.index)
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    for lag in [1, 2, 3, 7, 14]:
        df[f"lag_{lag}"] = df["consumption"].shift(lag)
    for w in [7, 14]:
        df[f"rolling_mean_{w}"] = df["consumption"].shift(1).rolling(w).mean()
        df[f"rolling_std_{w}"] = df["consumption"].shift(1).rolling(w).std()
    df.dropna(inplace=True)
    return df

st.subheader("4b. XGBoost Model with Time-Based Features")

xgb_df = make_xgb_features(daily_series)
split_idx = len(xgb_df) - test_days
train_xgb = xgb_df.iloc[:split_idx]
test_xgb = xgb_df.iloc[split_idx:]

feat_cols = [c for c in xgb_df.columns if c != "consumption"]

@st.cache_resource
def fit_xgb(X_tr, y_tr):
    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X_tr, y_tr, verbose=False)
    return model

xgb_model = fit_xgb(train_xgb[feat_cols], train_xgb["consumption"])
xgb_pred = xgb_model.predict(test_xgb[feat_cols])
xgb_mae = mean_absolute_error(test_xgb["consumption"], xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(test_xgb["consumption"], xgb_pred))

st.markdown(f"**XGBoost MAE:** `{xgb_mae:.4f}` kWh  |  **RMSE:** `{xgb_rmse:.4f}` kWh")

# Feature importance
fig, ax = plt.subplots(figsize=(8, 5))
feat_importance = pd.Series(xgb_model.feature_importances_, index=feat_cols).sort_values(ascending=True).tail(10)
feat_importance.plot.barh(ax=ax, color="#f39c12")
ax.set_title("XGBoost – Top-10 Feature Importances")
ax.set_xlabel("Importance Score")
plt.tight_layout()
st.pyplot(fig)
plt.close()


# ─── Seasonal Naive (Prophet-style baseline) ──────────────────────────────────
st.subheader("4c. Seasonal Naïve Baseline (7-day lag, Prophet-style)")
# Use same weekday from prior week as forecast
seasonal_pred = train_daily.values[-test_days:]
seasonal_mae = mean_absolute_error(test_daily.values, seasonal_pred)
seasonal_rmse = np.sqrt(mean_squared_error(test_daily.values, seasonal_pred))
st.markdown(f"**Seasonal Naïve MAE:** `{seasonal_mae:.4f}` kWh  |  **RMSE:** `{seasonal_rmse:.4f}` kWh")
st.caption("A Prophet-style seasonal decomposition baseline using the prior week as the forecast (sNaïve).")


# ─── Comparison & Plots ───────────────────────────────────────────────────────
st.subheader("5. Actual vs Forecasted – All Models")

test_idx = test_daily.index
arima_idx = test_idx[:len(arima_forecast)]

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(test_idx, test_daily.values, "o-", color="black", lw=2, label="Actual", zorder=5)
ax.plot(arima_idx, arima_forecast.values, "s--", color="#e74c3c", lw=1.8, label="ARIMA")
ax.plot(test_idx[:len(xgb_pred)], xgb_pred, "^--", color="#3498db", lw=1.8, label="XGBoost")
ax.plot(test_idx, seasonal_pred, "D--", color="#2ecc71", lw=1.8, label="Seasonal Naïve")
ax.set_title("7-Day Forecast: Actual vs Predicted (Daily kWh/h)")
ax.set_ylabel("Avg Daily Consumption (kWh/h)")
ax.set_xlabel("Date")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ─── Metrics Summary ──────────────────────────────────────────────────────────
st.subheader("6. Model Comparison")

results = pd.DataFrame({
    "Model": ["ARIMA(2,1,2)(1,1,1)[7]", "XGBoost", "Seasonal Naïve"],
    "MAE (kWh/h)": [round(arima_mae, 4), round(xgb_mae, 4), round(seasonal_mae, 4)],
    "RMSE (kWh/h)": [round(arima_rmse, 4), round(xgb_rmse, 4), round(seasonal_rmse, 4)],
})
st.dataframe(results.set_index("Model"), width="stretch")

best = results.loc[results["MAE (kWh/h)"].idxmin(), "Model"]
st.info(f"**Best model by MAE:** {best}")

# ─── Conclusion ───────────────────────────────────────────────────────────────
st.subheader("7. Conclusion & Insights")
st.success("""
- Energy consumption exhibits clear **daily** (morning/evening peaks) and **weekly** (lower weekends) patterns.
- **XGBoost** with engineered lag and rolling features generally produces the lowest forecasting error,
  thanks to its ability to capture non-linear temporal patterns.
- **ARIMA** is strong on stationary components but is sensitive to hyper-parameter tuning.
- **Seasonal Naïve** is a competitive baseline that leverages weekly periodicity.
- Accurate short-term forecasting enables **smart-grid load balancing**, demand-response pricing,
  and household energy-saving recommendations.
""")
