# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="India Milk Demand & Supply Forecast")

# --- Helper functions ---
def project_growth(last_value, years, rate, method="compound"):
    years_arr = np.arange(1, years+1)
    if method == "compound":
        return last_value * ((1 + rate) ** years_arr)
    else:
        return last_value + last_value * rate * years_arr

def forecast_linear(y_hist, exog_hist, exog_future):
    model = LinearRegression()
    model.fit(exog_hist, y_hist)
    preds = model.predict(exog_future)
    return preds, model

def forecast_sarimax(y_hist, exog_hist, exog_future, order=(1,1,0)):
    model = sm.tsa.SARIMAX(y_hist, exog=exog_hist, order=order,
                           enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    preds = res.get_forecast(steps=len(exog_future), exog=exog_future).predicted_mean
    return preds, res

def forecast_expsmooth(y_hist, steps):
    model = ExponentialSmoothing(y_hist, trend="add", seasonal=None)
    res = model.fit(optimized=True)
    preds = res.forecast(steps)
    return preds, res

# --- UI ---
st.title("Milk Demand and Production Forecast — India (2010–2047)")

uploaded = st.file_uploader("Upload CSV...", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    # Load a sample CSV from your project folder
   df = pd.read_csv("data/india_milk_data.csv")



df = df.sort_values("year").reset_index(drop=True)


df = df.sort_values("year").reset_index(drop=True)

hist_end = 2025
forecast_years = np.arange(2026, 2048)
n_forecast = len(forecast_years)

# Sidebar controls
st.sidebar.header("Growth parameters")
growth_method = st.sidebar.selectbox("Growth method", ["compound", "linear"])
pop_rate = st.sidebar.slider("Population growth %", -1.0, 5.0, 0.8, 0.1) / 100
cons_rate = st.sidebar.slider("Consumption per capita growth %", -5.0, 10.0, 0.5, 0.1) / 100
animal_rate = st.sidebar.slider("Milk animal growth %", -2.0, 5.0, 0.5, 0.1) / 100
yield_rate = st.sidebar.slider("Milk yield growth %", -1.0, 5.0, 1.0, 0.1) / 100

st.sidebar.header("Forecast models")
demand_model_choice = st.sidebar.selectbox("Demand model", ["Linear Regression", "SARIMAX", "Exponential Smoothing"])
production_model_choice = st.sidebar.selectbox("Production model", ["Linear Regression", "SARIMAX", "Exponential Smoothing"])

# Historical data
df_hist = df[df["year"] <= hist_end].set_index("year")

last_pop = df_hist["population"].iloc[-1]
last_cons = df_hist["consumptionpercapita"].iloc[-1]
last_animal = df_hist["milchanimal"].iloc[-1]
last_yield = df_hist["milkyield"].iloc[-1]

# Project explanatory variables
pop_proj = project_growth(last_pop, n_forecast, pop_rate, method=growth_method)
cons_proj = project_growth(last_cons, n_forecast, cons_rate, method=growth_method)
animal_proj = project_growth(last_animal, n_forecast, animal_rate, method=growth_method)
yield_proj = project_growth(last_yield, n_forecast, yield_rate, method=growth_method)

exog_demand_hist = df_hist[["population","consumptionpercapita"]].values
exog_demand_future = np.column_stack([pop_proj, cons_proj])

exog_prod_hist = df_hist[["milchanimal","milkyield"]].values
exog_prod_future = np.column_stack([animal_proj, yield_proj])

# Forecast demand
y_demand_hist = df_hist["milkdemand"].values
if demand_model_choice == "Linear Regression":
    demand_preds, _ = forecast_linear(y_demand_hist, exog_demand_hist, exog_demand_future)
elif demand_model_choice == "SARIMAX":
    demand_preds, _ = forecast_sarimax(y_demand_hist, exog_demand_hist, exog_demand_future)
else:
    demand_preds, _ = forecast_expsmooth(y_demand_hist, n_forecast)

# Forecast production
y_prod_hist = df_hist["milkproduction"].values
if production_model_choice == "Linear Regression":
    prod_preds, _ = forecast_linear(y_prod_hist, exog_prod_hist, exog_prod_future)
elif production_model_choice == "SARIMAX":
    prod_preds, _ = forecast_sarimax(y_prod_hist, exog_prod_hist, exog_prod_future)
else:
    prod_preds, _ = forecast_expsmooth(y_prod_hist, n_forecast)

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_hist.index, y=y_demand_hist, mode="lines+markers", name="Demand (actual)", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=forecast_years, y=demand_preds, mode="lines+markers", name="Demand (forecast)", line=dict(color="blue", dash="dash")))
fig.add_trace(go.Scatter(x=df_hist.index, y=y_prod_hist, mode="lines+markers", name="Production (actual)", line=dict(color="green")))
fig.add_trace(go.Scatter(x=forecast_years, y=prod_preds, mode="lines+markers", name="Production (forecast)", line=dict(color="green", dash="dash")))

fig.update_layout(title="Milk Demand and Production Forecast (2010–2047)", xaxis_title="Year", yaxis_title="Milk Million Ton", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True)

# Show projections
proj_df = pd.DataFrame({
    "year": forecast_years,
    "population-million": pop_proj,
    "consumptionpercapita-Gms/day": cons_proj,
    "milchanimal-Million heads": animal_proj,
    "milkyield-kg/year": yield_proj
})
st.subheader("Projected explanatory variables (2026–2047)")
st.dataframe(proj_df.style.format("{:.2f}"))
