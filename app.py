import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, time
from meteostat import Point, Daily
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from pathlib import Path
from sklearn.base import RegressorMixin
import copy
import numpy as np

GLOBAL_RANDOM_STATE = 42
MODELS = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0, random_state=GLOBAL_RANDOM_STATE),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=GLOBAL_RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=GLOBAL_RANDOM_STATE),
}

CITY_MAP = {
    # (Latitude, Longitude, Elevation, Location Name used in CSV/Meteostat)
    "New York": (40.71, -74.01, 10, "New York City"),
    "London": (51.5, -0.12, 25, "London, UK"),
    "Tokyo": (35.68, 139.75, 40, "Tokyo, Japan"),
    "Miami": (25.76, -80.19, 2, "Miami, USA"),
    "Delhi": (28.61, 77.21, 216, "Delhi, India"),
    "Berlin": (52.52, 13.4, 34, "Berlin, Germany"),
    "Paris": (48.86, 2.35, 35, "Paris, France")
}
DEFAULT_CITY = "New York"
DAYS_TO_ANALYZE = 365
END_DATE = datetime.now().date()
START_DATE = END_DATE - timedelta(days=DAYS_TO_ANALYZE)
# CRITICAL: File path assumption for Streamlit Cloud deployment
CSV_FILE_PATH = Path('historical_music_popularity.csv')

# Features used by the model
FEATURES = [
    'popularity_lag_1', 'energy', 'valence', 'tempo', 'danceability',
    'tavg', 'prcp', 'daylight_hours',
    'month_sin', 'month_cos',
    'is_weekend'
]

# CSV DATA LOADING 
@st.cache_data(show_spinner="1. Loading Historical Music Data from CSV...")
def load_constant_music_data(file_path):

    if not file_path.exists():
        st.error(f"FATAL ERROR: The required data file **{file_path.name}** was not found at {file_path.resolve()}.")
        st.stop()

    try:
        music_df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"FATAL ERROR: Could not read CSV file. Reason: {e}")
        st.stop()

    music_df['date'] = pd.to_datetime(music_df['date']).dt.normalize()
    required_cols = ['date', 'track_id', 'location', 'popularity', 'energy', 'valence', 'tempo', 'danceability']

    if not all(col in music_df.columns for col in required_cols):
         st.error(f"FATAL ERROR: CSV must contain the following columns: {required_cols}")
         st.stop()

    return music_df

#  DATA ACQUISITION & FEATURE ENGINEERING
@st.cache_data(show_spinner="2. Acquiring Weather and Merging Data...")
def get_integrated_data(lat, lon, elevation, location_name, start_date, end_date):

    def get_meteostat_weather_data(lat, lon, start, end, location_name, elevation):
        start_dt = datetime.combine(start, time(0, 0))
        end_dt = datetime.combine(end, time(0, 0))

        location = Point(lat, lon, elevation)
        data = Daily(location, start_dt, end_dt)
        weather_df = data.fetch().reset_index().rename(columns={'time': 'date'})
        weather_df['location'] = location_name

        weather_df = weather_df[['date', 'location', 'tavg', 'prcp', 'tsun']].copy()
        weather_df['daylight_hours'] = weather_df['tsun'].fillna(0) / 60
        weather_df.drop(columns=['tsun'], inplace=True)
        return weather_df.dropna(subset=['tavg'])

    # Get Music Data
    music_df_all = load_constant_music_data(CSV_FILE_PATH)

    # Filter for Location and Date Range
    music_df = music_df_all[
        (music_df_all['location'] == location_name) &
        (music_df_all['date'].dt.date >= start_date) &
        (music_df_all['date'].dt.date <= end_date)
    ].copy()

    if music_df.empty:
         st.warning(f"No music data found for {location_name} in the CSV for the selected dates.")
         st.stop()

    effective_start_date = music_df['date'].min().date()
    effective_end_date = music_df['date'].max().date()

    # Get Meteostat Weather Data
    weather_df = get_meteostat_weather_data(lat, lon, effective_start_date, effective_end_date, location_name, elevation)

    # Merge and Feature Engineering
    weather_df['date'] = pd.to_datetime(weather_df['date']).dt.normalize()
    master_df = pd.merge(music_df, weather_df, on=['date', 'location'], how='inner')

    if master_df.empty:
         st.warning("No overlapping data found between music data and weather data. Check dates and location names.")
         st.stop()

    # Time/Cyclical Features
    master_df['month'] = master_df['date'].dt.month
    master_df['day_of_week'] = master_df['date'].dt.dayofweek
    master_df['is_weekend'] = master_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    master_df['month_sin'] = np.sin(2 * np.pi * master_df['month'] / 12)
    master_df['month_cos'] = np.cos(2 * np.pi * master_df['month'] / 12)

    # Lagged Popularity
    master_df = master_df.sort_values(by=['track_id', 'date']).reset_index(drop=True)
    master_df['popularity_lag_1'] = master_df.groupby('track_id')['popularity'].shift(1)

    # Final cleanup of features
    master_df.dropna(subset=['popularity_lag_1'] + [col for col in FEATURES if col not in ['popularity_lag_1']], inplace=True)

    return master_df


# DYNAMIC MODEL TRAINING AND SELECTION
@st.cache_resource(show_spinner="3. Training and Selecting the Best Regression Model...")
def train_and_select_models(data_df):
    X = data_df[FEATURES]
    y = data_df['popularity']

    # Chronological Split (80% Train, 20% Test)
    split_point = int(len(X) * 0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    trained_models = {}
    best_model_name = None
    best_r2 = -np.inf

    # Initialize a base validation DF to merge predictions into
    validation_df_full = pd.DataFrame({
        'date': data_df['date'].dt.date.tail(len(y_test)),
        'actual_popularity': y_test
    }).reset_index(drop=True)

    for name, model_instance in MODELS.items():
        # Use deepcopy to ensure model instances are independent
        model_clone = copy.deepcopy(model_instance)

        # Train and Evaluate
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        trained_models[name] = model_clone

        # Merge prediction into the full validation DF
        validation_df_full[f'predicted_popularity_{name}'] = y_pred

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

    # R2 scores dict for displaying the table
    r2_scores = {name: r2_score(y_test, validation_df_full[f'predicted_popularity_{name}']) for name in MODELS.keys()}

    return trained_models[best_model_name], validation_df_full, best_model_name, r2_scores


# FORECASTING (Uses the dynamically selected best mode)
@st.cache_data(show_spinner="4. Generating 30-Day Forecast...")
def generate_forecast(_model, data_df, days=30):
    last_date = data_df['date'].max()
    future_dates = [last_date.date() + timedelta(days=d) for d in range(1, days + 1)]
    forecast_df = pd.DataFrame({'date': future_dates})

    seasonal_stats = data_df[['tavg', 'prcp', 'daylight_hours']].mean()

    # Simulate Future Features (Weather forecast - Use GLOBAL_RANDOM_STATE for reproducibility)
    np.random.seed(GLOBAL_RANDOM_STATE)
    forecast_df['tavg'] = seasonal_stats['tavg'] + np.random.normal(0, 3, days)
    forecast_df['prcp'] = seasonal_stats['prcp'] * np.random.uniform(0.5, 1.5, days)
    forecast_df['daylight_hours'] = seasonal_stats['daylight_hours'] + np.random.normal(0, 1, days)
    np.random.seed() # Reset seed

    # Feature Engineering for Forecast
    forecast_df['month'] = forecast_df['date'].apply(lambda x: x.month)
    forecast_df['is_weekend'] = forecast_df['date'].apply(lambda x: 1 if x.weekday() >= 5 else 0)
    forecast_df['month_sin'] = np.sin(2 * np.pi * forecast_df['month'] / 12)
    forecast_df['month_cos'] = np.cos(2 * np.pi * forecast_df['month'] / 12)

    # Lagged Popularity (Using the mean of the training set)
    last_lag_value = data_df['popularity_lag_1'].mean()
    forecast_df['popularity_lag_1'] = last_lag_value

    # Song Attributes (Average values from the training set)
    for feature in ['energy', 'valence', 'tempo', 'danceability']:
        forecast_df[feature] = data_df[feature].mean()

    # Make Forecast
    X_forecast = forecast_df[FEATURES]
    forecast_predictions = _model.predict(X_forecast)

    forecast_df['predicted_popularity'] = np.round(forecast_predictions).astype(int)
    return forecast_df[['date', 'predicted_popularity', 'tavg', 'daylight_hours', 'month']]


# ==============================================================================
# STREAMLIT APPLICATION UI & VISUALIZATION
# ==============================================================================

# Page Configuration
st.set_page_config(layout="wide", page_title="Song Trend Forecasting 🎶")
st.title("🎶 Dynamic Weather-Driven Song Trend Forecasting Dashboard")

# --- UI INPUT (Sidebar) ---
st.sidebar.header("Location & Analysis Settings")

selected_city = st.sidebar.selectbox(
    "Select a City to Analyze:",
    options=list(CITY_MAP.keys()),
    index=list(CITY_MAP.keys()).index(DEFAULT_CITY)
)

lat, lon, elevation, location_name = CITY_MAP[selected_city]

st.sidebar.markdown(f"""
    ---
    **Data Source:** **{CSV_FILE_PATH.name}** 💾 (Local File)
    **Date Range:** 1 Year ({START_DATE} to {END_DATE})
""")

# --- Execution Flow ---
master_df = get_integrated_data(lat, lon, elevation, location_name, START_DATE, END_DATE)
trained_best_model, validation_df_full, best_model_name, r2_scores = train_and_select_models(master_df)
forecast_df = generate_forecast(trained_best_model, master_df)

best_r2_val = r2_scores[best_model_name]

# Dashboard Header 
st.markdown(f"### Location: {location_name} | Best Model Selected: **{best_model_name}** | Best $R^2$ Score: **{best_r2_val:.4f}**")
st.dataframe(pd.Series(r2_scores).to_frame(name='R² Score').rename_axis('Model').sort_values(by='R² Score', ascending=False), use_container_width=True)
st.divider()

# ----------------------------------------------------
## Historical Model Validation (Actual vs. All Predicted)
# ----------------------------------------------------
st.header("1. Historical Model Validation (Actual vs. All Predicted)")
st.info("The model with the highest $R^2$ score is highlighted (solid line) and used for the final forecast.")

fig_validation = go.Figure()
fig_validation.add_trace(go.Scatter(x=validation_df_full['date'], y=validation_df_full['actual_popularity'],
                                     mode='lines', name='Actual Popularity (CSV Data)', line=dict(color='red', width=4)))

# Plot all predictions, highlighting the best one
colors = {
    'Random Forest Regressor': 'blue',
    'Gradient Boosting': 'green',
    'Linear Regression': 'orange',
    'Ridge Regression': 'purple'
}

for name, color in colors.items():
    if f'predicted_popularity_{name}' in validation_df_full.columns:
        line_style = dict(color=color, dash='solid' if name == best_model_name else 'dot', width=3 if name == best_model_name else 1.5)
        fig_validation.add_trace(go.Scatter(x=validation_df_full['date'], y=validation_df_full[f'predicted_popularity_{name}'],
                                             mode='lines', name=f'{name} Prediction', line=line_style, opacity=1 if name == best_model_name else 0.5))

fig_validation.update_layout(xaxis_title='Date', yaxis_title='Popularity Score', hovermode="x unified", height=450)
st.plotly_chart(fig_validation, use_container_width=True)

# ----------------------------------------------------
##  30-Day Forward Trend Forecast (Using Best Model)
# ----------------------------------------------------
st.header(f"2. 30-Day Forward Trend Forecast (Using **{best_model_name}**)")
st.caption(f"Forecast generated using the selected best model and simulated future weather.")

col1, col2 = st.columns(2)

with col1:
    fig_forecast_pop = go.Figure()
    fig_forecast_pop.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['predicted_popularity'],
                                             mode='lines+markers', name='Predicted Popularity', line=dict(color='red', width=3)))
    fig_forecast_pop.update_layout(title='Predicted Song Popularity Trend', xaxis_title='Date', yaxis_title='Popularity Score', hovermode="x unified", height=400)
    st.plotly_chart(fig_forecast_pop, use_container_width=True)

with col2:
    fig_weather = go.Figure()
    fig_weather.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['tavg'],
                                     mode='lines', name='Avg. Temp (°C)', yaxis='y1', line=dict(color='orange', width=2)))

    fig_weather.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['daylight_hours'],
                                     mode='lines', name='Daylight Hours', yaxis='y2', line=dict(color='skyblue', dash='dot', width=2)))

    fig_weather.update_layout(title='Forecasted Weather Drivers (Simulated)', xaxis_title='Date',
        yaxis=dict(title='Avg. Temp (°C)', color='orange'),
        yaxis2=dict(title='Daylight Hours', overlaying='y', side='right', color='skyblue'),
        hovermode="x unified", height=400)
    st.plotly_chart(fig_weather, use_container_width=True)

# ----------------------------------------------------
##  Seasonal Trends of Key Features
# ----------------------------------------------------
st.header("3. Seasonal Trends of Key Features (Based on Historical Data)")
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

seasonal_df = master_df.groupby(master_df['date'].dt.month)[['valence', 'energy', 'tavg']].mean().reset_index()
seasonal_df['Month'] = seasonal_df['date'].apply(lambda x: months[x - 1])

fig_seasonal = px.line(seasonal_df, x='Month', y=['valence', 'energy'],
                        title=f'Average Song Valence & Energy vs. Temperature in {selected_city}',
                        labels={'value': 'Score (0-1)', 'Month': 'Month'},
                        color_discrete_map={'valence': 'green', 'energy': 'blue'})

fig_seasonal.add_trace(go.Scatter(x=seasonal_df['Month'], y=seasonal_df['tavg'],
                                     name='Avg. Temp (°C)', yaxis='y2', mode='lines',
                                     line=dict(color='red', dash='dash')))

fig_seasonal.update_layout(
    yaxis=dict(title='Score (0-1)'),
    yaxis2=dict(title='Avg. Temp (°C)', overlaying='y', side='right', showgrid=False),
    hovermode="x unified"
)
st.plotly_chart(fig_seasonal, use_container_width=True)
