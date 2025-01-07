import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from numpy import sqrt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor

# Utility function to calculate metrics
def calculate_metrics(test_data, predictions):
    rmse = sqrt(mean_squared_error(test_data, predictions))
    mae = mean_absolute_error(test_data, predictions)
    mape = (abs((test_data - predictions) / test_data).mean()) * 100
    r2 = r2_score(test_data, predictions)
    explained_var = explained_variance_score(test_data, predictions)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "Explained Variance": explained_var}

# SARIMA forecasting function
def sarima_forecast(data, seasonal_period=12, test_size=0.2):
    train_size = int(len(data) * (1 - test_size))
    train_data = data[:train_size]
    test_data = data[train_size:]

    auto_arima_model = auto_arima(train_data, seasonal=True, m=seasonal_period, trace=False, error_action='ignore',
                                  suppress_warnings=True)
    arima_model = ARIMA(train_data, order=auto_arima_model.order, seasonal_order=auto_arima_model.seasonal_order)
    arima_result = arima_model.fit()

    start_date = test_data.index[0]
    end_date = test_data.index[-1]
    predictions = arima_result.predict(start=start_date, end=end_date)

    metrics = calculate_metrics(test_data, predictions)
    return metrics, predictions, test_data

# XGBoost forecasting function
def xgboost_forecast(data, lag_features=8, test_size=0.2):
    def get_lag(data, col, lagtime):
        for i in range(1, lagtime + 1):
            data[f"{col}_lag{i}"] = data[col].shift(i)
        return data

    data = data.to_frame(name="target")
    data = get_lag(data, "target", lag_features)
    data.dropna(inplace=True)

    split_date = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_date]
    test_data = data.iloc[split_date:]

    X_train, y_train = train_data.drop("target", axis=1), train_data["target"]
    X_test, y_test = test_data.drop("target", axis=1), test_data["target"]

    reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50, verbosity=0)
    reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

    predictions = reg.predict(X_test)

    metrics = calculate_metrics(y_test, predictions)
    return metrics, predictions, y_test

# Gradient Boosting forecasting function
def gradient_boosting_forecast(data, lag_features=8, test_size=0.2):
    def get_lag(data, col, lagtime):
        for i in range(1, lagtime + 1):
            data[f"{col}_lag{i}"] = data[col].shift(i)
        return data

    data = data.to_frame(name="target")
    data = get_lag(data, "target", lag_features)
    data.dropna(inplace=True)

    split_date = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_date]
    test_data = data.iloc[split_date:]

    X_train, y_train = train_data.drop("target", axis=1), train_data["target"]
    X_test, y_test = test_data.drop("target", axis=1), test_data["target"]

    gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=3, random_state=42)
    gbr.fit(X_train, y_train)

    predictions = gbr.predict(X_test)

    metrics = calculate_metrics(y_test, predictions)
    return metrics, predictions, y_test

# Streamlit UI
st.title("Energy Consumption Forecast")
st.sidebar.title("Model Selection")

@st.cache_data
def load_data(uploaded_file):
    consumption_df = pd.read_csv(uploaded_file, low_memory=False)

    consumption_df.columns = list(map(str.lower, consumption_df.columns))
    consumption_df['timestamp'] = pd.to_datetime(consumption_df['date'] + consumption_df['time'],
                                                 format='%d/%m/%Y%H:%M:%S', errors='coerce')
    consumption_df.drop(['date', 'time'], axis=1, inplace=True)

    float_cols = ['global_active_power', 'global_reactive_power', 'voltage',
                  'global_intensity', 'sub_metering_1', 'sub_metering_2',
                  'sub_metering_3']
    for col in float_cols:
        consumption_df[col] = pd.to_numeric(consumption_df[col], errors='coerce')

    consumption_df.set_index('timestamp', inplace=True)

    daily_consumption_df = consumption_df.resample(rule='D').mean()
    daily_consumption_df.ffill(inplace=True)
    monthly_consumption_df = daily_consumption_df.resample(rule='ME').mean()

    return monthly_consumption_df['global_active_power']

# Dataset analysis function
def analyze_dataset(data):
    st.subheader("Dataset Analysis")
    st.write("**Basic Statistics**")
    st.write(data.describe())

    st.write("**Missing Values**")
    st.write(data.isnull().sum())

    st.write("**Data Sample**")
    st.write(data.head())

    st.write("**Plotting Distribution**")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=30, color='skyblue', edgecolor='black')
    ax.set_title("Distribution of Global Active Power")
    ax.set_xlabel("Global Active Power")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# File upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV format)", type=["csv"])
if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("Dataset loaded successfully!")
    st.line_chart(data, width=800, height=300)

    # Analyze the dataset
    analyze_dataset(data)

    # Model selection
    model_choice = st.sidebar.selectbox("Choose a model:", ("SARIMA", "XGBoost", "Gradient Boosting"))
    seasonal_period = st.sidebar.slider("Seasonal Period (for SARIMA)", 1, 24, 12) if model_choice == "SARIMA" else None
    test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100
    train_button = st.sidebar.button("Train Model")

    if train_button:
        st.write(f"Training {model_choice} model...")
        if model_choice == "SARIMA":
            metrics, predictions, test_data = sarima_forecast(data, seasonal_period=seasonal_period, test_size=test_size)
        elif model_choice == "XGBoost":
            metrics, predictions, test_data = xgboost_forecast(data, test_size=test_size)
        elif model_choice == "Gradient Boosting":
            metrics, predictions, test_data = gradient_boosting_forecast(data, test_size=test_size)

        # Display metrics
        st.write("**Evaluation Metrics:**")
        metrics_df = pd.DataFrame(metrics, index=[0]).T.reset_index()
        metrics_df.columns = ["Metric", "Value"]
        st.write(metrics_df)

        # Static plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data, label='Actual', marker='o')
        ax.plot(test_data.index, predictions, label='Predicted (Test)', color='red', marker='o')
        ax.axvline(x=test_data.index[0], color='gray', linestyle='--', label='Train-Test Split')
        ax.set_title("Energy Consumption Forecast")
        ax.legend()
        st.pyplot(fig)
# Add forecast 10% future function
def forecast_future(data, model_choice, future_steps=0, seasonal_period=12):
    if model_choice == "SARIMA":
        auto_arima_model = auto_arima(data, seasonal=True, m=seasonal_period, trace=False, error_action='ignore',
                                      suppress_warnings=True)
        arima_model = ARIMA(data, order=auto_arima_model.order, seasonal_order=auto_arima_model.seasonal_order)
        arima_result = arima_model.fit()

        future_index = pd.date_range(start=data.index[-1], periods=future_steps + 1, freq='M')[1:]
        future_forecast = arima_result.get_forecast(steps=future_steps).predicted_mean

        return pd.Series(future_forecast, index=future_index)
    elif model_choice in ["XGBoost", "Gradient Boosting"]:
        # Prepare lagged features
        lag_features = 8
        future_data = data.to_frame(name="target")
        for i in range(1, lag_features + 1):
            future_data[f"target_lag{i}"] = future_data["target"].shift(i)
        future_data.dropna(inplace=True)

        X_train = future_data.drop("target", axis=1)
        y_train = future_data["target"]

        if model_choice == "XGBoost":
            model = xgb.XGBRegressor(n_estimators=1000, verbosity=0)
        elif model_choice == "Gradient Boosting":
            model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=3, random_state=42)

        model.fit(X_train, y_train)

        # Generate future predictions
        predictions = []
        input_data = X_train.iloc[-1].values
        for _ in range(future_steps):
            next_pred = model.predict([input_data])[0]
            predictions.append(next_pred)
            input_data = list(input_data[1:]) + [next_pred]  # Update input for next step

        future_index = pd.date_range(start=data.index[-1], periods=future_steps + 1, freq='M')[1:]
        return pd.Series(predictions, index=future_index)

# Add "Forecast 10% Future" option
forecast_future_button = st.sidebar.checkbox("Forecast 10% Future")
if uploaded_file is not None and forecast_future_button:
    future_steps = int(len(data) * 0.2)
    st.write(f"Forecasting {future_steps} future steps...")
    future_predictions = forecast_future(data, model_choice, future_steps, seasonal_period)

    # Display future predictions
    st.write("Future Predictions:")
    st.write(future_predictions)

    # Plot future predictions
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data, label='Actual Data', marker='o')
    ax.plot(future_predictions.index, future_predictions, label='Future Predictions', color='green', marker='o')
    ax.set_title("Energy Consumption Forecast with Future Prediction")
    ax.legend()
    st.pyplot(fig)