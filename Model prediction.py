import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# File path
file_path = r"MFile address"

# Load the data
data = pd.read_csv(file_path)

# Extract the 'Time (Days)' and the target variables
X = data['Time (Days)'].values.reshape(-1, 1)
y_rice = data['Rice Actual Prices'].values
y_wheat = data['Wheat Actual Prices'].values
y_corn = data['Corn Actual Prices'].values

# Scaling the features
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# Function for XGBoost Model
def xgboost_model(X_train, y_train, X_test):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

# Function for ARIMA Model
def arima_model(y_train, y_test):
    model = sm.tsa.ARIMA(y_train, order=(5, 1, 0))
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(y_test))
    return predictions

# Prepare the data for LSTM and TCN Models
def prepare_lstm_data(X, y):
    X_data = X.reshape((X.shape[0], 1, X.shape[1]))
    return X_data

# TCN Model
def tcn_model(X_train, y_train, X_test):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    predictions = model.predict(X_test)
    return predictions

# LSTM Model
def lstm_model(X_train, y_train, X_test):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    predictions = model.predict(X_test)
    return predictions

# Prepare the data for models
X_train, X_test, y_rice_train, y_rice_test = train_test_split(X_scaled, y_rice, test_size=0.2, random_state=42)
X_train_lstm = prepare_lstm_data(X_train, y_rice_train)
X_test_lstm = prepare_lstm_data(X_test, y_rice_test)

# Train and predict for each product (Rice, Wheat, Corn)
# Rice Predictions
rice_xgb = xgboost_model(X_train, y_rice_train, X_test)
rice_arima = arima_model(y_rice_train, y_rice_test)
rice_tcn = tcn_model(X_train_lstm, y_rice_train, X_test_lstm)
rice_lstm = lstm_model(X_train_lstm, y_rice_train, X_test_lstm)

# Wheat Predictions
wheat_xgb = xgboost_model(X_train, y_wheat_train, X_test)
wheat_arima = arima_model(y_wheat_train, y_wheat_test)
wheat_tcn = tcn_model(X_train_lstm, y_wheat_train, X_test_lstm)
wheat_lstm = lstm_model(X_train_lstm, y_wheat_train, X_test_lstm)

# Corn Predictions
corn_xgb = xgboost_model(X_train, y_corn_train, X_test)
corn_arima = arima_model(y_corn_train, y_corn_test)
corn_tcn = tcn_model(X_train_lstm, y_corn_train, X_test_lstm)
corn_lstm = lstm_model(X_train_lstm, y_corn_train, X_test_lstm)

# Prepare the results to be saved to a CSV file
results = pd.DataFrame({
    'Rice TCN Predictions': rice_tcn.flatten(),
    'Rice XGBoost Predictions': rice_xgb,
    'Rice TCN-XGBoost Predictions': (rice_tcn.flatten() + rice_xgb) / 2,  # Averaging TCN and XGBoost
    'Rice Single-Layer LSTM Predictions': rice_lstm.flatten(),
    'Rice Double-Layer LSTM Predictions': rice_lstm.flatten(),  # Example for double-layer (modify if needed)
    'Rice ARIMA Predictions': rice_arima,

    'Wheat TCN Predictions': wheat_tcn.flatten(),
    'Wheat XGBoost Predictions': wheat_xgb,
    'Wheat TCN-XGBoost Predictions': (wheat_tcn.flatten() + wheat_xgb) / 2,  # Averaging TCN and XGBoost
    'Wheat Single-Layer LSTM Predictions': wheat_lstm.flatten(),
    'Wheat Double-Layer LSTM Predictions': wheat_lstm.flatten(),  # Example for double-layer (modify if needed)
    'Wheat ARIMA Predictions': wheat_arima,

    'Corn TCN Predictions': corn_tcn.flatten(),
    'Corn XGBoost Predictions': corn_xgb,
    'Corn TCN-XGBoost Predictions': (corn_tcn.flatten() + corn_xgb) / 2,  # Averaging TCN and XGBoost
    'Corn Single-Layer LSTM Predictions': corn_lstm.flatten(),
    'Corn Double-Layer LSTM Predictions': corn_lstm.flatten(),  # Example for double-layer (modify if needed)
    'Corn ARIMA Predictions': corn_arima,
})

# Save the predictions to CSV
output_file_path = r"Final Prediction data"
results.to_csv(output_file_path, index=False)

# Show the first few rows of the result dataframe
print(results.head())
