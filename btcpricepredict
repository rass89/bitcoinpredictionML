import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import timedelta

# Load the data
file_path = "btcusd_1-min_data.csv"
data = pd.read_csv(file_path)

# Convert Timestamp to datetime and set as index
data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')
data.set_index('Timestamp', inplace=True)

# Ensure we are using only 'Close' prices for predictions
hourly_data = data['Close'].dropna()
hourly_data = hourly_data.resample('H').mean().dropna()  # Resample to hourly and drop NaNs

# Prepare lagged features for time series modeling
def create_lagged_features(series, n_lags):
    df = pd.DataFrame(series)
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = series.shift(lag)
    return df

# Use n_lags of 3 for better prediction with enough historical data
n_lags = 3
features = create_lagged_features(hourly_data, n_lags)
features.dropna(inplace=True)

# Check if there are enough data points
if len(features) < 2:
    raise ValueError("Not enough data after creating lagged features. Try using smaller n_lags or more data.")

# Split data into training and test sets
X = features.iloc[:, 1:].values  # Lagged features
y = features.iloc[:, 0].values  # Current price (target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# GradientBoostingRegressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# K-Nearest Neighbors Regressor (KNN)
knn_model = KNeighborsRegressor(n_neighbors=min(5, len(X_train)))
knn_model.fit(X_train, y_train)

# Evaluate models
rf_preds = rf_model.predict(X_test)
gb_preds = gb_model.predict(X_test)
knn_preds = knn_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_preds))
knn_rmse = np.sqrt(mean_squared_error(y_test, knn_preds))

print(f"Random Forest RMSE: {rf_rmse:.2f}")
print(f"Gradient Boosting RMSE: {gb_rmse:.2f}")
print(f"KNN RMSE: {knn_rmse:.2f}")

# Forecast the next 30 days dynamically
last_data = hourly_data[-n_lags:].values  # Initialize with last n_lags values
future_rf, future_gb, future_knn = [], [], []

for _ in range(30):
    # Prepare input for models
    rf_input = last_data.reshape(1, n_lags)  # Reshape for models with n_lags=3
    gb_input = rf_input  # Same shape as Random Forest input
    knn_input = rf_input  # Same shape as KNN input

    # Predict the next values
    rf_next = rf_model.predict(rf_input)[0]
    gb_next = gb_model.predict(gb_input)[0]
    knn_next = knn_model.predict(knn_input)[0]

    # Append predictions to respective lists
    future_rf.append(rf_next)
    future_gb.append(gb_next)
    future_knn.append(knn_next)

    # Update last_data to include the latest prediction (dynamic update)
    last_data = np.append(last_data, [rf_next])[-n_lags:]  # Keep last n_lags values

# Create a DataFrame for visualization
future_dates = [hourly_data.index[-1] + timedelta(days=i + 1) for i in range(30)]
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'RandomForest': future_rf,
    'GradientBoosting': future_gb,
    'KNN': future_knn,
})
forecast_df.set_index('Date', inplace=True)

# Print dates vertically with predictions
print("Forecast Results:")
for date, row in forecast_df.iterrows():
    print(f"{date.strftime('%Y-%m-%d')}: RF={row['RandomForest']:.2f}, GB={row['GradientBoosting']:.2f}, KNN={row['KNN']:.2f}")

# Plot results without actual prices and 2024 dates
plt.figure(figsize=(12, 6))
plt.axvline(x=hourly_data.index[-1], color='black', linestyle='--', label='Forecast Start')
plt.plot(forecast_df['RandomForest'], label='Random Forest Prediction', linestyle='--')
plt.plot(forecast_df['GradientBoosting'], label='Gradient Boosting Prediction', linestyle='--')
plt.plot(forecast_df['KNN'], label='KNN Prediction', linestyle='--')

# Adjust x-axis for better readability
plt.xticks(rotation=45, ha="right")  # Rotate dates to avoid overlap
plt.legend()
plt.title('Bitcoin Price Forecast for 30 Days')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid()
plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.show()
