
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load data
df = pd.read_csv('seattle-weather.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Feature Engineering
df['day_of_year'] = df.index.dayofyear
df['temp_diff'] = df['tempmax'] - df['tempmin']
df['rolling_mean_temp'] = df['tempavg'].rolling(window=7).mean()
df['rain_category'] = (df['precipitation'] > 0).astype(int)
df.fillna(method='bfill', inplace=True)

# Select features
features = ['tempavg', 'precipitation', 'wind', 'temp_diff', 'rolling_mean_temp', 'day_of_year', 'rain_category']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

# Create time series sequences
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]
        y = data[i+seq_len, 0]  # Predicting tempavg
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LEN = 30
X, y = create_sequences(scaled_data, SEQ_LEN)

# Train/Test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Predict and evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Test MSE: {mse:.4f}")

# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title("Temperature Prediction (Normalized)")
plt.xlabel("Time Step")
plt.ylabel("Temperature (scaled)")
plt.show()
