import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
df = pd.read_csv("new_synthetic_data_with_time.csv")

# Sort by timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# Select only sensor values
data = df[["methane", "air_quality", "temperature", "humidity"]]

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
import joblib

joblib.dump(scaler, "lstm_scaler.pkl")

# Create sequences
sequence_length = 10

X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

print(X.shape)
print(y.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Build model
model = Sequential([
    LSTM(64, input_shape=(sequence_length, 4)),
    Dense(32, activation="relu"),
    Dense(4)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

# Train
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

# Save
model.save("lstm_model.keras")