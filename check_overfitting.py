import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "train_processed.csv"
MODEL_FILE = BASE_DIR / "AQI_decision_tree.pkl"


# --------------------------------------------------
# Features and target
# --------------------------------------------------

FEATURES = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3",
    "Temperature",
    "Humidity"
]

TARGET = "AQI"


# --------------------------------------------------
# Load training data
# --------------------------------------------------

df = pd.read_csv(TRAIN_FILE)

X_train = df[FEATURES]
y_train = df[TARGET]


# --------------------------------------------------
# Load EXISTING trained model
# --------------------------------------------------

model = joblib.load(MODEL_FILE)


# --------------------------------------------------
# Generate training predictions
# --------------------------------------------------

y_train_pred = model.predict(X_train)


# --------------------------------------------------
# Calculate training metrics
# --------------------------------------------------

mae = mean_absolute_error(y_train, y_train_pred)

rmse = mean_squared_error(
    y_train,
    y_train_pred
) ** 0.5

r2 = r2_score(
    y_train,
    y_train_pred
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n========================================")
print("       TRAINING PERFORMANCE")
print("========================================")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

print("========================================")

print("\nExisting model was NOT retrained.")
print("Existing model was NOT modified.")