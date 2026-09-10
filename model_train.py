import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression


# --------------------------------------------------
# File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "train_processed.csv"
MODEL_FILE = BASE_DIR / "AQI_model.pkl"


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
# Read processed training dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Training dataset shape:", df.shape)


# --------------------------------------------------
# Separate inputs (X) and target (y)
# --------------------------------------------------

X_train = df[FEATURES]
y_train = df[TARGET]


# --------------------------------------------------
# Create Linear Regression model
# --------------------------------------------------

model = LinearRegression()


# --------------------------------------------------
# Train the model
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# Save trained model
# --------------------------------------------------

joblib.dump(model, MODEL_FILE)


# --------------------------------------------------
# Display model information
# --------------------------------------------------

print("\nModel trained successfully.")

print("Training samples:", len(X_train))

print("Features used:")
for feature in FEATURES:
    print("-", feature)

print("\nModel coefficients:")
for feature, coefficient in zip(FEATURES, model.coef_):
    print(f"{feature}: {coefficient}")

print("\nIntercept:", model.intercept_)

print("\nSaved as:", MODEL_FILE)