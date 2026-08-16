import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# --------------------------------------------------
# File names
# --------------------------------------------------

INPUT_FILE = "clean_data.csv"
MODEL_FILE = "AQI_model.pkl"


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
# Read cleaned dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# Separate inputs (X) and target (y)
# --------------------------------------------------

X = df[FEATURES]
y = df[TARGET]


# --------------------------------------------------
# Split data
# 80% training
# 20% testing
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


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

print("\nModel trained successfully.")
print("Saved as:", MODEL_FILE)