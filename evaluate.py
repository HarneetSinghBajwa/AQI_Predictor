import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
# Same split used during model training
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
# Load trained model
# --------------------------------------------------

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")


# --------------------------------------------------
# Predict AQI using test data
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# Calculate evaluation metrics
# --------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


# --------------------------------------------------
# Display model performance
# --------------------------------------------------

print("\nModel Performance")
print("-----------------")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))


# --------------------------------------------------
# Evaluation completed
# --------------------------------------------------

print("\nEvaluation completed successfully.")