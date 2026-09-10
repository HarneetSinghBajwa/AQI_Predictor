import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "train_data.csv"
TEST_FILE = BASE_DIR / "test_data.csv"
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
# Read training and testing datasets
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("Training dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)


# --------------------------------------------------
# Calculate medians from training data ONLY
# --------------------------------------------------

training_medians = {}

print("\nTraining-set median values:")

for column in FEATURES:

    median_value = train_df[column].median()

    training_medians[column] = median_value

    print(f"{column}: {median_value}")


# --------------------------------------------------
# Prepare test data
# --------------------------------------------------

X_test = test_df[FEATURES].copy()
y_test = test_df[TARGET]


# --------------------------------------------------
# Apply training medians to test data
# --------------------------------------------------

for column in FEATURES:

    X_test[column] = X_test[column].fillna(
        training_medians[column]
    )


# --------------------------------------------------
# Load trained Decision Tree model
# --------------------------------------------------

model = joblib.load(MODEL_FILE)


# --------------------------------------------------
# Generate predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# Calculate evaluation metrics
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(y_test, y_pred)


# --------------------------------------------------
# Display evaluation results
# --------------------------------------------------

print("\n========================================")
print("       DECISION TREE EVALUATION")
print("========================================")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

print("========================================")

print("\nEvaluation completed successfully.")