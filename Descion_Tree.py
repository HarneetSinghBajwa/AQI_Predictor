import pandas as pd
import joblib
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor

# --------------------------------------------------
# File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "train_processed.csv"
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

df = pd.read_csv(INPUT_FILE)

X = df[FEATURES]
y = df[TARGET]

print("Training dataset shape:", df.shape)


# --------------------------------------------------
# Train Decision Tree
# --------------------------------------------------

model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

model.fit(X, y)


# --------------------------------------------------
# Save model
# --------------------------------------------------

joblib.dump(model, MODEL_FILE)

print("\nDecision Tree trained successfully.")
print("Tree depth:", model.get_depth())
print("Number of leaves:", model.get_n_leaves())
print("Saved as:", MODEL_FILE)