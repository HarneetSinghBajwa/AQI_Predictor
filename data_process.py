import pandas as pd
from pathlib import Path


# --------------------------------------------------
# File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "train_data.csv"
OUTPUT_FILE = BASE_DIR / "train_processed.csv"


# --------------------------------------------------
# Features
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


# --------------------------------------------------
# Read training dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Training dataset shape:", df.shape)


# --------------------------------------------------
# Median Imputation
# --------------------------------------------------
# Median is calculated ONLY from the training dataset.

for column in FEATURES:

    median_value = df[column].median()

    print(f"{column} median: {median_value}")

    df[column] = df[column].fillna(median_value)


# --------------------------------------------------
# Save processed training dataset
# --------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)


# --------------------------------------------------
# Verify missing values
# --------------------------------------------------

print("\nMissing values after median imputation:")

print(df[FEATURES].isnull().sum())


# --------------------------------------------------
# Final information
# --------------------------------------------------

print("\nProcessed training dataset shape:", df.shape)

print("\nTraining data preprocessing completed successfully.")

print("Saved as:", OUTPUT_FILE)