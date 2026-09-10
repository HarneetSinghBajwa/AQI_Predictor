import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "globalAQI_raw.csv"

TRAIN_FILE = BASE_DIR / "train_data.csv"
TEST_FILE = BASE_DIR / "test_data.csv"


# --------------------------------------------------
# Required parameters
# --------------------------------------------------

FEATURES = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
    "temperature",
    "humidity"
]

TARGET = "aqi"

COLUMNS = FEATURES + [TARGET]


# --------------------------------------------------
# Read raw dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)


# --------------------------------------------------
# Check required columns
# --------------------------------------------------

missing_columns = [
    column for column in COLUMNS
    if column not in df.columns
]

if missing_columns:
    print("ERROR: Required columns are missing:")
    print(missing_columns)

    print("\nAvailable columns:")
    print(list(df.columns))

    raise SystemExit


# --------------------------------------------------
# Keep only required parameters
# --------------------------------------------------

df = df[COLUMNS].copy()


# --------------------------------------------------
# Rename columns
# --------------------------------------------------

df = df.rename(columns={
    "pm25": "PM2.5",
    "pm10": "PM10",
    "no2": "NO2",
    "so2": "SO2",
    "co": "CO",
    "o3": "O3",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "aqi": "AQI"
})


# --------------------------------------------------
# Convert values to numeric
# Invalid values become missing
# --------------------------------------------------

for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# --------------------------------------------------
# Remove invalid negative pollution values
# --------------------------------------------------

POLLUTANTS = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3"
]

for column in POLLUTANTS:
    df.loc[df[column] < 0, column] = pd.NA


# --------------------------------------------------
# Humidity must be between 0 and 100
# --------------------------------------------------

df.loc[df["Humidity"] < 0, "Humidity"] = pd.NA
df.loc[df["Humidity"] > 100, "Humidity"] = pd.NA


# --------------------------------------------------
# AQI cannot be negative
# --------------------------------------------------

df.loc[df["AQI"] < 0, "AQI"] = pd.NA


# --------------------------------------------------
# Remove duplicate rows
# --------------------------------------------------

df = df.drop_duplicates()


# --------------------------------------------------
# Remove rows where AQI is missing
# --------------------------------------------------

df = df.dropna(subset=["AQI"])


# --------------------------------------------------
# Split dataset into 80% training and 20% testing
# --------------------------------------------------

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# Save training and testing datasets
# --------------------------------------------------

train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\nData split completed successfully.")

print("Total dataset:", df.shape)
print("Training dataset:", train_df.shape)
print("Testing dataset:", test_df.shape)

print("\nTraining percentage:",
      round(len(train_df) / len(df) * 100, 2), "%")

print("Testing percentage:",
      round(len(test_df) / len(df) * 100, 2), "%")

print("\nTraining file saved as:")
print(TRAIN_FILE)

print("\nTesting file saved as:")
print(TEST_FILE)