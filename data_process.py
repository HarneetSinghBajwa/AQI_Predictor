import pandas as pd


# --------------------------------------------------
# File names
# --------------------------------------------------

INPUT_FILE = "globalAQI_raw.csv"
OUTPUT_FILE = "clean_data.csv"


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
# Rename columns to simple names
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
# Fill missing input values using median
# --------------------------------------------------

for column in [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3",
    "Temperature",
    "Humidity"
]:
    median_value = df[column].median()
    df[column] = df[column].fillna(median_value)


# --------------------------------------------------
# Remove rows where AQI is missing
# AQI is our target variable
# --------------------------------------------------

df = df.dropna(subset=["AQI"])


# --------------------------------------------------
# Save cleaned dataset
# --------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("Processed dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nFinal columns:")
print(list(df.columns))

print("\nFirst 5 rows:")
print(df.head())

print("\nData processing completed successfully.")
print("Saved as:", OUTPUT_FILE)