import pandas as pd
import os

# ==========================================
# WILDFIRE DETECTOR - DATA CLEANING
# ==========================================

# 1. Load the original dataset
input_file = "fire_archive_M-C61_786974.csv.xls"
output_file = "clean_data.csv"

if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"Dataset not found: {input_file}"
    )

df = pd.read_csv(input_file)

print("========== ORIGINAL DATASET ==========")
print("Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())

original_rows = len(df)

# 2. Check required columns
required_columns = [
    "acq_date", "acq_time", "latitude",
    "longitude", "confidence", "frp",
    "satellite", "daynight"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )

# 3. Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 4. Check duplicate records
print("\nDuplicate Records:", df.duplicated().sum())

# 5. Remove duplicate records
df = df.drop_duplicates().copy()

# 6. Convert columns into correct data types
df["acq_date"] = pd.to_datetime(
    df["acq_date"], errors="coerce"
)

numeric_columns = [
    "latitude", "longitude",
    "confidence", "frp", "acq_time"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

# 7. Handle missing values in essential columns
essential_columns = [
    "acq_date", "acq_time",
    "latitude", "longitude",
    "confidence", "frp"
]

df = df.dropna(subset=essential_columns).copy()

# Fill missing values in other numerical columns
for col in df.select_dtypes(include="number").columns:
    if df[col].isnull().any():
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

# Fill missing categorical values
for col in df.select_dtypes(include="object").columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna("Unknown")

# 8. Data validation - remove invalid records
valid_time = (
    df["acq_time"].between(0, 2359)
    & (df["acq_time"] % 100 < 60)
)

df = df[
    df["latitude"].between(-90, 90)
    & df["longitude"].between(-180, 180)
    & df["confidence"].between(0, 100)
    & (df["frp"] >= 0)
    & valid_time
].copy()

# 9. Standardize text formatting
df["satellite"] = (
    df["satellite"].astype(str).str.strip().str.upper()
)

df["daynight"] = (
    df["daynight"].astype(str).str.strip().str.upper()
)

# 10. Format acquisition time
df["acq_time"] = (
    df["acq_time"].astype(int).astype(str).str.zfill(4)
)

# 11. Create a combined date and time column
df["acq_datetime"] = pd.to_datetime(
    df["acq_date"].dt.strftime("%Y-%m-%d")
    + " "
    + df["acq_time"].str[:2]
    + ":"
    + df["acq_time"].str[2:],
    errors="coerce"
)

# Remove records with invalid combined datetime
df = df.dropna(subset=["acq_datetime"]).copy()

# 12. Final data validation
assert df["latitude"].between(-90, 90).all()
assert df["longitude"].between(-180, 180).all()
assert df["confidence"].between(0, 100).all()
assert (df["frp"] >= 0).all()
assert df["acq_datetime"].notna().all()
assert not df.duplicated().any()

# 13. Save the cleaned dataset
df.to_csv(output_file, index=False)

# 14. Display final results
print("\n========== CLEANING SUMMARY ==========")

print("Original Records:", original_rows)
print("Cleaned Records:", len(df))
print("Removed Records:", original_rows - len(df))

print("\nRemaining Missing Values:")
print(df.isnull().sum())

print("\nTotal Missing Values:",
      df.isnull().sum().sum())

print("\nDuplicate Records After Cleaning:",
      df.duplicated().sum())

print("\nFinal Dataset Shape:", df.shape)

print("\nCleaned Dataset Preview:")
print(df.head())

print("\nCleaned dataset saved successfully as:",
      output_file)

print("\n========== DATA CLEANING COMPLETED ==========")