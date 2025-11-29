"""
pythonApp.py  - Task 03 (Stage 1)
Performs:
 - load dataset02.csv
 - coerce columns to numeric, drop NaNs
 - remove outliers using IQR filter
 - normalize numeric columns (min-max)
 - save cleaned dataset as dataset02_cleaned.csv
"""
import os
import pandas as pd
import numpy as np

# Parameters
IN_CSV = "dataset02.csv"
OUT_CLEAN = "dataset02_cleaned.csv"
IQR_MULTIPLIER = 1.5  # change to 3.0 for stricter filtering

def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Place dataset02.csv in the repo folder.")
    return pd.read_csv(path)

def coerce_numeric_and_dropna(df):
    # coerce to numeric where possible; non-convertible become NaN
    df2 = df.copy()
    for col in df2.columns:
        df2[col] = pd.to_numeric(df2[col], errors='coerce')
    # drop rows containing any NaN
    df2 = df2.dropna(axis=0, how='any').reset_index(drop=True)
    return df2

def drop_outliers_iqr(df, multiplier=1.5):
    numeric = df.select_dtypes(include=[np.number])
    Q1 = numeric.quantile(0.25)
    Q3 = numeric.quantile(0.75)
    IQR = Q3 - Q1
    low = Q1 - multiplier * IQR
    high = Q3 + multiplier * IQR
    # mask rows where ALL numeric columns are within [low, high]
    mask = ((numeric >= low) & (numeric <= high)).all(axis=1)
    return df[mask].reset_index(drop=True)

def normalize_minmax(df):
    numeric = df.select_dtypes(include=[np.number])
    denom = (numeric.max() - numeric.min()).replace(0, 1)  # avoid division by zero
    norm = (numeric - numeric.min()) / denom
    non_numeric = df.select_dtypes(exclude=[np.number])
    return pd.concat([norm, non_numeric.reset_index(drop=True)], axis=1)

def main():
    print("Loading:", IN_CSV)
    df_raw = load_dataset(IN_CSV)
    print("Raw shape:", df_raw.shape)

    df_numeric = coerce_numeric_and_dropna(df_raw)
    print("After coercion & drop NaN shape:", df_numeric.shape)

    df_no_out = drop_outliers_iqr(df_numeric, multiplier=IQR_MULTIPLIER)
    print(f"After IQR outlier removal (mult={IQR_MULTIPLIER}) shape:", df_no_out.shape)

    df_norm = normalize_minmax(df_no_out)
    print("After min-max normalization sample:\n", df_norm.head())

    df_norm.to_csv(OUT_CLEAN, index=False)
    print("Saved cleaned dataset to", OUT_CLEAN)

if __name__ == '__main__':
    main()
