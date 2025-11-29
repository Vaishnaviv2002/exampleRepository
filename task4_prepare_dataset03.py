#!/usr/bin/env python3
"""
task4_prepare_dataset03.py

Prepare dataset03 for PyBrain and create a big-dataset visualization:
 - load dataset03.csv
 - coerce non-numeric -> NaN, drop rows with NaN
 - optional IQR outlier removal
 - min-max normalization (kept for training)
 - save cleaned snapshot
 - create 80/20 train/test split
 - save CSV splits (optional)
 - create PyBrain SupervisedDataSet pickles for train & test
 - create a visualization PDF UE_05_App3_BigDatasetVisualization.pdf with axis ticks 0,10,20...100

Place dataset03.csv in the same folder before running.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ensure pybrain importable if needed later
sys.path.append('/tmp/AIBAS_exercise_WorkingDirectory/pybrain')

from pybrain.datasets import SupervisedDataSet

# Filenames (adjust if needed)
IN_CSV = "dataset03.csv"                       # input
CLEAN_CSV = "dataset03_cleaned.csv"
TRAIN_CSV = "dataset03_training.csv"
TEST_CSV  = "dataset03_testing.csv"
TRAIN_DS_PKL = "dataset03_training_ds.pkl"
TEST_DS_PKL  = "dataset03_testing_ds.pkl"
VIS_PDF = "UE_05_App3_BigDatasetVisualization.pdf"

# Settings
IQR_MULTIPLIER = 1.5   # set to None to skip IQR outlier removal
RNG_SEED = 42
TEST_FRAC = 0.2

def load_and_coerce(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Place dataset03.csv in the repository folder.")
    df = pd.read_csv(path)
    # coerce to numeric (non-numeric -> NaN) then drop rows with any NaN
    df_num = df.copy()
    for c in df_num.columns:
        df_num[c] = pd.to_numeric(df_num[c], errors='coerce')
    df_num = df_num.dropna(axis=0, how='any').reset_index(drop=True)
    return df_num

def drop_outliers_iqr(df, multiplier=1.5):
    if multiplier is None:
        return df
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        return df
    Q1 = numeric.quantile(0.25)
    Q3 = numeric.quantile(0.75)
    IQR = Q3 - Q1
    low = Q1 - multiplier * IQR
    high = Q3 + multiplier * IQR
    mask = ((numeric >= low) & (numeric <= high)).all(axis=1)
    return df[mask].reset_index(drop=True)

def normalize_minmax(df, scale_to_0_1=True):
    """
    If scale_to_0_1=True -> normalizes to [0,1] (training default).
    Returns the dataframe with numeric columns normalized and non-numeric kept as-is.
    """
    numeric = df.select_dtypes(include=[np.number])
    denom = (numeric.max() - numeric.min()).replace(0, 1)
    norm = (numeric - numeric.min()) / denom if scale_to_0_1 else numeric
    others = df.select_dtypes(exclude=[np.number]).reset_index(drop=True)
    return pd.concat([norm.reset_index(drop=True), others], axis=1)

def build_pybrain_dataset(df, target_col='y'):
    # returns (SupervisedDataSet, input_cols)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset.")
    input_cols = [c for c in df.columns if c != target_col]
    n_inputs = len(input_cols)
    n_outputs = 1
    ds = SupervisedDataSet(n_inputs, n_outputs)
    for _, row in df.iterrows():
        inputs = tuple(row[input_cols].astype(float).values)
        target = (float(row[target_col]),)
        ds.addSample(inputs, target)
    return ds, input_cols

def create_big_dataset_visualization(df_original):
    """
    Create a scatter visualization using the original (un-normalized) values,
    then set axis ticks to 0,10,20,...100 and save to VIS_PDF.
    """
    # Expect columns 'x' and 'y' or take the first two numeric columns
    numeric = df_original.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        # fallback: if only 1 numeric column, plot it against index
        x = numeric.iloc[:, 0].values
        y = np.arange(len(x))
        xlabel = numeric.columns[0]
        ylabel = "index"
    else:
        x = numeric.iloc[:, 0].values
        y = numeric.iloc[:, 1].values
        xlabel = numeric.columns[0]
        ylabel = numeric.columns[1]

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=6, alpha=0.6)
    plt.title("Big Dataset Visualization (original units)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    # Set axis ticks to 0,10,20,...100 (clip to data bounds)
    import math
    ax = plt.gca()
    # choose ticks from 0 to 100
    xticks = np.arange(0, 101, 10)
    yticks = np.arange(0, 101, 10)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    # Optionally limit the axes so ticks outside data range still show (keeps consistent scale)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(VIS_PDF)
    plt.close()
    print("Saved big dataset visualization to", VIS_PDF)

def main():
    print("Loading and coercing:", IN_CSV)
    df = load_and_coerce(IN_CSV)
    print("After coercion & drop NaN:", df.shape)

    # Outlier removal (IQR)
    if IQR_MULTIPLIER is not None:
        df_no_out = drop_outliers_iqr(df, multiplier=IQR_MULTIPLIER)
        print(f"After IQR outlier removal (mult={IQR_MULTIPLIER}):", df_no_out.shape)
    else:
        df_no_out = df
        print("IQR outlier removal skipped.")

    # Save cleaned original snapshot (un-normalized) for plotting in original units
    df_no_out.to_csv(CLEAN_CSV, index=False)
    print("Saved cleaned original dataset to", CLEAN_CSV)

    # Create visualization using original (un-normalized) values
    create_big_dataset_visualization(df_no_out)

    # Normalize numeric columns (min-max -> 0..1) for training
    df_norm = normalize_minmax(df_no_out, scale_to_0_1=True)
    print("After normalization sample:")
    print(df_norm.head())

    # Shuffle and split (80/20)
    n = len(df_norm)
    rng = np.random.default_rng(RNG_SEED)
    perm = rng.permutation(n)
    split_at = int(n * (1 - TEST_FRAC))
    train_idx = perm[:split_at]
    test_idx  = perm[split_at:]
    df_train = df_norm.iloc[train_idx].reset_index(drop=True)
    df_test  = df_norm.iloc[test_idx].reset_index(drop=True)

    # Save CSVs (if allowed)
    df_train.to_csv(TRAIN_CSV, index=False)
    df_test.to_csv(TEST_CSV, index=False)
    print(f"Saved train ({len(df_train)}) -> {TRAIN_CSV}")
    print(f"Saved test  ({len(df_test)}) -> {TEST_CSV}")

    # Build PyBrain SupervisedDataSet objects
    ds_train, input_cols = build_pybrain_dataset(df_train, target_col='y')
    ds_test, _ = build_pybrain_dataset(df_test, target_col='y')

    # pickle datasets (PyBrain format)
    with open(TRAIN_DS_PKL, "wb") as f:
        pickle.dump({'dataset': ds_train, 'input_cols': input_cols}, f)
    with open(TEST_DS_PKL, "wb") as f:
        pickle.dump({'dataset': ds_test, 'input_cols': input_cols}, f)

    print("Saved PyBrain datasets:", TRAIN_DS_PKL, TEST_DS_PKL)
    print("Done.")

if __name__ == "__main__":
    main()
