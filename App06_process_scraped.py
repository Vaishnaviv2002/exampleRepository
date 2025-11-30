#!/usr/bin/env python3
"""
App06_process_scraped.py
Take markdown tables extracted from README (README_table_*.csv) or parse README_downloaded.md,
concatenate them, clean numeric columns, drop NaNs, remove outliers (IQR), normalize (min-max),
and save as UE_06_dataset04_joint_scraped_data.csv
"""
import os
import glob
import pandas as pd
import numpy as np
import csv
import re

OUT_CSV = "UE_06_dataset04_joint_scraped_data.csv"
IQR_MULTIPLIER = 1.5   # change to None to skip outlier removal

def read_table_csvs():
    files = sorted(glob.glob("README_table_*.csv"))
    dfs = []
    if files:
        for f in files:
            try:
                df = pd.read_csv(f)
                print(f"Loaded table file: {f} -> shape {df.shape}")
                dfs.append(df)
            except Exception as e:
                print(f"Failed to read {f}: {e}")
    return dfs

def try_extract_table_from_readme():
    # fallback: look for the first markdown table block inside README_downloaded.md
    fn = "README_downloaded.md"
    if not os.path.exists(fn):
        return None
    text = open(fn, "r", encoding="utf-8").read()
    lines = text.splitlines()
    # find first header |...| then separator |---|
    for i in range(len(lines)-1):
        if '|' in lines[i] and re.search(r'\|[\s:-]*\|', lines[i+1]):
            # collect until blank or non-pipe line
            tbl = [lines[i], lines[i+1]]
            j = i+2
            while j < len(lines) and ('|' in lines[j] and lines[j].strip() != ""):
                tbl.append(lines[j])
                j += 1
            # write temp CSV and load
            tmp = "README_table_extracted_tmp.csv"
            # parse header and rows
            def parse_row(line):
                s = line.strip()
                if s.startswith('|'): s = s[1:]
                if s.endswith('|'): s = s[:-1]
                return [c.strip() for c in s.split('|')]
            header = parse_row(tbl[0])
            data = [parse_row(r) for r in tbl[2:]]
            # save
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header)
                for r in data:
                    # pad if needed
                    if len(r) < len(header):
                        r = r + [''] * (len(header) - len(r))
                    w.writerow(r)
            try:
                df = pd.read_csv(tmp)
                os.remove(tmp)
                print("Extracted table from README_downloaded.md -> shape", df.shape)
                return df
            except Exception as e:
                print("Failed to parse extracted table:", e)
                if os.path.exists(tmp): os.remove(tmp)
                return None
    return None

def coerce_and_clean(df):
    # drop completely empty columns
    df = df.dropna(axis=1, how='all')
    # Try to coerce all columns to numeric if possible; keep columns that are numeric after coercion
    numeric_cols = []
    df2 = df.copy()
    for c in df.columns:
        # try convert to numeric with coercion
        coerced = pd.to_numeric(df[c], errors='coerce')
        # if at least half values convert to numeric, keep numeric version
        num_non_na = coerced.notna().sum()
        if num_non_na >= max(1, int(0.5 * len(coerced))):
            df2[c] = coerced
            numeric_cols.append(c)
        else:
            # drop non-numeric column
            df2 = df2.drop(columns=[c])
            print(f"Dropping non-numeric column: {c}")
    # drop rows with any NaN
    before = len(df2)
    df2 = df2.dropna(axis=0, how='any').reset_index(drop=True)
    after = len(df2)
    print(f"Dropped rows with NaN: before={before} after={after}")
    return df2, numeric_cols

def drop_outliers_iqr(df, numeric_cols, multiplier=1.5):
    if multiplier is None or len(numeric_cols) == 0:
        return df
    numeric = df[numeric_cols]
    Q1 = numeric.quantile(0.25)
    Q3 = numeric.quantile(0.75)
    IQR = Q3 - Q1
    low = Q1 - multiplier * IQR
    high = Q3 + multiplier * IQR
    mask = ((numeric >= low) & (numeric <= high)).all(axis=1)
    before = len(df)
    df_filtered = df[mask].reset_index(drop=True)
    after = len(df_filtered)
    print(f"IQR outlier removal (mult={multiplier}): before={before} after={after}")
    return df_filtered

def minmax_normalize(df, numeric_cols):
    df_norm = df.copy()
    for c in numeric_cols:
        col = df_norm[c]
        denom = (col.max() - col.min())
        if denom == 0 or np.isnan(denom):
            df_norm[c] = 0.0
        else:
            df_norm[c] = (col - col.min()) / denom
    return df_norm

def main():
    print("Starting App06 scraped-table processing...")
    dfs = read_table_csvs()
    if not dfs:
        print("No README_table_*.csv files found; attempting to extract a table from README_downloaded.md")
        df_ex = try_extract_table_from_readme()
        if df_ex is not None:
            dfs = [df_ex]
    if not dfs:
        print("No tables found to process. Exiting.")
        return 1
    # concat all tables with same columns by intersection
    # find common columns
    all_cols = [set(d.columns.tolist()) for d in dfs]
    common_cols = set.intersection(*all_cols) if all_cols else set()
    if not common_cols:
        # if no common, use union but align by position / heuristics — here we just union and fill missing
        print("No common columns across tables; aligning by union of columns.")
        df_concat = pd.concat(dfs, ignore_index=True, sort=False)
    else:
        # keep only common columns
        print("Common columns detected:", common_cols)
        df_list = [d[list(common_cols)].copy() for d in dfs]
        df_concat = pd.concat(df_list, ignore_index=True, sort=False)
    print("Concatenated shape:", df_concat.shape)
    # coerce and clean numeric
    df_clean, numeric_cols = coerce_and_clean(df_concat)
    if df_clean.empty:
        print("No numeric rows left after coercion/NaN drop. Exiting.")
        return 1
    # outlier removal
    df_filtered = drop_outliers_iqr(df_clean, numeric_cols, IQR_MULTIPLIER)
    # normalization
    df_norm = minmax_normalize(df_filtered, numeric_cols)
    # final save
    df_norm.to_csv(OUT_CSV, index=False)
    print("Saved cleaned joint dataset to:", OUT_CSV)
    print("Final shape:", df_norm.shape)
    print("Numeric columns normalized:", numeric_cols)
    return 0

if __name__ == "__main__":
    exit(main())
