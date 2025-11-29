#!/usr/bin/env python3
"""
pythonApp.py - Task 03 (complete)
- Load dataset02.csv (dirty)
- Coerce non-numeric -> NaN, drop NaNs
- Drop outliers using IQR
- Normalize (min-max)
- Split 80/20 into dataset02_training.csv & dataset02_testing.csv
- Fit OLS on training set only and save OLS_model.txt
- Create scatter subplots (train orange, test blue) with red OLS line:
  -> UE_04_App2_ScatterVisualizationAndOlsModel.pdf
- Create boxplot -> UE_04_App2_BoxPlot.pdf
- Create diagnostic plots (try helper -> UE_04_LinearRegDiagnostic, else statsmodels fallback)
  -> UE_04_App2_DiagnosticPlots.pdf
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ---------- Parameters / Filenames ----------
IN_CSV = "dataset02.csv"                    # input dirty dataset (place it in repo)
CLEAN_CSV = "dataset02_cleaned.csv"         # cleaned + normalized snapshot
TRAIN_CSV = "dataset02_training.csv"        # 80% training set
TEST_CSV = "dataset02_testing.csv"          # 20% testing set
OLS_SUMMARY = "OLS_model.txt"

SCATTER_PDF = "UE_04_App2_ScatterVisualizationAndOlsModel.pdf"
BOX_PDF = "UE_04_App2_BoxPlot.pdf"
DIAG_PDF = "UE_04_App2_DiagnosticPlots.pdf"

IQR_MULTIPLIER = 1.5   # IQR filter multiplier (change to e.g. 3.0 for stricter)
RNG_SEED = 42          # reproducible split

# ---------- Helper functions ----------
def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Place dataset02.csv in the repo folder.")
    return pd.read_csv(path)

def coerce_numeric_and_dropna(df):
    df2 = df.copy()
    for col in df2.columns:
        df2[col] = pd.to_numeric(df2[col], errors='coerce')
    df2 = df2.dropna(axis=0, how='any').reset_index(drop=True)
    return df2

def drop_outliers_iqr(df, multiplier=1.5):
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

def normalize_minmax(df):
    numeric = df.select_dtypes(include=[np.number])
    denom = (numeric.max() - numeric.min()).replace(0, 1)
    norm = (numeric - numeric.min()) / denom
    others = df.select_dtypes(exclude=[np.number]).reset_index(drop=True)
    return pd.concat([norm.reset_index(drop=True), others], axis=1)

# ---------- Main ----------
def main():
    # 1) load
    print("Loading:", IN_CSV)
    df_raw = load_dataset(IN_CSV)
    print("Raw shape:", df_raw.shape)

    # 2) coerce numeric + drop NaNs
    df_num = coerce_numeric_and_dropna(df_raw)
    print("After coercion & drop NaN shape:", df_num.shape)

    # 3) drop outliers using IQR
    df_no_out = drop_outliers_iqr(df_num, multiplier=IQR_MULTIPLIER)
    print(f"After IQR outlier removal (mult={IQR_MULTIPLIER}) shape:", df_no_out.shape)

    # 4) normalize numeric columns (min-max)
    df_norm = normalize_minmax(df_no_out)
    print("After normalization sample:")
    print(df_norm.head())

    # save cleaned snapshot
    df_norm.to_csv(CLEAN_CSV, index=False)
    print("Saved cleaned+normalized dataset to", CLEAN_CSV)

    # 5) split 80/20 (seeded)
    n = len(df_norm)
    rng = np.random.default_rng(RNG_SEED)
    perm = rng.permutation(n)
    split_at = int(n * 0.8)
    train_idx = perm[:split_at]
    test_idx = perm[split_at:]
    df_train = df_norm.iloc[train_idx].reset_index(drop=True)
    df_test = df_norm.iloc[test_idx].reset_index(drop=True)

    df_train.to_csv(TRAIN_CSV, index=False)
    df_test.to_csv(TEST_CSV, index=False)
    print(f"Saved train ({len(df_train)}) -> {TRAIN_CSV}")
    print(f"Saved test  ({len(df_test)})  -> {TEST_CSV}")

    # 6) Fit OLS on training set only
    if 'y' not in df_train.columns:
        raise KeyError("Target column 'y' not found in training data. Column must be named 'y'.")

    X_train = df_train.drop(columns=['y'])
    y_train = df_train['y']
    X_test = df_test.drop(columns=['y'])
    y_test = df_test['y']

    X_train_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_const).fit()
    print(model.summary())

    # save model summary
    with open(OLS_SUMMARY, "w") as f:
        f.write(model.summary().as_text())
    print("Saved OLS model summary to", OLS_SUMMARY)

    # 7) Scatter plots (one figure with subplots for each influence)
    influence_cols = [c for c in df_train.columns if c != 'y']
    if len(influence_cols) == 0:
        print("No influence columns found (only 'y' present). Skipping scatter plot.")
    else:
        n_infl = len(influence_cols)
        fig_sc, axes = plt.subplots(n_infl, 1, figsize=(8, 4 * n_infl), squeeze=False)
        for i, col in enumerate(influence_cols):
            ax = axes[i, 0]
            # training: orange
            ax.scatter(df_train[col], df_train['y'], color='orange', marker='o', label='training', alpha=0.8)
            # testing: blue
            ax.scatter(df_test[col], df_test['y'], color='blue', marker='x', label='testing', alpha=0.8)

            # red OLS line: vary this column, keep others at mean
            x_vals = np.linspace(df_norm[col].min(), df_norm[col].max(), 200)
            means = X_train.mean()
            # build DF for line predictions
            df_line = pd.DataFrame({c: means[c] for c in X_train.columns}, index=range(len(x_vals)))
            df_line[col] = x_vals
            df_line_const = sm.add_constant(df_line)
            y_pred_line = model.predict(df_line_const)
            ax.plot(x_vals, y_pred_line, color='red', linewidth=1.5, label='OLS fit (line)')

            ax.set_xlabel(col)
            ax.set_ylabel('y')
            ax.legend()
            ax.grid(True)

        fig_sc.tight_layout()
        fig_sc.savefig(SCATTER_PDF)
        plt.close(fig_sc)
        print("Saved scatter+OLS figure to", SCATTER_PDF)

    # 8) Boxplot of all numeric dimensions
    numeric_all = df_norm.select_dtypes(include=[np.number])
    if numeric_all.shape[1] == 0:
        print("No numeric columns for boxplot. Skipping.")
    else:
        fig_box, ax_box = plt.subplots(figsize=(10, 6))
        ax_box.boxplot([numeric_all[c].values for c in numeric_all.columns], labels=numeric_all.columns)
        ax_box.set_title('Boxplot of numeric dimensions')
        ax_box.set_ylabel('value (normalized)')
        fig_box.tight_layout()
        fig_box.savefig(BOX_PDF)
        plt.close(fig_box)
        print("Saved boxplot figure to", BOX_PDF)

    # 9) Diagnostic plots: try helper UE_04_LinearRegDiagnostic then fallback
    def try_helper_and_save():
        try:
            from UE_04_LinearRegDiagnostic import LinearRegDiagnostic
        except Exception as e:
            print("UE_04_LinearRegDiagnostic import failed:", e)
            return False
        try:
            diag = LinearRegDiagnostic(model, X_train_const, y_train)
        except Exception as e:
            print("Could not instantiate LinearRegDiagnostic:", e)
            return False

        # try common method names
        for method_name in ("create_plots", "plot", "generate_plots", "make_plots"):
            if hasattr(diag, method_name):
                try:
                    res = getattr(diag, method_name)()
                    # If helper returned a Figure
                    import matplotlib.figure as mfig
                    if isinstance(res, mfig.Figure):
                        res.savefig(DIAG_PDF)
                        print("Saved diagnostic PDF via helper to", DIAG_PDF)
                        return True
                    # If helper returned None, maybe it saved itself
                    if res is None:
                        if os.path.exists(DIAG_PDF):
                            print("Helper created DIAG_PDF:", DIAG_PDF)
                            return True
                        else:
                            print("Helper returned None and did not create PDF.")
                            return False
                    # Best-effort: save current figure(s)
                    plt.gcf().savefig(DIAG_PDF)
                    print("Saved current figure to", DIAG_PDF)
                    return True
                except Exception as e:
                    print(f"Calling helper.{method_name}() failed:", e)
        print("Helper present but no usable plotting method found.")
        return False

    def fallback_statsmodels_plots():
        resid = model.resid
        fitted = model.fittedvalues
        influence = model.get_influence()
        leverage = influence.hat_matrix_diag
        cooks = influence.cooks_distance[0]

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        # Residuals vs Fitted
        ax1 = axs[0, 0]
        ax1.scatter(fitted, resid, alpha=0.6)
        ax1.axhline(0, color='red', linestyle='--', linewidth=1)
        ax1.set_xlabel('Fitted values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Fitted')

        # Normal Q-Q
        ax2 = axs[0, 1]
        sm.qqplot(resid, line='45', ax=ax2)
        ax2.set_title('Normal Q-Q')

        # Scale-Location
        ax3 = axs[1, 0]
        ax3.scatter(fitted, np.sqrt(np.abs(resid)), alpha=0.6)
        ax3.set_xlabel('Fitted values')
        ax3.set_ylabel('Sqrt(|residuals|)')
        ax3.set_title('Scale-Location')

        # Leverage vs Residuals (size ~ Cook's distance)
        ax4 = axs[1, 1]
        sc = ax4.scatter(leverage, resid, s=40 + 300 * (cooks / (cooks.max() + 1e-12)), alpha=0.6)
        ax4.set_xlabel('Leverage')
        ax4.set_ylabel('Residuals')
        ax4.set_title("Leverage vs Residuals (size ~ Cook's distance)")

        fig.tight_layout()
        fig.savefig(DIAG_PDF)
        plt.close(fig)
        print("Saved fallback diagnostic plots to", DIAG_PDF)
        return True

    saved = try_helper_and_save()
    if not saved:
        fallback_statsmodels_plots()

    print("\nTask 03 complete. All outputs (cleaned dataset, train/test split, model summary and PDFs) written to repository folder.")

if __name__ == "__main__":
    main()

