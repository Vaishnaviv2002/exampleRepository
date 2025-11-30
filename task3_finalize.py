import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Files
CLEAN_IN = "dataset02_cleaned.csv"
TRAIN_OUT = "dataset02_training.csv"
TEST_OUT = "dataset02_testing.csv"
MODEL_OUT = "OLS_model.txt"
SCATTER_PDF = "UE_04_App2_ScatterVisualizationAndOlsModel.pdf"
BOX_PDF = "UE_04_App2_BoxPlot.pdf"
DIAG_PDF = "UE_04_App2_DiagnosticPlots.pdf"

RNG_SEED = 42

# 1) load cleaned data
if not os.path.exists(CLEAN_IN):
    raise FileNotFoundError(f"{CLEAN_IN} not found. Make sure dataset02_cleaned.csv is present.")
df = pd.read_csv(CLEAN_IN)
print("Loaded cleaned data shape:", df.shape)

# check required target
if 'y' not in df.columns:
    raise KeyError("Target column 'y' not found. Make sure column name is exactly 'y'.")

# 2) shuffle & split 80/20
rng = np.random.default_rng(RNG_SEED)
perm = rng.permutation(len(df))
split_at = int(len(df) * 0.8)
train_idx = perm[:split_at]
test_idx  = perm[split_at:]

df_train = df.iloc[train_idx].reset_index(drop=True)
df_test  = df.iloc[test_idx].reset_index(drop=True)

df_train.to_csv(TRAIN_OUT, index=False)
df_test.to_csv(TEST_OUT, index=False)
print(f"Saved train ({len(df_train)}) -> {TRAIN_OUT}")
print(f"Saved test  ({len(df_test)})  -> {TEST_OUT}")

# 3) Fit OLS on training set only
X_train = df_train.drop(columns=['y'])
y_train = df_train['y']

X_train_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_const).fit()

with open(MODEL_OUT, "w") as f:
    f.write(model.summary().as_text())
print("Saved OLS model summary to", MODEL_OUT)

# 4) Scatter plots
influence_cols = [c for c in df.columns if c != 'y']
n = len(influence_cols)
fig_sc, axs = plt.subplots(n, 1, figsize=(8, 4*n), squeeze=False)

for i, col in enumerate(influence_cols):
    ax = axs[i,0]
    # training (orange), testing (blue)
    ax.scatter(df_train[col], df_train['y'], c='orange', marker='o', label='train')
    ax.scatter(df_test[col], df_test['y'], c='blue', marker='x', label='test')

    # OLS red line
    x_vals = np.linspace(df[col].min(), df[col].max(), 200)
    means = X_train.mean()
    df_line = pd.DataFrame({c: means[c] for c in X_train.columns}, index=range(len(x_vals)))
    df_line[col] = x_vals
    df_line_const = sm.add_constant(df_line)
    y_pred = model.predict(df_line_const)
    ax.plot(x_vals, y_pred, color='red', linewidth=1.5, label='OLS fit')

    ax.set_xlabel(col)
    ax.set_ylabel("y")
    ax.grid()
    ax.legend()

fig_sc.tight_layout()
fig_sc.savefig(SCATTER_PDF)
print("Saved scatter+OLS PDF:", SCATTER_PDF)

# 5) Boxplot
numeric = df.select_dtypes(include=[float, int])
fig_box, ax = plt.subplots(figsize=(10,6))
ax.boxplot([numeric[c] for c in numeric.columns], labels=numeric.columns)
ax.set_title("Boxplot of numeric columns")
fig_box.tight_layout()
fig_box.savefig(BOX_PDF)
print("Saved boxplot PDF:", BOX_PDF)

# 6) Diagnostic plots (try helper)
try:
    from UE_04_LinearRegDiagnostic import LinearRegDiagnostic
    diag = LinearRegDiagnostic(model, X_train_const, y_train)
    if hasattr(diag, "create_plots"):
        fig_d = diag.create_plots()
        fig_d.savefig(DIAG_PDF)
    elif hasattr(diag, "plot"):
        fig_d = diag.plot()
        fig_d.savefig(DIAG_PDF)
    print("Saved diagnostic PDF using helper:", DIAG_PDF)
except Exception as e:
    print("Helper not available, fallback:", e)
    import statsmodels.api as sm
    sm.graphics.plot_regress_exog(model, influence_cols[0])
    plt.savefig(DIAG_PDF)
    print("Saved fallback diagnostic PDF:", DIAG_PDF)

print("Task 3 finalization complete.")
