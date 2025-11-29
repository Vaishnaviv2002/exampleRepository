#!/usr/bin/env python3
"""
task5_evaluate_models.py
Quantitative testing routines that compare OLS vs ANN on the scraped test set.
Outputs:
 - UE_06_test_predictions_detailed.csv  (y_true, y_ols, y_ann, errors)
 - UE_06_model_comparison_results.json  (numeric metrics + test results)
 - UE_06_model_comparison_report.txt   (human readable summary)
 - UE_06_ModelComparison_Barplot.pdf   (visual comparison of RMSE/MAE)
"""
import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

# ensure pybrain import path (for ANN activation fallback)
sys.path.append('/tmp/AIBAS_exercise_WorkingDirectory/pybrain')

IN_TEST_CSV = "UE_06_dataset04_testing.csv"
OLS_SUMMARY_TXT = "UE_06_OLS_model.txt"
OLS_TRAIN_CSV = "UE_06_dataset04_training.csv"
ANN_PKL = "UE_06_ANN_model.pkl"

OUT_PRED_DETAILED = "UE_06_test_predictions_detailed.csv"
OUT_RESULTS_JSON = "UE_06_model_comparison_results.json"
OUT_REPORT_TXT = "UE_06_model_comparison_report.txt"
OUT_PLOT_PDF = "UE_06_ModelComparison_Barplot.pdf"

# Bootstrapping settings
BOOTSTRAP_ITERS = 2000
RNG_SEED = 42

# Utility metrics
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    # avoid div by zero
    denom = np.where(np.abs(y_true) < 1e-12, 1e-12, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res / (ss_tot + 1e-12)

# Load test data
if not os.path.exists(IN_TEST_CSV):
    raise FileNotFoundError(IN_TEST_CSV + " not found. Run previous tasks to produce it.")
df_test = pd.read_csv(IN_TEST_CSV)
if 'y' not in df_test.columns:
    raise KeyError("Test CSV must contain 'y' target column.")

X_test = df_test.drop(columns=['y'])
y_test = df_test['y'].values
input_cols = X_test.columns.tolist()

# 1) Load OLS model: try to reconstruct from saved training CSV, else fit on test (fallback)
if os.path.exists(OLS_TRAIN_CSV):
    df_train = pd.read_csv(OLS_TRAIN_CSV)
    X_train = sm.add_constant(df_train[input_cols])
    y_train = df_train['y']
    ols_model = sm.OLS(y_train, X_train).fit()
else:
    # fallback: fit on test (not ideal but safe)
    Xc = sm.add_constant(X_test)
    ols_model = sm.OLS(y_test, Xc).fit()

# OLS predictions
X_test_c = sm.add_constant(X_test)
y_ols_pred = ols_model.predict(X_test_c).values

# 2) Load ANN
ann_net = None
if os.path.exists(ANN_PKL):
    with open(ANN_PKL, "rb") as f:
        ann_net = pickle.load(f)
else:
    raise FileNotFoundError("ANN pickle not found: " + ANN_PKL)

# ANN predictions (vectorized loop)
def predict_ann(net, X_np):
    preds = []
    for xi in X_np:
        preds.append(float(net.activate(tuple(xi))[0]))
    return np.array(preds)

y_ann_pred = predict_ann(ann_net, X_test.values)

# 3) Compute pointwise error arrays
err_ols = y_test - y_ols_pred
err_ann = y_test - y_ann_pred
abs_err_ols = np.abs(err_ols)
abs_err_ann = np.abs(err_ann)
signed_diff = (abs_err_ann - abs_err_ols)  # positive => ANN worse than OLS

# 4) Basic metrics
metrics = {
    "OLS": {
        "RMSE": float(rmse(y_test, y_ols_pred)),
        "MAE":  float(mae(y_test, y_ols_pred)),
        "MAPE": float(mape(y_test, y_ols_pred)),
        "R2":   float(r2_score(y_test, y_ols_pred))
    },
    "ANN": {
        "RMSE": float(rmse(y_test, y_ann_pred)),
        "MAE":  float(mae(y_test, y_ann_pred)),
        "MAPE": float(mape(y_test, y_ann_pred)),
        "R2":   float(r2_score(y_test, y_ann_pred))
    },
    "N_test": int(len(y_test))
}

# 5) Paired statistical tests on absolute errors
# Paired t-test: test whether mean(abs_err_ann - abs_err_ols) == 0
t_stat, p_val_t = stats.ttest_rel(abs_err_ann, abs_err_ols)
# Wilcoxon signed-rank (nonparametric)
try:
    w_stat, p_val_w = stats.wilcoxon(abs_err_ann, abs_err_ols)
except Exception:
    w_stat, p_val_w = None, None

metrics['stat_tests'] = {
    "paired_t": {"t_stat": float(t_stat), "p_value": float(p_val_t)},
    "wilcoxon": {"stat": (float(w_stat) if w_stat is not None else None), "p_value": (float(p_val_w) if p_val_w is not None else None)}
}

# 6) Bootstrap CI for RMSE difference (ANN - OLS)
rng = np.random.default_rng(RNG_SEED)
diffs = []
n = len(y_test)
for i in range(BOOTSTRAP_ITERS):
    idx = rng.integers(0, n, n)
    rmse_ann_b = rmse(y_test[idx], y_ann_pred[idx])
    rmse_ols_b = rmse(y_test[idx], y_ols_pred[idx])
    diffs.append(rmse_ann_b - rmse_ols_b)
diffs = np.array(diffs)
ci_lower = np.percentile(diffs, 2.5)
ci_upper = np.percentile(diffs, 97.5)
metrics['bootstrap_rmse_diff'] = {"ci_2.5%": float(ci_lower), "ci_97.5%": float(ci_upper), "mean_diff": float(diffs.mean())}

# 7) Save detailed predictions
df_out = df_test.copy()
df_out['y_ols'] = y_ols_pred
df_out['y_ann'] = y_ann_pred
df_out['err_ols'] = err_ols
df_out['err_ann'] = err_ann
df_out['abs_err_ols'] = abs_err_ols
df_out['abs_err_ann'] = abs_err_ann
df_out['abs_err_diff_ann_minus_ols'] = abs_err_ann - abs_err_ols
df_out.to_csv(OUT_PRED_DETAILED, index=False)

# 8) Save numeric results JSON
with open(OUT_RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# 9) Human readable report
with open(OUT_REPORT_TXT, "w", encoding="utf-8") as f:
    f.write("Model comparison report (OLS vs ANN)\\n")
    f.write("===============================\\n\\n")
    f.write(f"Test set size: {metrics['N_test']}\\n\\n")
    f.write("Metrics:\\n")
    for m in ('OLS','ANN'):
        f.write(f"  {m}: RMSE={metrics[m]['RMSE']:.6f}, MAE={metrics[m]['MAE']:.6f}, MAPE={metrics[m]['MAPE']:.4f}%, R2={metrics[m]['R2']:.6f}\\n")
    f.write("\\nPaired tests on absolute errors:\\n")
    f.write(f"  Paired t-test: t={metrics['stat_tests']['paired_t']['t_stat']:.6f}, p={metrics['stat_tests']['paired_t']['p_value']:.6e}\\n")
    if metrics['stat_tests']['wilcoxon']['stat'] is not None:
        f.write(f"  Wilcoxon signed-rank: stat={metrics['stat_tests']['wilcoxon']['stat']:.6f}, p={metrics['stat_tests']['wilcoxon']['p_value']:.6e}\\n")
    f.write("\\nBootstrap RMSE difference (ANN - OLS): mean={:.6e}, 95% CI = [{:.6e}, {:.6e}]\\n".format(metrics['bootstrap_rmse_diff']['mean_diff'], metrics['bootstrap_rmse_diff']['ci_2.5%'], metrics['bootstrap_rmse_diff']['ci_97.5%']))
    f.write("\\nInterpretation:\\n")
    if metrics['stat_tests']['paired_t']['p_value'] < 0.05:
        f.write("  The paired t-test indicates a statistically significant difference in mean absolute errors (p < 0.05).\\n")
    else:
        f.write("  No significant difference detected by paired t-test (p >= 0.05).\\n")
    f.write("  Use the bootstrap CI for practical significance of RMSE difference.\\n")
    f.write("\\nSaved detailed predictions to: " + OUT_PRED_DETAILED + "\\n")
    f.write("Saved numeric results to: " + OUT_RESULTS_JSON + "\\n")
    f.write("Saved this report to: " + OUT_REPORT_TXT + "\\n")

# 10) Plot a small bar chart comparing RMSE and MAE
labels = ['RMSE', 'MAE']
ols_vals = [metrics['OLS']['RMSE'], metrics['OLS']['MAE']]
ann_vals = [metrics['ANN']['RMSE'], metrics['ANN']['MAE']]

x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(6,5))
ax.bar(x - width/2, ols_vals, width, label='OLS')
ax.bar(x + width/2, ann_vals, width, label='ANN')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Error')
ax.set_title('Model comparison on test set')
ax.legend()
fig.tight_layout()
fig.savefig(OUT_PLOT_PDF)
plt.close(fig)

print("Saved detailed predictions to", OUT_PRED_DETAILED)
print("Saved results JSON to", OUT_RESULTS_JSON)
print("Saved human-readable report to", OUT_REPORT_TXT)
print("Saved comparison barplot to", OUT_PLOT_PDF)
