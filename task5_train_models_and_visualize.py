#!/usr/bin/env python3
"""
task5_train_models_and_visualize.py
- Load scraped dataset UE_06_dataset04_joint_scraped_data.csv
- Shuffle + 80/20 split (seeded)
- Fit OLS on training set and save summary (UE_06_OLS_model.txt)
- Train PyBrain feedforward ANN on training set, save pickle (UE_06_ANN_model.pkl) and XML (UE_06_ANN_Model.xml)
- Evaluate both models on test set (RMSE, R^2)
- Produce:
    - UE_06_Scraped_ScatterVisualizationAndOlsModel.pdf
    - UE_06_Scraped_BoxPlot.pdf
    - UE_06_Scraped_DiagnosticPlots.pdf
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# make sure pybrain clone available
sys.path.append('/tmp/AIBAS_exercise_WorkingDirectory/pybrain')
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
try:
    from pybrain.tools.xml.networkwriter import NetworkWriter
    XML_AVAILABLE = True
except Exception:
    NetworkWriter = None
    XML_AVAILABLE = False
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure.connections.full import FullConnection

# File names
IN_CSV = "UE_06_dataset04_joint_scraped_data.csv"
TRAIN_CSV = "UE_06_dataset04_training.csv"
TEST_CSV  = "UE_06_dataset04_testing.csv"
OLS_SUMMARY = "UE_06_OLS_model.txt"
ANN_PKL = "UE_06_ANN_model.pkl"
ANN_XML = "UE_06_ANN_Model.xml"

SCATTER_PDF = "UE_06_Scraped_ScatterVisualizationAndOlsModel.pdf"
BOX_PDF     = "UE_06_Scraped_BoxPlot.pdf"
DIAG_PDF    = "UE_06_Scraped_DiagnosticPlots.pdf"

RNG_SEED = 42
TEST_FRAC = 0.2

# Hyperparams for ANN
HIDDEN_UNITS = 8
EPOCHS = 200
LEARNINGRATE = 0.01
MOMENTUM = 0.0

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Prepare dataset first.")
    df = pd.read_csv(path)
    return df

def train_ols(X_train, y_train):
    Xc = sm.add_constant(X_train)
    model = sm.OLS(y_train, Xc).fit()
    return model

def save_ols_summary(model, path):
    with open(path, "w") as f:
        f.write(model.summary().as_text())
    print("Saved OLS summary to", path)

def train_ann(X_train, y_train):
    n_inputs = X_train.shape[1]
    n_outputs = 1
    ds = []
    # build pybrain SupervisedDataSet inline to train (we'll avoid pickling the dataset)
    from pybrain.datasets import SupervisedDataSet
    ds_train = SupervisedDataSet(n_inputs, n_outputs)
    for _, row in X_train.iterrows():
        inputs = tuple(row.values.astype(float))
        target = (float(y_train.loc[row.name]),)
        ds_train.addSample(inputs, target)
    net = buildNetwork(n_inputs, HIDDEN_UNITS, n_outputs, bias=True)
    trainer = BackpropTrainer(net, ds_train, learningrate=LEARNINGRATE, momentum=MOMENTUM, verbose=False)
    print(f"Training ANN: inputs={n_inputs}, hidden={HIDDEN_UNITS}, outputs={n_outputs}, epochs={EPOCHS}")
    for epoch in range(1, EPOCHS+1):
        mse = trainer.train()
        if epoch == 1 or epoch % 50 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch}/{EPOCHS} training MSE: {mse:.6f}")
    return net

def save_ann_pickle(net, path):
    with open(path, "wb") as f:
        pickle.dump(net, f)
    print("Saved ANN pickle to", path)

def export_ann_xml(net, xml_out):
    # try NetworkWriter if available
    if XML_AVAILABLE and NetworkWriter is not None:
        try:
            NetworkWriter.writeToFile(net, xml_out)
            print("Saved ANN XML via NetworkWriter:", xml_out)
            return True
        except Exception as e:
            print("NetworkWriter failed:", e)
    # else manual export as earlier
    root = ET.Element("FeedForwardNetwork")
    root.set("modules", str(len(net.modules)))
    modules_el = ET.SubElement(root, "Modules")
    for m in net.modules:
        mod_el = ET.SubElement(modules_el, "Module")
        mod_el.set("name", m.name)
        mod_el.set("size", str(getattr(m, "outdim", 0)))
        mod_el.set("type", m.__class__.__name__)
        mod_el.set("bias", str(getattr(m, "bias", False)))
    conns_el = ET.SubElement(root, "Connections")
    for connlist in net.connections.values():
        for c in connlist:
            conn_el = ET.SubElement(conns_el, "Connection")
            conn_el.set("from", c.inmod.name)
            conn_el.set("to", c.outmod.name)
            conn_el.set("type", c.__class__.__name__)
            w_el = ET.SubElement(conn_el, "Weights")
            # c.params flattened; reconstruct rows as outdim x indim
            outdim = c.outdim
            indim = c.indim
            arr = np.array(c.params).reshape(outdim, indim)
            for i, row in enumerate(arr):
                row_el = ET.SubElement(w_el, "Row")
                row_el.set("index", str(i))
                row_el.text = " ".join(f"{v:.10f}" for v in row)
    tree = ET.ElementTree(root)
    tree.write(xml_out, encoding="utf-8", xml_declaration=True)
    print("Saved ANN XML (manual) to", xml_out)
    return True

def predict_ann(net, X_np):
    preds = []
    for xi in X_np:
        preds.append(float(net.activate(tuple(xi))[0]))
    return np.array(preds)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res / (ss_tot + 1e-12)

def plot_scatter_ols_ann(df_train, df_test, ols_model, ann_net, out_pdf):
    input_cols = [c for c in df_train.columns if c != 'y']
    if len(input_cols) != 1:
        raise ValueError("This plotting routine expects one input column.")
    col = input_cols[0]
    x_all = pd.concat([df_train[col], df_test[col]])
    x_line = np.linspace(x_all.min(), x_all.max(), 400)
    # OLS predictions
    X_line = sm.add_constant(pd.DataFrame({col: x_line}))
    y_ols_line = ols_model.predict(X_line)
    # ANN predictions
    ann_preds_line = predict_ann(ann_net, x_line.reshape(-1,1))
    # Plot
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(df_train[col], df_train['y'], c='orange', marker='o', label='training', alpha=0.7)
    ax.scatter(df_test[col], df_test['y'], c='blue', marker='x', label='testing', alpha=0.7)
    ax.plot(x_line, y_ols_line, color='blue', linewidth=2, label='OLS fit')
    ax.plot(x_line, ann_preds_line, color='black', linewidth=2, label='ANN fit')
    ax.set_xlabel(col); ax.set_ylabel('y')
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
    print("Saved scatter+OLS+ANN to", out_pdf)

def plot_boxplot(df, out_pdf):
    numeric = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10,6))
    ax.boxplot([numeric[c] for c in numeric.columns], labels=numeric.columns)
    ax.set_title("Boxplot of numeric features")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
    print("Saved boxplot to", out_pdf)

def diagnostic_plots(ols_model, out_pdf):
    resid = ols_model.resid
    fitted = ols_model.fittedvalues
    influence = ols_model.get_influence()
    leverage = influence.hat_matrix_diag
    cooks = influence.cooks_distance[0]
    fig, axs = plt.subplots(2,2, figsize=(12,10))
    # Residuals vs Fitted
    ax1 = axs[0,0]; ax1.scatter(fitted, resid, alpha=0.6); ax1.axhline(0, color='red', linestyle='--'); ax1.set_title('Residuals vs Fitted'); ax1.set_xlabel('Fitted'); ax1.set_ylabel('Residuals')
    # Q-Q
    ax2 = axs[0,1]; sm.qqplot(resid, line='45', ax=ax2); ax2.set_title('Normal Q-Q')
    # Scale-Location
    ax3 = axs[1,0]; ax3.scatter(fitted, np.sqrt(np.abs(resid)), alpha=0.6); ax3.set_title('Scale-Location'); ax3.set_xlabel('Fitted'); ax3.set_ylabel('Sqrt(|resid|)')
    # Leverage vs Residuals
    ax4 = axs[1,1]; ax4.scatter(leverage, resid, s=40 + 300*(cooks / (cooks.max()+1e-12)), alpha=0.6); ax4.set_title("Leverage vs Residuals (size ~ Cook's)")
    fig.tight_layout(); fig.savefig(out_pdf); plt.close(fig)
    print("Saved diagnostic plots to", out_pdf)

def main():
    print("Loading scraped dataset:", IN_CSV)
    df = load_data(IN_CSV)
    print("Shape:", df.shape)
    # split
    n = len(df)
    rng = np.random.default_rng(RNG_SEED)
    perm = rng.permutation(n)
    split_at = int(n * (1 - TEST_FRAC))
    train_idx = perm[:split_at]; test_idx = perm[split_at:]
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)
    df_train.to_csv(TRAIN_CSV, index=False)
    df_test.to_csv(TEST_CSV, index=False)
    print("Saved train/test splits:", TRAIN_CSV, TEST_CSV, "sizes:", len(df_train), len(df_test))

    # Fit OLS on train only
    X_train = df_train.drop(columns=['y'])
    y_train = df_train['y']
    X_test = df_test.drop(columns=['y'])
    y_test = df_test['y']

    ols_model = train_ols(X_train, y_train)
    save_ols_summary(ols_model, OLS_SUMMARY)

    # Train ANN on train only
    ann_net = train_ann(X_train, y_train)
    save_ann_pickle(ann_net, ANN_PKL)
    export_ann_xml(ann_net, ANN_XML)

    # Evaluate on test
    # OLS predictions
    X_test_c = sm.add_constant(X_test)
    y_ols_pred = ols_model.predict(X_test_c)
    # ANN predictions
    y_ann_pred = predict_ann(ann_net, X_test.values)

    # Metrics
    ols_rmse = rmse(y_test.values, y_ols_pred)
    ann_rmse = rmse(y_test.values, y_ann_pred)
    ols_r2 = r2_score(y_test.values, y_ols_pred)
    ann_r2 = r2_score(y_test.values, y_ann_pred)

    print("\n=== Test metrics (scraped test set) ===")
    print(f"OLS  RMSE: {ols_rmse:.6f}, R2: {ols_r2:.6f}")
    print(f"ANN  RMSE: {ann_rmse:.6f}, R2: {ann_r2:.6f}")

    # Save predictions for inspection
    pd.DataFrame({'y_true': y_test.values, 'y_ols': y_ols_pred, 'y_ann': y_ann_pred}).to_csv("UE_06_test_predictions.csv", index=False)
    print("Saved test predictions to UE_06_test_predictions.csv")

    # Plots: scatter (train+test), boxplot (all), diagnostics (OLS)
    plot_scatter_ols_ann(df_train, df_test, ols_model, ann_net, SCATTER_PDF)
    plot_boxplot(pd.concat([df_train, df_test], ignore_index=True), BOX_PDF)
    diagnostic_plots(ols_model, DIAG_PDF)

    print("\nDone. Outputs:\n", OLS_SUMMARY, ANN_PKL, ANN_XML, SCATTER_PDF, BOX_PDF, DIAG_PDF, "UE_06_test_predictions.csv")

if __name__ == "__main__":
    main()
