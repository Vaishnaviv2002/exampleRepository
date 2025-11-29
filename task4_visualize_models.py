#!/usr/bin/env python3
"""
task4_visualize_models.py
Create two visualizations:
 - UE_05_App3_SmallDatasetVisualization.pdf  (dataset02)
 - UE_05_App3_BigDatasetVisualization.pdf    (dataset03)

Requirements:
 - dataset02_cleaned.csv and dataset03_cleaned.csv exist in working dir
 - ANN model available as UE_05_App3_ANN_Model.xml OR pybrain_trained_net_dataset03.pkl
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# make sure pybrain clone is importable for fallback/pickle usage
sys.path.append('/tmp/AIBAS_exercise_WorkingDirectory/pybrain')
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure.connections.full import FullConnection

# File names
DATA02 = "dataset02_cleaned.csv"
DATA03 = "dataset03_cleaned.csv"
XML_NET = "UE_05_App3_ANN_Model.xml"
PKL_NET = "pybrain_trained_net_dataset03.pkl"

OUT_SMALL = "UE_05_App3_SmallDatasetVisualization.pdf"
OUT_BIG   = "UE_05_App3_BigDatasetVisualization.pdf"

# Which input column and target
TARGET = 'y'
# Try to find input column(s) (assume single influence x like before)
def find_input_cols(df):
    cols = [c for c in df.columns if c != TARGET]
    return cols

# --- Load ANN: prefer XML reconstruct, else load pickle ---
def build_network_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    modules_by_name = {}
    order = []
    for m_el in root.find("Modules").findall("Module"):
        name = m_el.get("name")
        mtype = m_el.get("type")
        size = int(m_el.get("size"))
        if mtype == "LinearLayer" and name == "in":
            layer = LinearLayer(size, name=name); kind = "input"
        elif mtype == "LinearLayer" and name == "out":
            layer = LinearLayer(size, name=name); kind = "output"
        elif mtype == "SigmoidLayer":
            layer = SigmoidLayer(size, name=name); kind = "hidden"
        elif mtype == "BiasUnit":
            layer = BiasUnit(name=name); kind = "bias"
        else:
            layer = SigmoidLayer(size, name=name); kind = "hidden"
        modules_by_name[name] = (layer, kind)
        order.append(name)
    net = FeedForwardNetwork()
    for name in order:
        layer, kind = modules_by_name[name]
        if kind == "input":
            net.addInputModule(layer)
        elif kind == "output":
            net.addOutputModule(layer)
        else:
            net.addModule(layer)
    for conn_el in root.find("Connections").findall("Connection"):
        frm = conn_el.get("from"); to = conn_el.get("to")
        inmod, _ = modules_by_name.get(frm)
        outmod, _ = modules_by_name.get(to)
        conn = FullConnection(inmod, outmod)
        net.addConnection(conn)
        # parse weight matrix rows
        w_el = conn_el.find("Weights")
        rows = []
        for row_el in w_el.findall("Row"):
            text = (row_el.text or "").strip()
            if text == "":
                row_vals = []
            else:
                row_vals = [float(v) for v in text.split()]
            rows.append(row_vals)
        if len(rows) > 0:
            W = np.array(rows, dtype=float)
            conn.params[:] = W.reshape(-1)
    net.sortModules()
    net.reset()
    return net

def load_ann():
    # prefer xml
    if os.path.exists(XML_NET):
        try:
            net = build_network_from_xml(XML_NET)
            print("Loaded ANN from XML:", XML_NET)
            return net
        except Exception as e:
            print("XML load failed:", e)
    # fallback to pickle
    if os.path.exists(PKL_NET):
        with open(PKL_NET, "rb") as f:
            net = pickle.load(f)
        print("Loaded ANN from pickle:", PKL_NET)
        return net
    raise FileNotFoundError("No ANN model found (XML or pickle).")

# --- Helpers: OLS fit & predictions ---
def fit_ols(df, input_cols):
    X = df[input_cols]
    Xc = sm.add_constant(X)
    y = df[TARGET]
    model = sm.OLS(y, Xc).fit()
    return model

def ann_predict(net, X):
    # X: numpy array shape (n_samples, n_inputs)
    preds = []
    for xi in X:
        preds.append(float(net.activate(tuple(xi))[0]))
    return np.array(preds)

# --- Plotting function ---
def plot_dataset(df, ann_net, out_pdf, title):
    input_cols = find_input_cols(df)
    if len(input_cols) != 1:
        raise ValueError("Visualization script expects exactly one influence column.")
    col = input_cols[0]
    x = df[col].values
    y = df[TARGET].values

    # OLS fit on full dataset for visualization
    ols_model = fit_ols(df, [col])
    x_line = np.linspace(x.min(), x.max(), 400)
    X_line = sm.add_constant(pd.DataFrame({col: x_line}))
    y_ols = ols_model.predict(X_line)

    # ANN predictions on grid
    ann_preds = ann_predict(ann_net, x_line.reshape(-1,1))

    # plot
    fig, ax = plt.subplots(figsize=(8,6))
    # dense red points
    ax.scatter(x, y, s=8, c='red', marker='+', label='data', alpha=0.6)
    # ANN black smooth curve
    ax.plot(x_line, ann_preds, color='black', linewidth=2, label='ANN (nonlinear)')
    # OLS blue line
    ax.plot(x_line, y_ols, color='blue', linewidth=2, label='OLS (linear)')
    # sinus basis at bottom (scaled and shifted for visibility)
    # creating a sinus curve for demonstration: amplitude scaled to data
    amp = 0.15 * (y.max() - y.min())
    sin_curve = amp * np.sin(2 * np.pi * (x_line - x_line.min())/(x_line.max()-x_line.min()) * 5)  # 5 cycles
    # shift it below min(y) for visual separation
    shift = y.min() - 1.5*amp
    ax.plot(x_line, sin_curve + shift, color='blue', linewidth=2, label='basis (demo)')

    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel(TARGET)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
    print("Saved:", out_pdf)

# --- Main ---
def main():
    ann_net = load_ann()

    # small dataset
    if not os.path.exists(DATA02):
        print("Warning: dataset02 file not found:", DATA02)
    else:
        df2 = pd.read_csv(DATA02)
        plot_dataset(df2, ann_net, OUT_SMALL, "Small dataset visualization (dataset02)")

    # big dataset
    if not os.path.exists(DATA03):
        print("Warning: dataset03 file not found:", DATA03)
    else:
        df3 = pd.read_csv(DATA03)
        plot_dataset(df3, ann_net, OUT_BIG, "Big dataset visualization (dataset03)")

if __name__ == "__main__":
    main()
