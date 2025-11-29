#!/usr/bin/env python3
"""
Compare saved ANN XML model vs pickle model on two identical data entries.

- Loads first two rows from dataset03_testing.csv (inputs)
- Loads trained network from pybrain_trained_net_dataset03.pkl (pickle)
- Loads XML model UE_05_App3_ANN_Model.xml and reconstructs a PyBrain network
- Activates both networks on the same inputs and prints outputs

Change TEST_INDICES to choose different test rows.
"""
import sys
import pickle
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

# ensure pybrain is importable
sys.path.append('/tmp/AIBAS_exercise_WorkingDirectory/pybrain')

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure.connections.full import FullConnection

# Paths
TEST_CSV = "dataset03_testing.csv"
PICKLE_NET = "pybrain_trained_net_dataset03.pkl"
XML_NET = "UE_05_App3_ANN_Model.xml"

# Choose which test rows to use (zero-based indices within dataset03_testing.csv)
TEST_INDICES = [0, 1]

def load_inputs_from_test(indices):
    df = pd.read_csv(TEST_CSV)
    if max(indices) >= len(df):
        raise IndexError("Requested test index out of range.")
    input_cols = [c for c in df.columns if c != 'y']
    X = df.loc[indices, input_cols].astype(float).values
    return X, input_cols

def load_pickle_network(pkl_path):
    with open(pkl_path, "rb") as f:
        net = pickle.load(f)
    return net

def build_network_from_xml(xml_path):
    """
    Reconstruct a pybrain FeedForwardNetwork from the XML format we exported.
    Uses addInputModule/addOutputModule so buffers are created correctly.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # parse modules
    modules_by_name = {}
    order = []  # preserve order for inspection (not required)
    for m_el in root.find("Modules").findall("Module"):
        name = m_el.get("name")
        mtype = m_el.get("type")
        size = int(m_el.get("size"))
        # create appropriate module type
        if mtype == "LinearLayer" and name == "in":
            layer = LinearLayer(size, name=name)
            kind = "input"
        elif mtype == "LinearLayer" and name == "out":
            layer = LinearLayer(size, name=name)
            kind = "output"
        elif mtype == "SigmoidLayer":
            layer = SigmoidLayer(size, name=name)
            kind = "hidden"
        elif mtype == "BiasUnit":
            layer = BiasUnit(name=name)
            kind = "bias"
        else:
            # fallback: choose SigmoidLayer for unknown hidden types
            layer = SigmoidLayer(size, name=name)
            kind = "hidden"
        modules_by_name[name] = (layer, kind)
        order.append(name)

    # create network and add modules using correct add*Module calls
    net = FeedForwardNetwork()
    for name in order:
        layer, kind = modules_by_name[name]
        if kind == "input":
            net.addInputModule(layer)
        elif kind == "output":
            net.addOutputModule(layer)
        else:
            # bias or hidden -> generic addModule
            net.addModule(layer)

    # parse connections and add them
    for conn_el in root.find("Connections").findall("Connection"):
        frm = conn_el.get("from")
        to = conn_el.get("to")
        inmod, _ = modules_by_name.get(frm)
        outmod, _ = modules_by_name.get(to)
        if inmod is None or outmod is None:
            raise ValueError(f"Module not found for connection {frm} -> {to}")
        conn = FullConnection(inmod, outmod)
        net.addConnection(conn)

        # read weight rows
        w_el = conn_el.find("Weights")
        rows = []
        for row_el in w_el.findall("Row"):
            text = (row_el.text or "").strip()
            if text == "":
                row_vals = []
            else:
                row_vals = [float(v) for v in text.split()]
            rows.append(row_vals)
        if len(rows) == 0:
            continue
        W = np.array(rows, dtype=float)
        flat = W.reshape(-1)
        conn.params[:] = flat

    # finalize network
    net.sortModules()
    net.reset()
    return net

def activate_and_print(net, X, label):
    print(f"\nActivations for model: {label}")
    for i, xi in enumerate(X):
        out = net.activate(tuple(xi))
        print(f" Input #{i} -> {out}")

def main():
    print("Loading two inputs from test CSV:", TEST_CSV)
    X, input_cols = load_inputs_from_test(TEST_INDICES)
    print("Input columns:", input_cols)
    print("Inputs (first two rows):")
    print(X)

    # load pickle network
    print("\nLoading pickle network:", PICKLE_NET)
    net_pickle = load_pickle_network(PICKLE_NET)

    # load xml network
    print("Reconstructing network from XML:", XML_NET)
    net_xml = build_network_from_xml(XML_NET)

    # Activate both on the same inputs
    activate_and_print(net_pickle, X, "pickle-trained-network")
    activate_and_print(net_xml, X, "xml-reconstructed-network")

if __name__ == "__main__":
    main()
