#!/usr/bin/env python3
"""
task4_train_ann.py
Train a feedforward ANN (PyBrain) on dataset03_training_ds.pkl (or dataset03_training.csv)
and save the trained network as UE_05_App3_ANN_Model.xml.
Also save a .pkl fallback copy (pybrain_trained_net_dataset03.pkl).
"""
import os
import sys
import pickle
import numpy as np

# ensure pybrain is importable
sys.path.append('/tmp/AIBAS_exercise_WorkingDirectory/pybrain')

from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
try:
    # NetworkWriter is used to write XML
    from pybrain.tools.xml.networkwriter import NetworkWriter
    XML_WRITER_AVAILABLE = True
except Exception:
    NetworkWriter = None
    XML_WRITER_AVAILABLE = False

# Paths
TRAIN_DS_PKL = "dataset03_training_ds.pkl"   # produced by task4_prepare_dataset03.py
XML_OUT = "UE_05_App3_ANN_Model.xml"
PKL_OUT = "pybrain_trained_net_dataset03.pkl"

# Hyperparameters (tune if needed)
HIDDEN_UNITS = 8
EPOCHS = 200
LEARNINGRATE = 0.01
MOMENTUM = 0.0

def load_pybrain_ds(pkl_path):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"{pkl_path} not found. Prepare dataset03 first.")
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    # we expect dict {'dataset': SupervisedDataSet, 'input_cols': [...]}
    if isinstance(obj, dict) and 'dataset' in obj:
        ds = obj['dataset']
        input_cols = obj.get('input_cols', None)
    else:
        # fallback: the pickle is the dataset itself
        ds = obj
        input_cols = None
    return ds, input_cols

def main():
    print("Loading PyBrain training dataset:", TRAIN_DS_PKL)
    ds_train, input_cols = load_pybrain_ds(TRAIN_DS_PKL)
    n_inputs = ds_train.indim
    n_outputs = ds_train.outdim
    print(f"Dataset dims: inputs={n_inputs}, outputs={n_outputs}")

    # Build feedforward network: input -> hidden -> output
    net = buildNetwork(n_inputs, HIDDEN_UNITS, n_outputs, bias=True)
    print("Network built: input->%d->output" % HIDDEN_UNITS)

    # Trainer
    trainer = BackpropTrainer(net, ds_train, learningrate=LEARNINGRATE, momentum=MOMENTUM, verbose=False)

    print(f"Training for {EPOCHS} epochs (lr={LEARNINGRATE}, momentum={MOMENTUM})")
    for epoch in range(1, EPOCHS + 1):
        mse = trainer.train()
        if epoch == 1 or epoch % 20 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch}/{EPOCHS} training MSE: {mse:.6f}")

    # Save as XML if possible
    if XML_WRITER_AVAILABLE:
        try:
            NetworkWriter.writeToFile(net, XML_OUT)
            print("Saved ANN model as XML:", XML_OUT)
        except Exception as e:
            print("Failed to save XML via NetworkWriter:", e)

    # fallback: pickle the network
    with open(PKL_OUT, "wb") as f:
        pickle.dump(net, f)
    print("Saved trained network (pickle) as:", PKL_OUT)

    if NetworkWriter is None:
        print("Note: XML write not available; pickle saved as fallback. If XML required, ensure pybrain XML writer is installed or use Python environment where NetworkWriter exists.")

if __name__ == '__main__':
    main()
