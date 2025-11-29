# task4_nn_example.py
# Small PyBrain NN example - trains on dataset02_training.csv and evaluates on dataset02_testing.csv
import sys
sys.path.append('/tmp/AIBAS_exercise_WorkingDirectory/pybrain')

import pandas as pd
import numpy as np
import pickle
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

# Paths
TRAIN_CSV = "dataset02_training.csv"
TEST_CSV  = "dataset02_testing.csv"
PRED_CSV  = "dataset02_test_predictions_pybrain.csv"
NET_PKL   = "pybrain_trained_net.pkl"

# Hyperparams
HIDDEN_UNITS = 5
EPOCHS = 100
LEARNINGRATE = 0.01
MOMENTUM = 0.0

# Load data
df_train = pd.read_csv(TRAIN_CSV)
df_test  = pd.read_csv(TEST_CSV)

# Basic checks
if 'y' not in df_train.columns:
    raise KeyError("Training CSV must contain column 'y' as target.")

input_cols = [c for c in df_train.columns if c != 'y']
n_inputs = len(input_cols)
n_outputs = 1

ds_train = SupervisedDataSet(n_inputs, n_outputs)
for _, row in df_train.iterrows():
    inputs = tuple(row[input_cols].astype(float).values)
    target = (float(row['y']),)
    ds_train.addSample(inputs, target)

ds_test_inputs = df_test[input_cols].astype(float).values
y_test = df_test['y'].astype(float).values

# Build network
net = buildNetwork(n_inputs, HIDDEN_UNITS, n_outputs, bias=True)

# Train
trainer = BackpropTrainer(net, ds_train, learningrate=LEARNINGRATE, momentum=MOMENTUM, verbose=False)
print("Training network: inputs=%d hidden=%d outputs=%d epochs=%d" % (n_inputs, HIDDEN_UNITS, n_outputs, EPOCHS))
for epoch in range(1, EPOCHS + 1):
    mse = trainer.train()
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{EPOCHS} training MSE: {mse:.6f}")

# Predict on test
y_pred = []
for xi in ds_test_inputs:
    y_p = net.activate(tuple(xi))
    y_pred.append(float(y_p[0]))

y_pred = np.array(y_pred)
mse_test = np.mean((y_test - y_pred) ** 2)
ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2_test = 1.0 - ss_res / (ss_tot + 1e-12)

print(f"Test MSE: {mse_test:.6f}")
print(f"Test R2: {r2_test:.6f}")

# Save predictions and network
pd.DataFrame({'y_true': y_test, 'y_pred': y_pred}).to_csv(PRED_CSV, index=False)
with open(NET_PKL, "wb") as f:
    pickle.dump(net, f)

print("Saved test predictions to", PRED_CSV)
print("Saved trained network to", NET_PKL)
