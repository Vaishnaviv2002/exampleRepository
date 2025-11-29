import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load CSV
data = pd.read_csv("dataset01.csv")

# Extract columns
y = data['y']
x = data[['x']]  # x is the independent variable

# Print required statistics
print("Number of entries in y:", y.count())
print("Mean of y:", y.mean())
print("Standard deviation of y:", y.std())
print("Variance of y:", y.var())
print("Min of y:", y.min())
print("Max of y:", y.max())

# OLS Regression (y = a + b*x)
x_with_const = sm.add_constant(x)
model = sm.OLS(y, x_with_const).fit()

# Print model summary
print(model.summary())

# Save OLS model
with open("UE_05_App1_OLS_model.txt", "w") as f:
    f.write(model.summary().as_text())

print("OLS model saved to UE_05_App1_OLS_model.txt")

