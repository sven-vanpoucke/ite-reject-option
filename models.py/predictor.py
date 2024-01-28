from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Assuming train_x is your feature matrix, train_t is your treatment assignment variable,
# and train_y is your outcome variable

# Splitting the data into treated and control groups
treated_x = train_x[train_t == 1]
control_x = train_x[train_t == 0] # no treatment given...
treated_y = train_y[train_t == 1]
control_y = train_y[train_t == 0]

# Splitting the data into treated and control groups
treated_x_test = test_x[test_t == 1]
control_x_test = test_x[test_t == 0] # no treatment given...
treated_y_test = test_y[test_t == 1]
control_y_test = test_y[test_t == 0]

# Training separate models for treated and control groups
treated_model = LogisticRegression(max_iter=10000, solver='saga', random_state=42)
treated_model.fit(treated_x, treated_y)

control_model = LogisticRegression(max_iter=10000, solver='saga', random_state=42)
control_model.fit(control_x, control_y)

# Predictions for treated and control groups
treated_y_pred = treated_model.predict(treated_x_test)
control_y_pred = control_model.predict(control_x_test)
