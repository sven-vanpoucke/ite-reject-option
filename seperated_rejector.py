from datasets.lalonde import processing_get_data_lalonde, processing_transform_data_lalonde
from datasets.twins import preprocessing_get_data_twin, preprocessing_transform_data_twin
from datasets.processing import preprocessing_split_t_c_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# PREPROCESSING

# for lalaonde
# all_data = processing_get_data_lalonde()
# train_x, test_x, train_y, test_y, train_t, test_t = processing_transform_data_lalonde(all_data)

# for twins
train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y = preprocessing_get_data_twin()
train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y = preprocessing_transform_data_twin(train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y)

# split the data in treated and controlled
train_treated_x, train_control_x, train_treated_y, train_control_y, test_treated_x, test_control_x, test_treated_y, test_control_y = preprocessing_split_t_c_data(train_x, train_y, train_t, test_x, test_y, test_t)

# MODEL

from models.predictor import predictor_t_model

# Training separate models for treated and control groups
treated_model, control_model = predictor_t_model(train_treated_x, train_treated_y, train_control_x, train_control_y, model_class=LogisticRegression, max_iter=10000, solver='saga', random_state=42)

# REJECT

# Here should come the code to reject certain predictions in the test_set...

# PREDICT

# Predictions for treated and control groups
train_treated_y_pred = treated_model.predict(train_treated_x)
train_control_y_pred = control_model.predict(train_control_x)

test_treated_y_pred = treated_model.predict(test_treated_x)
test_control_y_pred = control_model.predict(test_control_x)

# EVALUATE
from models.evaluator import evaluation_binary
evaluation_binary(train_treated_y, train_treated_y_pred, train_control_y, train_control_y_pred)
evaluation_binary(test_treated_y, test_treated_y_pred, test_control_y, test_control_y_pred)