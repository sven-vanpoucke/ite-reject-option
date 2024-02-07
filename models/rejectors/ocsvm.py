from sklearn.svm import OneClassSVM
import pandas as pd

# Assuming you have the necessary functions like 'nbrs_train', 'distance_test_to_train', etc.

# Function to train OCSVM model
def train_ocsvm(train_data):
    model = OneClassSVM()
    model.fit(train_data)
    return model

# Function to get distances from the OCSVM model
def distance_test_to_train_ocsvm(model, test_data):
    distances = model.decision_function(test_data)
    return pd.Series(distances, index=test_data.index)

# Function to determine if a point is out of distribution based on a threshold
def is_out_of_distribution_ocsvm(distance, threshold):
    return distance < threshold
