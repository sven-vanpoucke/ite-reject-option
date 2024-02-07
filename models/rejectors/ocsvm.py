from sklearn.svm import OneClassSVM
import pandas as pd
from models.evaluator import calculate_crosstab
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


### 3D ROC
# Define the objective function
def calculate_objective_threedroc_threshold(threshold_distance, *args):
    test_set = args[0]
    file_path = args[1]
    distances_ocsvm = args[2]

    # test_set['y_t1_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t1_prob'] < prob_reject_upper_bound else False, axis=1)
    # test_set['y_t0_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t0_prob'] < prob_reject_upper_bound else False, axis=1)
    # test_set['y_reject_prob'] = test_set.apply(lambda row: True if row['y_t0_reject_prob'] and row['y_t1_reject_prob'] else False, axis=1)
    # test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['y_reject_prob'] else row['ite_pred'], axis=1)
    test_set['ood'] = distances_ocsvm.apply(is_out_of_distribution_ocsvm, threshold=threshold_distance)
    test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['ood'] else row['ite_pred'], axis=1)

    accurancy, rr, micro_tpr, micro_fpr, macro_tpr, macro_fpr, micro_distance_threedroc, macro_distance_threedroc = calculate_crosstab('ite', 'ite_reject', test_set, file_path)
    # print(f"The current under bound is: {prob_reject_under_bound}")
    # print(f"The current upper bound is: {prob_reject_upper_bound}")
    # print(f"The current rejection rate is is: {rr}")
    # print(f"Thwith a micro distance threedroc of : {micro_distance_threedroc}")
    # print(rr)

    return micro_distance_threedroc
