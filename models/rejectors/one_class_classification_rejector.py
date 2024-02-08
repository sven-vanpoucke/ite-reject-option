from sklearn.svm import OneClassSVM
import pandas as pd
from models.evaluators.performance_evaluator import calculate_performance_metrics
from scipy.optimize import minimize_scalar, minimize
from models.evaluators.evaluator import calculate_all_metrics

# Function to train a model
def train_model(train_data, model_class=OneClassSVM, **model_params):
    model = model_class(**model_params)
    model.fit(train_data)
    return model

# Function to get distances
def distance_test_to_train(model, test_data):
    # use decision_function method to get distances from SVMs
    distances = model.decision_function(test_data)
    return pd.Series(distances, index=test_data.index)

# Function to determine if a point is out of distribution based on a threshold
def is_too_far(distance, threshold):
    return distance < threshold

# Objective function
def calculate_objective(threshold_distance, *args):
    test_set = args[0]
    file_path = args[1]
    distances_ocsvm = args[2]
    #key_metric = "Micro Distance (3D ROC)"
    key_metric = args[3]

    # test_set['y_t1_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t1_prob'] < prob_reject_upper_bound else False, axis=1)
    # test_set['y_t0_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t0_prob'] < prob_reject_upper_bound else False, axis=1)
    # test_set['y_reject_prob'] = test_set.apply(lambda row: True if row['y_t0_reject_prob'] and row['y_t1_reject_prob'] else False, axis=1)
    # test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['y_reject_prob'] else row['ite_pred'], axis=1)
    
    test_set['ood'] = distances_ocsvm.apply(is_too_far, threshold=threshold_distance)
    test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['ood'] else row['ite_pred'], axis=1)

    metrics_dict = calculate_performance_metrics('ite', 'ite_reject', test_set, file_path)

    # Check if key_metric is in metrics_dict
    if key_metric in metrics_dict:
        metric = metrics_dict[key_metric]
    else:
        metric = 100
    return metric

def execute_one_class_classification_experiment(train_x, model_class, test_x, bounds, test_set, file_path, key_metric, metrics_results, model_options=None):
    # Step 1: Train distance model
    trained_model = train_model(train_x, model_class=model_class, **model_options)
    distances = distance_test_to_train(trained_model, test_x)
    # Step 2: Train rejector: optimize the threshold
    result = minimize_scalar(calculate_objective, bounds=bounds, method='bounded', args=(test_set, file_path, distances, key_metric), options={'disp': False})
    threshold_distance = result.x
    # Step 3: Apply rejector to the code
    test_set['ood'] = distances.apply(is_too_far, threshold=threshold_distance)
    test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['ood'] else row['ite_pred'], axis=1)
    calculate_all_metrics('ite', 'ite_reject', test_set, file_path, metrics_results, append_metrics_results=True, print=False)