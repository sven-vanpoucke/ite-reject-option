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

def distance_test_to_train(model, test_data, n_neighbors=4):
    # Use kneighbors method to get distances and indices of neighbors
    distances, _ = model.kneighbors(test_data, n_neighbors=n_neighbors)
    return pd.Series(distances[:, 0], index=test_data.index)

# Function to determine if a point is out of distribution based on a threshold
def is_too_far(distance, threshold):
    return distance > threshold

# Objective function
def calculate_objective(threshold_distance, *args):
    test_set = args[0]
    file_path = args[1]
    distances = args[2]
    key_metric = args[3]
    minmax = args[4]

    # test_set['y_t1_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t1_prob'] < prob_reject_upper_bound else False, axis=1)
    # test_set['y_t0_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t0_prob'] < prob_reject_upper_bound else False, axis=1)
    # test_set['y_reject_prob'] = test_set.apply(lambda row: True if row['y_t0_reject_prob'] and row['y_t1_reject_prob'] else False, axis=1)
    # test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['y_reject_prob'] else row['ite_pred'], axis=1)
    
    test_set['ood'] = distances.apply(is_too_far, threshold=threshold_distance)
    test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['ood'] else row['ite_pred'], axis=1)

    metrics_dict = calculate_performance_metrics('ite', 'ite_reject', test_set, file_path)
    # Check if key_metric is in metrics_dict
    if key_metric in metrics_dict:
        metric = metrics_dict[key_metric]
    else:
        metric = 100
    
    if minmax == 'min':
        metric = metric
    else:
        metric = -metric

    with open(file_path, 'a') as file:
        file.write(f"\n Current value for the metric: {metric} with threshold: {threshold_distance}")

    return metric


def execute_ood_experiment(train_x, model_class, test_x, bounds, test_set, file_path, key_metric, minmax, metrics_results, model_options=None):
    # Step 1: Train distance model
    trained_model = train_model(train_x, model_class=model_class, **model_options)
    distances = distance_test_to_train(trained_model, test_x)
    # Step 2: Train rejector: optimize the threshold
    result = minimize_scalar(calculate_objective, bounds=bounds, method='bounded', args=(test_set, file_path, distances, key_metric, minmax), options={'disp': False})
    threshold_distance = result.x
    # Step 3: Apply rejector to the code
    test_set['ood'] = distances.apply(is_too_far, threshold=threshold_distance)
    test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['ood'] else row['ite_pred'], axis=1)
    calculate_all_metrics('ite', 'ite_reject', test_set, file_path, metrics_results, append_metrics_results=True, print=False)