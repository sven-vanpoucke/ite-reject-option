from sklearn.neighbors import NearestNeighbors
import pandas as pd 
from models.evaluators.performance_evaluator import calculate_performance_metrics
from models.evaluators.cost_evaluator import calculate_misclassification_cost, calculate_cost_metrics
def nbrs_train(train_data, n_neighbors=5):
    nbrs_train = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(train_data)
    return nbrs_train

def calculate_ood_distances(nbrs_train, test_data):
    distances_test_to_train, _ = nbrs_train.kneighbors(test_data)
    return distances_test_to_train.mean(axis=1)

def distance_test_to_train(nbrs_train, test_x):
    # Assuming train_treated_x and test_x are Pandas DataFrames
    distance_test_to_train = []

    # Iterate over rows in test_x
    for index, row in test_x.iterrows():
        # Calculate distance for each row in test_x against all train_x
        distance = calculate_ood_distances(nbrs_train, row.values.reshape(1, -1))
        distance_test_to_train.append(distance[0])

    return pd.Series(distance_test_to_train)

def is_out_of_distribution(distance, threshold_distance=3):
    return distance > threshold_distance


### 3D ROC
# Define the objective function
def calculate_objective_threedroc_single_variable(prob_reject_upper_bound, *args):
    test_set = args[0]
    file_path = args[1]

    prob_reject_under_bound = 1 - prob_reject_upper_bound
    test_set['y_t1_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t1_prob'] < prob_reject_upper_bound else False, axis=1)
    test_set['y_t0_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t0_prob'] < prob_reject_upper_bound else False, axis=1)
    test_set['y_reject_prob'] = test_set.apply(lambda row: True if row['y_t0_reject_prob'] and row['y_t1_reject_prob'] else False, axis=1)
    test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['y_reject_prob'] else row['ite_pred'], axis=1)
    
    metrics_dict = calculate_performance_metrics('ite', 'ite_reject', test_set, file_path)
    micro_distance_threedroc = metrics_dict['Micro Distance (3D ROC)']

    return micro_distance_threedroc


def calculate_objective(prob_reject_upper_bound, *args):
    test_set = args[0]
    file_path = args[1]
    key_metric = args[2]
    minmax = args[3]

    prob_reject_under_bound = 1 - prob_reject_upper_bound
    test_set['y_t1_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t1_prob'] < prob_reject_upper_bound else False, axis=1)
    test_set['y_t0_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t0_prob'] < prob_reject_upper_bound else False, axis=1)
    test_set['y_reject_prob'] = test_set.apply(lambda row: True if row['y_t0_reject_prob'] and row['y_t1_reject_prob'] else False, axis=1)
    test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['y_reject_prob'] else row['ite_pred'], axis=1)

    metrics_dict = calculate_performance_metrics('ite', 'ite_reject', test_set, file_path)

    # Check if key_metric is in metrics_dict
    if key_metric in metrics_dict:
        metric = metrics_dict[key_metric]
    else:
        metrics_dict = calculate_cost_metrics('ite', 'ite_reject', test_set, file_path)
        if key_metric in metrics_dict:
            metric = metrics_dict[key_metric]
        else:
            metric = 100
    if minmax == 'min':
        metric = metric
    else:
        metric = -metric

    with open(file_path, 'a') as file:
        file.write(f"\nCurrent value for the metric: {metric} with threshold: {prob_reject_upper_bound}")

    return metric


def calculate_objective_threedroc_double_variable(param, *args):
    prob_reject_under_bound, prob_reject_upper_bound = param
    test_set = args[0]
    file_path = args[1]

    # prob_reject_under_bound = 1 - prob_reject_upper_bound
    test_set['y_t1_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t1_prob'] < prob_reject_upper_bound else False, axis=1)
    test_set['y_t0_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t0_prob'] < prob_reject_upper_bound else False, axis=1)
    test_set['y_reject_prob'] = test_set.apply(lambda row: True if row['y_t0_reject_prob'] and row['y_t1_reject_prob'] else False, axis=1)
    test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['y_reject_prob'] else row['ite_pred'], axis=1)
    
    metrics_dict = calculate_performance_metrics('ite', 'ite_reject', test_set, file_path)
    micro_distance_threedroc = metrics_dict['Micro Distance (3D ROC)']

    # print(f"The current under bound is: {prob_reject_under_bound}")
    # print(f"The current upper bound is: {prob_reject_upper_bound}")
    # print(f"The current rejection rate is is: {rr}")
    # print(f"Thwith a micro distance threedroc of : {micro_distance_threedroc}")
    # print(rr)

    return micro_distance_threedroc

def calculate_objective_misclassificationcost_single_variable(prob_reject_upper_bound, *args):
    test_set = args[0]
    file_path = args[1]
    rejection_cost = 10

    prob_reject_under_bound = 1 - prob_reject_upper_bound
    test_set['y_t1_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t1_prob'] < prob_reject_upper_bound else False, axis=1)
    test_set['y_t0_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t0_prob'] < prob_reject_upper_bound else False, axis=1)
    test_set['y_reject_prob'] = test_set.apply(lambda row: True if row['y_t0_reject_prob'] and row['y_t1_reject_prob'] else False, axis=1)
    test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['y_reject_prob'] else row['ite_pred'], axis=1)
    
    # Calculate total misclassification cost
    total_cost_ite_2 = calculate_misclassification_cost(test_set, 2)

    return total_cost_ite_2
