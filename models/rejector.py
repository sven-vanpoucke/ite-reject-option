from sklearn.neighbors import NearestNeighbors
import pandas as pd 
from models.evaluator import calculate_crosstab
from models.cost import calculate_misclassification_cost
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
    
    accurancy, rr, micro_tpr, micro_fpr, macro_tpr, macro_fpr, micro_distance_threedroc, macro_distance_threedroc = calculate_crosstab('ite', 'ite_reject', test_set, file_path)
    # print(f"The current under bound is: {prob_reject_under_bound}")
    # print(f"The current upper bound is: {prob_reject_upper_bound}")
    # print(f"The current rejection rate is is: {rr}")
    # print(f"Thwith a micro distance threedroc of : {micro_distance_threedroc}")
    # print(rr)

    return micro_distance_threedroc


def calculate_objective_threedroc_double_variable(param, *args):
    prob_reject_under_bound, prob_reject_upper_bound = param
    test_set = args[0]
    file_path = args[1]

    # prob_reject_under_bound = 1 - prob_reject_upper_bound
    test_set['y_t1_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t1_prob'] < prob_reject_upper_bound else False, axis=1)
    test_set['y_t0_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t0_prob'] < prob_reject_upper_bound else False, axis=1)
    test_set['y_reject_prob'] = test_set.apply(lambda row: True if row['y_t0_reject_prob'] and row['y_t1_reject_prob'] else False, axis=1)
    test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['y_reject_prob'] else row['ite_pred'], axis=1)
    
    accurancy, rr, micro_tpr, micro_fpr, macro_tpr, macro_fpr, micro_distance_threedroc, macro_distance_threedroc = calculate_crosstab('ite', 'ite_reject', test_set, file_path)
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
