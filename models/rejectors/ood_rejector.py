import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from models.evaluators.evaluator import calculate_all_metrics
from .helper import train_model, is_too_far, calculate_objective
import matplotlib.pyplot as plt
from models.evaluators.evaluator import calculate_performance_metrics

def distance_test_to_train(model, test_data, n_neighbors=4):
    # Use kneighbors method to get distances and indices of neighbors
    distances, _ = model.kneighbors(test_data, n_neighbors=n_neighbors)
    return pd.Series(distances[:, 0], index=test_data.index)

def execute_ood_experiment(train_x, model_class, test_x, bounds, test_set, train_set, file_path, key_metric, minmax, metrics_results, model_options=None):
    # Step 1: Train distance model
    trained_model = train_model(train_x, model_class=model_class, **model_options)
    distances = distance_test_to_train(trained_model, test_x)
    
    # Step 2: Train rejector: optimize the threshold
    result = minimize_scalar(calculate_objective, bounds=bounds, method='bounded', args=(train_set, file_path, distances, key_metric, minmax, train_set), options={'disp': False})
    threshold_distance = result.x
    print(f"Threshold Distance: {threshold_distance}")
    # Step 3: Apply rejector to the code
    test_set['ood'] = distances.apply(is_too_far, threshold=threshold_distance)
    test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['ood'] else row['ite_pred'], axis=1)
    calculate_all_metrics('ite', 'ite_reject', test_set, file_path, metrics_results, append_metrics_results=True, print=False)

    # Step 3: make a graph of the results
    # x-axis: threshold_distance
    # y-axis: key_metric
    threshold_distances = []
    key_metrics = []

    # #for threshold_distance in range(int(bounds[0] * 100), int(bounds[1] * 100) + 1):
    # for threshold_distance in range(int(1 * 100), int(20 * 100) + 1):
    #     threshold_distance /= 100
    #     test_set['ood'] = distances.apply(is_too_far, threshold=threshold_distance)
    #     test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['ood'] else row['ite_pred'], axis=1)
    #     # metrics_result = calculate_all_metrics('ite', 'ite_reject', test_set, file_path, metrics_results, append_metrics_results=False, print=False)    
    #     metrics_result = calculate_performance_metrics('ite', 'ite_reject', test_set, file_path)
    #     threshold_distances.append(threshold_distance)
    #     key_metrics.append(metrics_result[key_metric])
    #     print(f"Threshold Distance: {threshold_distance}, {key_metric}: {metrics_result[key_metric]}")

    # plt.plot(threshold_distances, key_metrics)
    # plt.xlabel('Threshold Distance')
    # plt.ylabel(key_metric)
    # plt.title('Threshold Distance vs. ' + key_metric)
    # plt.show()