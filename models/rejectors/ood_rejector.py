import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from models.evaluators.evaluator import calculate_all_metrics
from .helper import train_model, is_too_far, calculate_objective
import matplotlib.pyplot as plt
from models.evaluators.evaluator import calculate_performance_metrics
from sklearn.preprocessing import StandardScaler

def distance_test_to_train(model, data, n_neighbors=4):
    # Use kneighbors method to get distances and indices of neighbors
    distances, _ = model.kneighbors(data, n_neighbors=n_neighbors)
    return pd.Series(distances[:, 0], index=data.index)

def execute_ood_experiment(train_x, model_class, test_x, bounds, test_set, train_set, file_path, key_metric, minmax, metrics_results, model_options=None):
    # Step 1: Train Scalar
    scaler = StandardScaler()
    train_x = pd.DataFrame(scaler.fit_transform(train_x))

    # Step 2: Train Distance model
    trained_model = train_model(train_x, model_class=model_class, **model_options)
    distances = distance_test_to_train(trained_model, train_x, 10)
    train_set['distance'] = distances
    # Step 3: Train rejector: optimize the threshold
    result = minimize_scalar(calculate_objective, bounds=bounds, method='bounded', args=(train_set, file_path, distances, key_metric, minmax, metrics_results), options={'disp': False})
    threshold_distance = result.x
    print(f"Threshold Distance: {threshold_distance}")

    # Step 4: Use rejector
    ## Scalor
    test_x = pd.DataFrame(scaler.transform(test_x)) # Use same scalor 
    ## Distances
    distances = distance_test_to_train(trained_model, test_x)
    test_set['distance'] = distances
    ## Apply rejector
    test_set['ood'] = distances.apply(is_too_far, threshold=threshold_distance)
    test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['ood'] else row['ite_pred'], axis=1)
    ## Calculate metrics
    metrics_dict = calculate_all_metrics('ite', 'ite_reject', test_set, file_path, metrics_results, append_metrics_results=True, print=False)

    # Step 5: make a graph of the results

    # x-axis: threshold_distance
    # y-axis: key_metric
    threshold_distances = []
    key_metrics = []
    reject_rates = []

    true_rejected = []
    false_rejected = []

    # #for threshold_distance in range(int(bounds[0] * 100), int(bounds[1] * 100) + 1):
    for threshold_distance in range(int(0 * 10), int(30 * 10) + 1):
        threshold_distance /= 100
        test_set['ood'] = distances.apply(is_too_far, threshold=threshold_distance)
        test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['ood'] else row['ite_pred'], axis=1)
        # metrics_result = calculate_all_metrics('ite', 'ite_reject', test_set, file_path, metrics_results, append_metrics_results=False, print=False)    
        metrics_result = calculate_performance_metrics('ite', 'ite_reject', test_set, file_path)
        
        if metrics_result is not None and key_metric in metrics_result:
            result = metrics_result[key_metric]
            key_metrics.append(result)
            reject_rates.append(metrics_result["Rejection Rate"])
        else:
                key_metrics.append(None)
                reject_rates.append(None)

        if metrics_result is not None and 'True Rejected' in metrics_result:
            true_rejected.append(metrics_result['True Rejected'])
        else:
             true_rejected.append(None)
        if metrics_result is not None and 'False Rejected' in metrics_result:
            false_rejected.append(metrics_result['False Rejected'])
        else:
            false_rejected.append(None)


        threshold_distances.append(threshold_distance)
        print(f"Threshold Distance: {threshold_distance}, {key_metric}: {result}")

    plt.plot(reject_rates, key_metrics)
    plt.xlabel('Reject Rate')
    plt.ylabel(key_metric)
    plt.title('Reject Rate vs. ' + key_metric)
    plt.show()

    plt.plot(reject_rates, true_rejected, color='green', label='True Rejected')
    plt.plot(reject_rates, false_rejected, color='red', label='False Rejected')
    plt.xlabel('Reject Rate')
    # plt.ylabel(key_metric)
    plt.title('Reject Rate')
    plt.legend()
    plt.show()
    plt.xlabel('Reject Rate')
    plt.ylabel(key_metric)
    plt.title('Reject Rate vs. ' + key_metric)
    plt.show()

    return metrics_dict