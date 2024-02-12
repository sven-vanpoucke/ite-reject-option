import pandas as pd
from scipy.optimize import minimize_scalar
from models.evaluators.evaluator import calculate_all_metrics
from .helper import train_model, is_too_far, calculate_objective

# Function to get distances
def distance_test_to_train(model, test_data):
    # use decision_function method to get distances from SVMs
    distances = model.decision_function(test_data)
    return pd.Series(distances, index=test_data.index)

def execute_one_class_classification_experiment(train_x, model_class, test_x, bounds, test_set, train_set, file_path, key_metric, minmax, metrics_results, model_options=None):
    # Step 1: Train distance model
    trained_model = train_model(train_x, model_class=model_class, **model_options)
    distances = distance_test_to_train(trained_model, train_x)

    #distances = distance_test_to_train(trained_model, test_x)
    
    # Step 2: Train rejector: optimize the threshold
    with open(file_path, 'a') as file:
        file.write(f"\nMode of the experiment {key_metric}: {minmax}")
    
    result = minimize_scalar(calculate_objective, bounds=bounds, method='bounded', args=(train_set, file_path, distances, key_metric, minmax), options={'disp': False})
    threshold_distance = result.x
    
    # Step 3: Apply rejector to the code
    distances = distance_test_to_train(trained_model, test_x)
    test_set['ood'] = distances.apply(is_too_far, threshold=threshold_distance)
    test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['ood'] else row['ite_pred'], axis=1)
    calculate_all_metrics('ite', 'ite_reject', test_set, file_path, metrics_results, append_metrics_results=True, print=False)