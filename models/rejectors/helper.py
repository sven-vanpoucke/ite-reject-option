from models.evaluators.performance_evaluator import calculate_performance_metrics


# Function to train a model equal to the one in one_class_classification_rejector.py
def train_model(train_data, model_class, **model_params):
    model = model_class(**model_params)
    model.fit(train_data)
    return model

# Function to determine if a point is out of distribution based on a threshold
def is_too_far(distance, threshold):
    return distance > threshold


# Objective function
def calculate_objective(threshold_distance, *args):
    train_set = args[0]
    file_path = args[1]
    distances = args[2]
    key_metric = args[3]
    minmax = args[4]
    
    train_set['ood'] = distances.apply(is_too_far, threshold=threshold_distance)
    train_set['ite_reject'] = train_set.apply(lambda row: "R" if row['ood'] else row['ite_pred'], axis=1)

    metrics_dict = calculate_performance_metrics('ite', 'ite_reject', train_set, file_path)
    
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