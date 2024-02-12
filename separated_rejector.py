"""
Table of contents:
# CHAPTER 0: Imports

# CHAPTER 1: Initialization

# CHAPTER 2: Preprocessing
## Chapter 2A: Output to file
## Chapter 2B: Retrieval of data

# CHAPTER 3: Training of the ITE Model
## Chapter 3A: Output to file
## Chapter 3B: Training of the ITE Model
## Chapter 3C: Predicting the ITE and related variables (y_t0 and y_t1)

# CHAPTER 4: Evaluate treated and control groups seperately
## Chapter 4A: Output to file
## Chapter 4B: Evaluation of the individual models based on the training data
## Chapter 4C: Evaluation of the individual models based on the training data

# CHAPTER 5: Evaluate overall ITE Model: Performance
## Chapter 5A: Output to file
## Chapter 5B: Preprocessing of the test_set
## Chapter 5C: Calculate and report performance measurements

# CHAPTER 6: Evaluate overall ITE Model: Costs
## Chapter 6A: Output to file
## Chapter 6B: Preprocessing of the test_set
## Chapter 6C: Calculate and report performance measurements

# CHAPTER 7: REJECTION
# Baseline Model - No Rejection // Experiment 0
# OOD: K-Nearest Neighbors // Experiment 1
# One Class Classification - OCSVM // Experiment 2
# Probabilities symetric upper and under bound // Experiment 3
# Probabilities asymetric upper and under bound // Experiment 4
# REJECTION TYPE 2A: REJECTION BASED ON PROBABILITIES BY MINIMIZING MISCLASSIFICATION COSTS // Experiment 5


# CHAPTER 8: Output to file

"""

# Chapter 0: Imports
import pandas as pd
from tabulate import tabulate
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
## INIT
from models.helper import helper_output
## PREPROCESSING
from datasets.lalonde import processing_get_data_lalonde, processing_transform_data_lalonde
from datasets.twins import preprocessing_get_data_twin, preprocessing_transform_data_twin
from datasets.processing import preprocessing_split_t_c_data
## MODEL T-LEARNER
from models.predictor import predictor_t_model
from sklearn.linear_model import LogisticRegression
## PREDICT 
from models.predictor import predictor_train_predictions, predictor_test_predictions, predictor_ite_predictions
## EVALUATE INDIVIDUAL MODELS
from models.model_evaluator import evaluation_binary
# EVALUATE OVERALL ITE MODEL: PERFORMANCE
from models.evaluators.cost_evaluator import categorize # calculate_performance_metrics
## EVALUATE OVERALL ITE MODEL: COSTS
from models.evaluators.cost_evaluator import calculate_cost_ite
## REJECTION
from models.evaluators.evaluator import calculate_all_metrics
from models.rejectors.one_class_classification_rejector import execute_one_class_classification_experiment
from models.rejectors.ood_rejector import execute_ood_experiment
## REJECTION OOD
from models.rejectors.dependent_prob_rejector import distance_test_to_train, is_out_of_distribution, nbrs_train
## REJECTION OOD - OCSVM
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
## REJECTION PROBABILITIES
from scipy.optimize import minimize_scalar, minimize
from models.rejectors.dependent_prob_rejector import calculate_objective_threedroc_double_variable, calculate_objective_misclassificationcost_single_variable, calculate_objective
from models.evaluators.evaluator import calculate_performance_metrics

# Chapter 1: Initialization
## Parameters
folder_path = 'output/'
dataset = "TWINS" # Choose out of TWINS or LALONDE
model_class = LogisticRegression # Which two models do we want to generate in the t-models
rejection_architecture = 'dependent' # dependent_rejector or separated_rejector
prob_reject_upper_bound = 0.55
prob_reject_under_bound = 0.45
timestamp, file_name, file_path = helper_output(folder_path=folder_path)
## Assuming you have the following metrics for each experiment
metrics_results = {}

# Chapter 2: Preprocessing
## Chapter 2A: Output to file
with open(file_path, 'a') as file:
    file.write(f"\Chapter 2: Preprocessing\n\n")
    file.write("# This section executes the data retrieval, preprocessing and splitting in a training and dataset.")
    file.write(f"During the whole file, the used dataset is: {dataset}\n\n")

## Chapter 2B: Retrieval of data
if dataset == "LALONDE":
    all_data = processing_get_data_lalonde()
    train_x, test_x, train_y, test_y, train_t, test_t = processing_transform_data_lalonde(all_data)
elif dataset == "TWINS":
    train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y = preprocessing_get_data_twin()
    train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y = preprocessing_transform_data_twin(train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y)
    # Calculate ITE
    test_ite = pd.DataFrame({'ite': test_potential_y["y_t1"] - test_potential_y["y_t0"]})
    train_ite = pd.DataFrame({'ite': train_potential_y["y_t1"] - train_potential_y["y_t0"]})
    # split the data in treated and controlled
    train_treated_x, train_control_x, train_treated_y, train_control_y, test_treated_x, test_control_x, test_treated_y, test_control_y = preprocessing_split_t_c_data(train_x, train_y, train_t, test_x, test_y, test_t)

# Chapter 3: Training of the ITE Model
## Chapter 3A: Output to file
with open(file_path, 'a') as file:
    file.write(f"CHAPTER 3: Training of the ITE Model\n\n")
    file.write("# This section provides details about the model selection, training process, and any hyperparameter tuning.\n")
    file.write(f"The trained ITE model is a T-LEARNER.\n")
    file.write(f"The two individually trained models are: {model_class.__name__}\n\n")

## Chapter 3B: Training of the ITE Model
## We adopt an T-leraner as our ITE model. This model is trained on the treated and control groups separately.
treated_model, control_model = predictor_t_model(train_treated_x, train_treated_y, train_control_x, train_control_y, model_class=model_class, max_iter=10000, solver='saga', random_state=42)

## Chapter 3C: Predicting the ITE and related variables (y_t0 and y_t1)

## Training and Testing predictions to evaluate individual models
train_treated_y_pred, train_treated_y_prob, train_control_y_pred, train_control_y_prob = predictor_train_predictions(treated_model, control_model, train_treated_x, train_control_x)
test_treated_y_pred, test_treated_y_prob, test_control_y_pred, test_control_y_prob = predictor_test_predictions(treated_model, control_model, test_treated_x, test_control_x)
train_y_t1_pred, train_y_t0_pred, train_y_t1_prob, train_y_t0_prob, train_ite_prob, train_ite_pred = predictor_ite_predictions(treated_model, control_model, train_x)

# Testing Predictions to evaluate ITE
test_y_t1_pred, test_y_t0_pred, test_y_t1_prob, test_y_t0_prob, test_ite_prob, test_ite_pred = predictor_ite_predictions(treated_model, control_model, test_x)

# Chapter 4: Evaluate treated and control groups seperately
## Chapter 4A: Output to file
with open(file_path, 'a') as file:
    file.write("CHAPTER 4: Evaluate treated and control groups seperately\n\n")
    file.write("# This section evaluates the individually trained models (two as we used a T-learner).\n")
    file.write("The used performance measures are:\n\n")
    file.write(" - Confusion Matrix\n")
    file.write(" - Accuracy: overall correctness of the model ((TP + TN) / (TP + TN + FP + FN))\n")
    file.write(" - Precision: It measures the accuracy of positive predictions (TP / (TP + FP))\n")
    file.write(" - Recall: ability of the model to capture all the relevant cases (TP / (TP + FN))\n")
    file.write(" - F1 Score: It balances precision and recall, providing a single metric for model evaluation (2 * (Precision * Recall) / (Precision + Recall))\n")
    file.write(" - ROC\n\n")

## Chapter 4B: Evaluation of the individual models based on the training data
with open(file_path, 'a') as file:
    file.write("Evaluation of the individual models based on the **training data**\n")
evaluation_binary(train_treated_y, train_treated_y_pred, train_treated_y_prob, train_control_y, train_control_y_pred, train_control_y_prob, file_path)

## Chapter 4C: Evaluation of the individual models based on the training data
with open(file_path, 'a') as file:
    file.write("\nEvaluation of the individual models based on the **test data**\n")
evaluation_binary(test_treated_y, test_treated_y_pred, test_treated_y_prob, test_control_y, test_control_y_pred, test_control_y_prob, file_path)

# Chapter 5: Evaluate overall ITE Model: Performance
## Chapter 5A: Output to file
with open(file_path, 'a') as file:
    file.write(f"Chapter 4: Evaluate overall ITE Model: Performance \n\n")
    file.write("# This section evaluates the overal performance of the ITE model.\n")
    file.write(f"The used performance measures are: \n\n")
    file.write(f" - Root Mean Squared Error (RMSE) of the ITE \n")
    file.write(f" - Accurate estimate of the ATE \n")
    file.write(f" - Accurancy of ITE\n")

## Chapter 5B: Preprocessing of the test_set
test_set = pd.concat([test_t, test_y_t1_pred, test_y_t1_prob, test_y_t0_pred, test_y_t0_prob, test_ite_pred, test_ite_prob, test_potential_y["y_t0"], test_potential_y["y_t1"], test_ite], axis=1)
train_set = pd.concat([test_t, train_y_t1_pred, train_y_t1_prob, train_y_t0_pred, train_y_t0_prob, train_ite_pred, train_ite_prob, train_potential_y["y_t0"], train_potential_y["y_t1"], train_ite], axis=1)

# Chapter 6: Evaluate overall ITE Model: Costs
## Chapter 6A: Output to file
with open(file_path, 'a') as file:
    file.write(f"\n\nCHAPTER 7: EVALUATE OVERALL ITE MODEL: COST \n\n")
    file.write("# This section evaluates the overal misclassification costs of the ITE model.\n")

## Chapter 5B: Preprocessing of the test_set
### Apply the categorization function to create the 'Category' column
test_set['category'] = test_set.apply(categorize, axis=1, is_pred=False)
test_set['category_pred'] = test_set.apply(categorize, axis=1)
test_set['category_rej'] = test_set.apply(categorize, axis=1)

#######################################################################################################################

# CHAPTER 7: REJECTION
## CHAPTER 7A: Output to file
with open(file_path, 'a') as file:
    file.write(f"\nCHAPTER 7: REJECTION \n\n")
    file.write("# This section executes and reports metrics for ITE models with rejection.\n")
    
#######################################################################################################################
# Baseline Model - No Rejection // Experiment 0
experiment_id = 0
experiment_names = {}
experiment_names.update({experiment_id: f"No Rejector - Baseline Model"})

    # Step 4 Apply rejector to the code
test_set['ite_reject'] = test_set.apply(lambda row: row['ite_pred'], axis=1)
train_set['ite_reject'] = test_set.apply(lambda row: row['ite_pred'], axis=1)

    # Step 5 Calculate the performance metrics
calculate_all_metrics('ite', 'ite_reject', test_set, file_path, metrics_results, append_metrics_results=True, print=False)

#######################################################################################################################
# Architecture Type = Separated
#######################################################################################################################
# OOD - KNN

experiment_id += 1
experiments = [
    {
        'id': experiment_id,
        'architecture': "Separated Rejector",
        'model_class': NearestNeighbors,
        'bounds': (0,25),
        'key_metric': "Combined Quality",
        'minmax': 'max',
        'model_options': {'n_neighbors': 5, 'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'p': 2, 'n_jobs': None}
    },
]

def run_experiment_ood(experiment_id, architecture, model_class, bounds, key_metric, minmax, model_options, train_x, test_x, test_set, file_path, metrics_results, experiment_names):
    with open(file_path, 'a') as file:
        file.write(f"\n\nRunning Experiment {experiment_id} - {architecture} - {model_class.__name__} with optimizing {key_metric}")
    experiment_names.update({experiment_id: f"{architecture} - {model_class.__name__} with optimizing {key_metric}"})

    execute_ood_experiment(train_x, model_class, test_x, bounds, test_set, train_set, file_path, key_metric, minmax, metrics_results, model_options)

# Execute experiments
for experiment in experiments:
    run_experiment_ood(experiment['id'], experiment['architecture'], experiment['model_class'], 
                   experiment['bounds'], experiment['key_metric'], experiment['minmax'], experiment['model_options'], train_x, test_x, test_set, 
                   file_path, metrics_results, experiment_names, )
#######################################################################################################################
# Rejection based on Logistic Regression Prediction of Rejection
experiment_id += 1
architecture="Dependent architecture"
model_class_name =  "Rejection based on Logistic Regression Prediction of Rejection"

def train_model(train_x, ite_correctly_predicted, model_class=LogisticRegression, **model_options):
    # Train the model
    model = model_class(**model_options)
    model.fit(train_x, ite_correctly_predicted)
    return model

train_set['ite_mistake'] = train_set.apply(lambda row: 0 if row['ite_pred']==row['ite'] else 1, axis=1)

model = train_model(train_x, train_set['ite_mistake'], LogisticRegression, max_iter=10000, solver='saga', random_state=42)

train_reject_pred = pd.Series(model.predict(train_x), name='train_treated_y_pred')
test_set['test_reject_pred'] = pd.Series(model.predict(test_x), name='test_reject_pred')

test_set['y_reject'] = test_set.apply(lambda row: True if row['test_reject_pred'] else False, axis=1)
test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['y_reject'] else row['ite_pred'], axis=1)
experiment_names.update({experiment_id: f"{architecture} - {model_class.__name__}"})

calculate_all_metrics('ite', 'ite_reject', test_set, file_path, metrics_results, append_metrics_results=True, print=False)

#######################################################################################################################
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
import pandas as pd

# Increment experiment_id
experiment_id += 1

# Set experiment details
architecture = "Dependent architecture"
model_class_name = "Rejection based on GridsearchCV Prediction of Rejection"

# Define a custom scorer (use your own metric)
custom_scorer = make_scorer(f1_score)
# Define a custom scorer using the calculated metric
metrics_dict = calculate_performance_metrics('ite', 'ite_reject', train_set, file_path)
custom_metric_train = metrics_dict['Combined Quality']

custom_scorer = make_scorer(lambda y_true, y_pred: custom_metric_train, greater_is_better=True)


# Define the parameter grid for GridSearchCV
param_grid = {
    'max_iter': [10000],
    'solver': ['saga'],
    'random_state': [42]
}

# Instantiate the model class
model = LogisticRegression()

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, scoring=custom_scorer, cv=5)
grid_search.fit(train_x, train_set['ite_mistake'])

# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Predictions on training and test sets
train_reject_pred = pd.Series(best_model.predict(train_x), name='train_treated_y_pred')
test_set['test_reject_pred'] = pd.Series(best_model.predict(test_x), name='test_reject_pred')

# Create additional columns in the test set
test_set['y_reject'] = test_set.apply(lambda row: True if row['test_reject_pred'] else False, axis=1)
test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['y_reject'] else row['ite_pred'], axis=1)

# Update experiment names
experiment_names.update({experiment_id: f"{architecture} - {model_class_name}"})

# Calculate metrics
calculate_all_metrics('ite', 'ite_reject', test_set, file_path, metrics_results, append_metrics_results=True, print=False)

"""
#######################################################################################################################
# One Class Classification - OCSVM
experiment_id += 1
experiments = [
    {
        'id': experiment_id,
        'architecture': "Separated Rejector",
        'model_class': OneClassSVM,
        'bounds': (0, 2000),
        'key_metric': "Micro Distance (3D ROC)",
        'minmax': 'min',
        'model_options': {'kernel': 'rbf', 'nu': 0.5, }
    },]
experiment_id += 1

experiments +=  [
    {
        'id': experiment_id,
        'architecture': "Separated Rejector",
        'model_class': OneClassSVM,
        'bounds': (0,2000),
        'key_metric': "Combined Quality",
        'minmax': 'max',
        'model_options': {'kernel': 'rbf', 'nu': 0.5, }
    },
]

def run_experiment_one_class_svm(experiment_id, architecture, model_class, bounds, key_metric, minmax, model_options, train_x, test_x, test_set, train_set, file_path, metrics_results, experiment_names):
    with open(file_path, 'a') as file:
        file.write(f"\n\nRunning Experiment {experiment_id} - {architecture} - {model_class.__name__} with optimizing {key_metric}")
    
    experiment_names.update({experiment_id: f"{architecture} - {model_class.__name__} with optimizing {key_metric}"})
    execute_one_class_classification_experiment(train_x, model_class, test_x, bounds, test_set, train_set, file_path, key_metric, minmax, metrics_results, model_options)


# Execute experiments
for experiment in experiments:
    run_experiment_one_class_svm(experiment['id'], experiment['architecture'], experiment['model_class'], 
                   experiment['bounds'], experiment['key_metric'], experiment['minmax'], experiment['model_options'], train_x, test_x, test_set, train_set,
                   file_path, metrics_results, experiment_names, )

#######################################################################################################################
# Rejection based on SCORES MODEL

"""
# Alternatively, some models assign scores that represent the degree of novelty
# of each example (i.e., the higher the more novel), such as LOF (Van der Plas et al.
# 2023) or Neural Networks (Hsu et al. 2020). When dealing with these methods,
# one often initially transforms the scores into novelty probabilities using heuristic
# functions, such as sigmoid and squashing (Vercruyssen et al. 2018), or Gaussian
# Processes (Martens et al. 2023). Then, the rejection threshold can be set to reject
# examples with high novelty probability.
"""

#######################################################################################################################
# ARCHITECTURE TYPE 2: DEPENDENT
# Probabilities symetric upper and under bound
experiment_id += 1
architecture="Dependent architecture"
model_class_name =  "Rejection based on probabilities: symmetric upper & under bound"
key_metric = "Micro Distance (3D ROC)"
minmax = 'min'

experiment_names.update({experiment_id: f"{architecture} - {model_class_name} with optimizing {key_metric}"})
with open(file_path, 'a') as file:
    file.write(f"\n\nRunning Experiment {experiment_id} - {architecture} - {model_class_name} with optimizing {key_metric}")

    # Step 3 Optimize the threshold
result = minimize_scalar(calculate_objective, bounds=(0.5, 1), method='bounded', args=(test_set, file_path, key_metric, minmax), options={'disp': False})
prob_reject_upper_bound = result.x
prob_reject_under_bound = 1 - prob_reject_upper_bound

    # Step 4 Apply rejector to the code
test_set['y_t1_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t1_prob'] < prob_reject_upper_bound else False, axis=1)
test_set['y_t0_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t0_prob'] < prob_reject_upper_bound else False, axis=1)
test_set['y_reject'] = test_set.apply(lambda row: True if row['y_t0_reject_prob'] and row['y_t1_reject_prob'] else False, axis=1)
test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['y_reject_prob'] else row['ite_pred'], axis=1)

    # Step 5 Calculate and report the performance metrics
calculate_all_metrics('ite', 'ite_reject', test_set, file_path, metrics_results, append_metrics_results=True, print=False)


#######################################################################################################################
# Probabilities symetric upper and under bound
experiment_id += 1
architecture="Dependent architecture"
model_class_name =  "Rejection based on probabilities: symmetric upper & under bound"
key_metric = "Combined Quality"
minmax = 'max'

experiment_names.update({experiment_id: f"{architecture} - {model_class_name} with optimizing {key_metric}"})
with open(file_path, 'a') as file:
    file.write(f"\n\nRunning Experiment {experiment_id} - {architecture} - {model_class_name} with optimizing {key_metric}")

    # Step 3 Optimize the threshold
result = minimize_scalar(calculate_objective, bounds=(0.5, 1), method='bounded', args=(test_set, file_path, key_metric, minmax), options={'disp': False})
prob_reject_upper_bound = result.x
prob_reject_under_bound = 1 - prob_reject_upper_bound

    # Step 4 Apply rejector to the code
test_set['y_t1_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t1_prob'] < prob_reject_upper_bound else False, axis=1)
test_set['y_t0_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t0_prob'] < prob_reject_upper_bound else False, axis=1)
test_set['y_reject'] = test_set.apply(lambda row: True if row['y_t0_reject_prob'] and row['y_t1_reject_prob'] else False, axis=1)
test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['y_reject_prob'] else row['ite_pred'], axis=1)

    # Step 5 Calculate and report the performance metrics
calculate_all_metrics('ite', 'ite_reject', test_set, file_path, metrics_results, append_metrics_results=True, print=False)

#######################################################################################################################
# Probabilities asymetric upper and under bound
experiment_id += 1
model_class_name =  "Rejection based on probabilities: asymetric upper & under bound"
key_metric = "Micro Distance (3D ROC)"

experiment_names.update({experiment_id: f"{architecture} - {model_class_name} with optimizing {key_metric}"})
with open(file_path, 'a') as file:
    file.write(f"\n\nRunning Experiment {experiment_id} - {architecture} - {model_class_name} with optimizing {key_metric}")

    # Step 3 Optimize the threshold
initial_guess = [0.45, 0.55]
result = minimize(calculate_objective_threedroc_double_variable, initial_guess, args=(test_set, file_path), bounds=[(0, 0.5), (0.5, 1)])
prob_reject_under_bound, prob_reject_upper_bound = result.x

    # Step 4 Apply rejector to the code
test_set['y_t1_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t1_prob'] < prob_reject_upper_bound else False, axis=1)
test_set['y_t0_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t0_prob'] < prob_reject_upper_bound else False, axis=1)
test_set['y_reject'] = test_set.apply(lambda row: True if row['y_t0_reject_prob'] and row['y_t1_reject_prob'] else False, axis=1)
test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['y_reject'] else row['ite_pred'], axis=1)

    # Step 5 Calculate and report the performance metrics
calculate_all_metrics('ite', 'ite_reject', test_set, file_path, metrics_results, append_metrics_results=True, print=False)

#######################################################################################################################
# Rejection based on probabilities: symetric symmetric upper & under bound by min MISCLASSIFICATION COSTS
experiment_id += 1
model_class_name =  "Rejection based on probabilities: symmetric upper & under bound"
key_metric = "Misclassification Cost"

experiment_names.update({experiment_id: f"{architecture} - {model_class_name} with optimizing {key_metric}"})
with open(file_path, 'a') as file:
    file.write(f"\n\nRunning Experiment {experiment_id} - {architecture} - {model_class_name} with optimizing {key_metric}")

    # Step 3 Optimize the threshold
result = minimize_scalar(calculate_objective_misclassificationcost_single_variable, bounds=(0.5, 1), method='bounded', args=(test_set, file_path), options={'disp': True})
prob_reject_upper_bound = result.x
prob_reject_under_bound = 1 - prob_reject_upper_bound
with open(file_path, 'a') as file:
    file.write(f"\nITE values witht a probability between the optimal underbound {prob_reject_under_bound} and the optimal upperbound {prob_reject_upper_bound} are rejected ")

    # Step 4 Apply rejector to the code
test_set['y_t1_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t1_prob'] < prob_reject_upper_bound else False, axis=1)
test_set['y_t0_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t0_prob'] < prob_reject_upper_bound else False, axis=1)
test_set['y_reject'] = test_set.apply(lambda row: True if row['y_t0_reject_prob'] and row['y_t1_reject_prob'] else False, axis=1)
test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['y_reject_prob'] else row['ite_pred'], axis=1)

    # Step 5 Calculate and report the performance metrics
calculate_all_metrics('ite', 'ite_reject', test_set, file_path, metrics_results, append_metrics_results=True, print=False)

"""
#######################################################################################################################
metrics_results = pd.DataFrame(metrics_results)
improvement_matrix = metrics_results.copy()
for col in improvement_matrix.columns[0:]:
    improvement_matrix[col] = round((improvement_matrix[col] - improvement_matrix[col].iloc[0]) / improvement_matrix[col].iloc[0] * 100, 2)
improvement_matrix = pd.DataFrame(improvement_matrix)



# Chapter 8: Output to file
with open(file_path, 'a') as file:

    file.write("\n\nTable of test_set (First 20 rows)\n")
    file.write(tabulate(test_set.head(20), headers='keys', tablefmt='pretty', showindex=False))
    
    file.write ("\n")
    for exp_number, description in experiment_names.items():
        file.write(f"# Experiment {exp_number}: {description}\n")

    file.write("\nTable of results of the experiments\n")
    file.write(tabulate(metrics_results.T, headers='keys', tablefmt='rounded_grid', showindex=True))
    
    file.write ("\n")
    for exp_number, description in experiment_names.items():
        file.write(f"# Experiment {exp_number}: {description}\n")

    file.write("\nTable of change (%) of each experiment in comparision with the baseline model\n")
    file.write(tabulate(improvement_matrix.T, headers='keys', tablefmt='rounded_grid', showindex=True))

    # Category != category_pred
    file.write("\n\nTable of test_set with wrong classification\n")
    file.write(tabulate(test_set[test_set['category'] != test_set['category_pred']], headers='keys', tablefmt='pretty', showindex=False))




# Select numeric columns excluding the "experiment" column
numeric_columns = metrics_results.select_dtypes(include=['number'])
# numeric_columns = metrics_results.select_dtypes(include=['number']).drop(columns=['Experiment'])

# Create a heatmap
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(numeric_columns, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap of Numeric Columns in DataFrame")
# Add row labels
#heatmap.set_yticklabels(metrics_results['Rejection Type'], rotation=0)
# Save the heatmap as an image file (e.g., PNG)
plt.savefig("output/heatmap.png")
plt.close()
