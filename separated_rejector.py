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
from models.helper import print_rejection
from models.evaluators.evaluator import calculate_all_metrics
## REJECTION OOD
from models.rejectors.rejector import distance_test_to_train, is_out_of_distribution, nbrs_train
from models.helper import improvement
## REJECTION OOD - OCSVM
from models.rejectors.ocsvm import train_ocsvm, distance_test_to_train_ocsvm, is_out_of_distribution_ocsvm, calculate_objective_threedroc_threshold
## REJECTION PROBABILITIES
from scipy.optimize import minimize_scalar, minimize
from models.rejectors.rejector import calculate_objective_threedroc_double_variable, calculate_objective_threedroc_single_variable, calculate_objective_misclassificationcost_single_variable

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
metrics_results = {
    # 'Experiment': [],
    # 'Architecture Type': [],
    # 'Rejection Type': [],
    # 'Accuracy': [],
    # 'Rejection Rate': [],
    # 'Micro TPR': [],
    # 'Micro FPR': [],
    # 'Macro TPR': [],
    # 'Macro FPR': [],
}

def append_result(experiment, architecture_type, rejection_type, metric1, metric2, metric3, metric4, metric5, metric6):
    metrics_results['Experiment'].append(experiment)
    metrics_results['Architecture Type'].append(architecture_type)
    metrics_results['Rejection Type'].append(rejection_type)
    metrics_results['Accuracy'].append(round(metric1, 4))
    metrics_results['Rejection Rate'].append(round(metric2, 4))
    metrics_results['Micro TPR'].append(round(metric3, 4))
    metrics_results['Micro FPR'].append(round(metric4, 4))
    metrics_results['Macro TPR'].append(round(metric5, 4))
    metrics_results['Macro FPR'].append(round(metric6, 4))

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
    file.write("# Every indicated change are in comparision to the base ITE model without rejection.\n")
    file.write(f"\nARCHITECTURE TYPE 0: NO REJECTION -- BASELINE MODEL\n")
    
## CHAPTER 7B: Baseline Model - No Rejection // Experiment 0
    
    # Step 1 Set variables
exp_number = 0
arch_type = "No Rejection"
rej_type = "No Rejection"

    # Step 2 Train the Rejector

    # Step 3 Optimize the threshold

    # Step 4 Apply rejector to the code
test_set['ite_reject'] = test_set.apply(lambda row: row['ite_pred'], axis=1)

    # Step 5 Calculate the performance metrics
#metrics_dict = calculate_performance_metrics('ite', 'ite_reject', test_set, file_path)
# accurancy_2, rr_2, micro_tpr_2, micro_fpr_2, macro_tpr_2, macro_fpr_2, micro_distance_threedroc_2, macro_distance_threedroc_2, accurancy_rejection_2, coverage_rejection_2, prediction_quality_2, rejection_quality_2, combined_quality_2 = calculate_performance_metrics('ite', 'ite_reject', test_set, file_path, print = True)
# append_result(exp_number, arch_type, rej_type, accurancy_2, rr_2, micro_tpr_2, micro_fpr_2, macro_tpr_2, macro_fpr_2)
# print_rejection(file_path, test_set, total_cost_ite, accurancy, micro_distance_threedroc, macro_distance_threedroc)

calculate_all_metrics('ite', 'ite_reject', test_set, file_path, metrics_results, append_metrics_results=True, print=False)

"""
with open(file_path, 'a') as file:
    file.write(f"\n\nRoot Mean Squared Error (RMSE) between the ite and ite_prob: {rmse.round(4)}\n\n")
    file.write(f"The Actual Average Treatment Effect (ATE): {test_set['ite'].mean().round(4)}\n")
    file.write(f"The Predicted Average Treatment Effect (ATE): {test_set['ite_pred'].mean().round(4)}\n")
    file.write(f"Accuracy of Average Treatment Effect (ATE): {ate_accuracy.round(4)}\n")
    file.write(f"\nTotal Misclassification Cost: {total_cost_ite}\n")
"""

#######################################################################################################################
## CHAPTER 7C: Architecture Type = Separated
with open(file_path, 'a') as file:
    file.write(f"\nARCHITECTURE TYPE 1: SEPARATED\n")

arch_type = "separated"

### Rejection based on Out Of Distribution Detecten - OOD
#### OOD: K-Nearest Neighbors // Experiment 1

    # Step 1 Set variables
rej_type =  "OOD - KNN"
exp_number += 1

with open(file_path, 'a') as file:
    file.write(f"\nREJECTION TYPE 1A: OUT OF DISRIBUTION\n")

    # Step 2 Train the Rejector
model = nbrs_train(train_x)
d = distance_test_to_train(model, test_x)

    # Step 3 Optimize the threshold
# We don't optimize yet, we use a fixed threshold

    # Step 4 Apply rejector to the code
test_set['ood'] = d.apply(is_out_of_distribution, threshold_distance=6)
test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['ood'] else row['ite_pred'], axis=1)

    # Step 5 Calculate and report the performance metrics
# accurancy_2, rr_2, micro_tpr_2, micro_fpr_2, macro_tpr_2, macro_fpr_2, micro_distance_threedroc_2, macro_distance_threedroc_2, accurancy_rejection_2, coverage_rejection_2, prediction_quality_2, rejection_quality_2, combined_quality_2 = calculate_performance_metrics('ite', 'ite_reject', test_set, file_path, print = True)
# append_result(exp_number, arch_type, rej_type, accurancy_2, rr_2, micro_tpr_2, micro_fpr_2, macro_tpr_2, macro_fpr_2)
# print_rejection(file_path, test_set, total_cost_ite, accurancy, micro_distance_threedroc, macro_distance_threedroc)

calculate_all_metrics('ite', 'ite_reject', test_set, file_path, metrics_results, append_metrics_results=True, print=False)

### Rejection based on One Class Classification Model
"""
# Generally, they enclose the dataset into a specific surface and
# flag any example that falls outside such region as novelty. For instance, a typical
# approach is to use a One-Class Support Vector Machine (OCSVM) to encapsulate the training data through a hypersphere (Coenen et al. 2020; Homenda et al.
# 2014). By adjusting the size of the hypersphere, the proportion of non-rejected
# examples can be increased (Wu et al. 2007)
"""
#### One Class Classification - OCSVM // Experiment 2

    # Step 1 Set variables
rej_type =  "OCSVM"
exp_number += 1

with open(file_path, 'a') as file:
    file.write(f"\nREJECTION TYPE 1B: ONE CLASS CLASSIFICATION MODEL using OCSVM\n")

    # Step 2 Train the Rejector
model_ocsvm = train_ocsvm(train_x)
distances_ocsvm = distance_test_to_train_ocsvm(model_ocsvm, test_x)

    # Step 3 Optimize the threshold
result = minimize_scalar(calculate_objective_threedroc_threshold, bounds=(0, 40), method='bounded', args=(test_set, file_path, distances_ocsvm), options={'disp': True})
threshold_distance = result.x
with open(file_path, 'a') as file:
    file.write(f"\nThe best threshold is {threshold_distance}\n")

    # Step 4 Apply rejector to the code
test_set['ood'] = distances_ocsvm.apply(is_out_of_distribution_ocsvm, threshold=threshold_distance)
test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['ood'] else row['ite_pred'], axis=1)

    # Step 5 Calculate and report the performance metrics
# accurancy_2, rr_2, micro_tpr_2, micro_fpr_2, macro_tpr_2, macro_fpr_2, micro_distance_threedroc_2, macro_distance_threedroc_2, accurancy_rejection_2, coverage_rejection_2, prediction_quality_2, rejection_quality_2, combined_quality_2 = calculate_performance_metrics('ite', 'ite_reject', test_set, file_path, print = True)
# append_result(exp_number, arch_type, rej_type, accurancy_2, rr_2, micro_tpr_2, micro_fpr_2, macro_tpr_2, macro_fpr_2)
# print_rejection(file_path, test_set, total_cost_ite, accurancy, micro_distance_threedroc, macro_distance_threedroc)

calculate_all_metrics('ite', 'ite_reject', test_set, file_path, metrics_results, append_metrics_results=True, print=False)

### Rejection based on SCORES MODEL
"""
# Alternatively, some models assign scores that represent the degree of novelty
# of each example (i.e., the higher the more novel), such as LOF (Van der Plas et al.
# 2023) or Neural Networks (Hsu et al. 2020). When dealing with these methods,
# one often initially transforms the scores into novelty probabilities using heuristic
# functions, such as sigmoid and squashing (Vercruyssen et al. 2018), or Gaussian
# Processes (Martens et al. 2023). Then, the rejection threshold can be set to reject
# examples with high novelty probability.
"""
### REJECTION SCORES MODEL // Experiment 3

    # Step 1 Set variables
exp_number += 1
rej_type =  "Scores Model"

with open(file_path, 'a') as file:
    file.write(f"\nREJECTION TYPE 1C: SCORE MODEL\n")
    file.write(f"\n - Not done yet\n")

    # Step 2 Train the Rejector


    # Step 3 Optimize the threshold


    # Step 4 Apply rejector to the code
test_set['ite_reject'] = test_set.apply(lambda row: row['ite_pred'], axis=1)

    # Step 5 Calculate and report the performance metrics
# accurancy_2, rr_2, micro_tpr_2, micro_fpr_2, macro_tpr_2, macro_fpr_2, micro_distance_threedroc_2, macro_distance_threedroc_2, accurancy_rejection_2, coverage_rejection_2, prediction_quality_2, rejection_quality_2, combined_quality_2 = calculate_performance_metrics('ite', 'ite_reject', test_set, file_path, print = True)
# append_result(exp_number, arch_type, rej_type, accurancy_2, rr_2, micro_tpr_2, micro_fpr_2, macro_tpr_2, macro_fpr_2)
# print_rejection(file_path, test_set, total_cost_ite, accurancy, micro_distance_threedroc, macro_distance_threedroc)

#######################################################################################################################

# ARCHITECTURE TYPE 2: DEPENDENT
arch_type = "dependent"

with open(file_path, 'a') as file:
    file.write(f"\nARCHITECTURE TYPE 2: DEPENDENT\n")
    file.write(f"\nREJECTION TYPE 2A: REJECTION BASED ON PROBABILITIES BY MINIMIZING 3DROC \n")


# Probabilities symetric upper and under bound // Experiment 3
    
    # Step 1 Set variables
rej_type =  "prob symetric bounds"
exp_number += 1

with open(file_path, 'a') as file:
    file.write(f"\nVARIANT TYPE 2A I: OPTIMIZATION OF SINGLE BOUNDARIES BY MINIMIZING 3DROC \n")

    # Step 2 Define  the threshold

    # Step 3 Optimize the threshold
result = minimize_scalar(calculate_objective_threedroc_single_variable, bounds=(0.5, 1), method='bounded', args=(test_set, file_path), options={'disp': True})
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
# accurancy_2, rr_2, micro_tpr_2, micro_fpr_2, macro_tpr_2, macro_fpr_2, micro_distance_threedroc_2, macro_distance_threedroc_2, accurancy_rejection_2, coverage_rejection_2, prediction_quality_2, rejection_quality_2, combined_quality_2 = calculate_performance_metrics('ite', 'ite_reject', test_set, file_path, print = True)
# append_result(exp_number, arch_type, rej_type, accurancy_2, rr_2, micro_tpr_2, micro_fpr_2, macro_tpr_2, macro_fpr_2)
# print_rejection(file_path, test_set, total_cost_ite, accurancy, micro_distance_threedroc, macro_distance_threedroc)
calculate_all_metrics('ite', 'ite_reject', test_set, file_path, metrics_results, append_metrics_results=True, print=False)

# Probabilities asymetric upper and under bound // Experiment 4

    # Step 1 Set variables
rej_type =  "prob asymetric bounds"
exp_number += 1
with open(file_path, 'a') as file:
    file.write(f"\nVARIANT TYPE 2A II: OPTIMIZATION OF DOUBLE BOUNDARIES BY MINIMIZING 3DROC  \n")

    # Step 2 Define  the threshold

    # Step 3 Optimize the threshold
initial_guess = [0.45, 0.55]
result = minimize(calculate_objective_threedroc_double_variable, initial_guess, args=(test_set, file_path), bounds=[(0, 0.5), (0.5, 1)])
prob_reject_under_bound, prob_reject_upper_bound = result.x
with open(file_path, 'a') as file:
    file.write(f"\nITE values witht a probability between the optimal underbound {prob_reject_under_bound} and the optimal upperbound {prob_reject_upper_bound} are rejected ")

    # Step 4 Apply rejector to the code
test_set['y_t1_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t1_prob'] < prob_reject_upper_bound else False, axis=1)
test_set['y_t0_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t0_prob'] < prob_reject_upper_bound else False, axis=1)
test_set['y_reject'] = test_set.apply(lambda row: True if row['y_t0_reject_prob'] and row['y_t1_reject_prob'] else False, axis=1)
test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['y_reject'] else row['ite_pred'], axis=1)

    # Step 5 Calculate and report the performance metrics
# accurancy_2, rr_2, micro_tpr_2, micro_fpr_2, macro_tpr_2, macro_fpr_2, micro_distance_threedroc_2, macro_distance_threedroc_2, accurancy_rejection_2, coverage_rejection_2, prediction_quality_2, rejection_quality_2, combined_quality_2 = calculate_performance_metrics('ite', 'ite_reject', test_set, file_path, print = True)
# append_result(exp_number, arch_type, rej_type, accurancy_2, rr_2, micro_tpr_2, micro_fpr_2, macro_tpr_2, macro_fpr_2)
# print_rejection(file_path, test_set, total_cost_ite, accurancy, micro_distance_threedroc, macro_distance_threedroc)
calculate_all_metrics('ite', 'ite_reject', test_set, file_path, metrics_results, append_metrics_results=True, print=False)


# REJECTION TYPE 2A: REJECTION BASED ON PROBABILITIES BY MINIMIZING MISCLASSIFICATION COSTS // Experiment 5
    # Step 1 Set variables
rej_type =  "prob miscost"
exp_number += 1
with open(file_path, 'a') as file:
    file.write(f"\nREJECTION TYPE 2A: REJECTION BASED ON PROBABILITIES BY MINIMIZING MISCLASSIFICATION COSTS \n")

    # Step 2 Define  the threshold

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
# accurancy_2, rr_2, micro_tpr_2, micro_fpr_2, macro_tpr_2, macro_fpr_2, micro_distance_threedroc_2, macro_distance_threedroc_2, accurancy_rejection_2, coverage_rejection_2, prediction_quality_2, rejection_quality_2, combined_quality_2 = calculate_performance_metrics('ite', 'ite_reject', test_set, file_path, print = True)
# append_result(exp_number, arch_type, rej_type, accurancy_2, rr_2, micro_tpr_2, micro_fpr_2, macro_tpr_2, macro_fpr_2)
# print_rejection(file_path, test_set, total_cost_ite, accurancy, micro_distance_threedroc, macro_distance_threedroc)
calculate_all_metrics('ite', 'ite_reject', test_set, file_path, metrics_results, append_metrics_results=True, print=False)

metrics_results = pd.DataFrame(metrics_results)
improvement_matrix = metrics_results.copy()
for col in improvement_matrix.columns[0:]:
    improvement_matrix[col] = (improvement_matrix[col] - improvement_matrix[col].iloc[0]) / improvement_matrix[col].iloc[0] * 100
improvement_matrix = pd.DataFrame(improvement_matrix)

experiments = {
    0: "No Rejector: Baseline Model",
    1: "Separated Rejector - O.O.D.: K-Nearest Neighbors",
    2: "Separated Rejector - One Class Classification: OCSVM",
    3: "Dependent Rejector - Rejection based on probabilities: symetric symmetric upper & under bound (minimization of 3D ROC)",
    4: "Dependent Rejector - Rejection based on probabilities: asymmetric upper & under bound (minimization of 3D ROC)",
    5: "Dependent Rejector - Rejection based on probabilities: symmetric upper & under bound (minimization of misclassification costs)"
}

# Chapter 8: Output to file
with open(file_path, 'a') as file:

    file.write("\n\nTable of test_set (First 20 rows)\n")
    file.write(tabulate(test_set.head(20), headers='keys', tablefmt='pretty', showindex=False))
    
    file.write ("\n")
    for exp_number, description in experiments.items():
        file.write(f"# Experiment {exp_number}: {description}\n")

    file.write("\nTable of results of the experiments\n")
    file.write(tabulate(metrics_results.T, headers='keys', tablefmt='rounded_grid', showindex=True))
    
    file.write ("\n")
    for exp_number, description in experiments.items():
        file.write(f"# Experiment {exp_number}: {description}\n")

    file.write("\nTable of change (%) of each experiment in comparision with the baseline model\n")
    file.write(tabulate(improvement_matrix.T, headers='keys', tablefmt='rounded_grid', showindex=True))

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
