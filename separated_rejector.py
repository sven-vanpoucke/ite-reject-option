# IMPORTS
# GENERAL
import pandas as pd
from tabulate import tabulate
import numpy as np
from sklearn.metrics import mean_squared_error
# INIT
from models.helper import helper_output
# PREPROCESSING
from datasets.lalonde import processing_get_data_lalonde, processing_transform_data_lalonde
from datasets.twins import preprocessing_get_data_twin, preprocessing_transform_data_twin
from datasets.processing import preprocessing_split_t_c_data
# MODEL T-LEARNER
from models.predictor import predictor_t_model
from sklearn.linear_model import LogisticRegression
# PREDICT 
from models.predictor import predictor_train_predictions, predictor_test_predictions, predictor_ite_predictions
# EVALUATE INDIVIDUAL MODELS
from models.model_evaluator import evaluation_binary
# EVALUATE OVERALL ITE MODEL: PERFORMANCE
from models.evaluator import categorize, categorize_pred, instructions_matrix, calculate_crosstab, calculate_crosstab
# EVALUATE OVERALL ITE MODEL: COSTS
from models.cost import calculate_cost_ite
from models.evaluator import calculate_crosstab_matrix_names
# REJECTION
from models.helper import print_rejection
# REJECTION OOD
from models.rejectors.rejector import distance_test_to_train, is_out_of_distribution, nbrs_train
from models.helper import improvement
# REJECTION OOD - OCSVM
from models.rejectors.ocsvm import train_ocsvm, distance_test_to_train_ocsvm, is_out_of_distribution_ocsvm
# REJECTION PROBABILITIES
from scipy.optimize import minimize_scalar, minimize
from models.rejectors.rejector import calculate_objective_threedroc_double_variable, calculate_objective_threedroc_single_variable, calculate_objective_misclassificationcost_single_variable
# PARAMETERS
folder_path = 'output/dependent/'
dataset = "twins" # Choose out of twins or lalonde
model_class = LogisticRegression # Which two models do we want to generate in the t-models
rejection_architecture = 'dependent' # dependent_rejector or separated_rejector
prob_reject_upper_bound = 0.55
prob_reject_under_bound = 0.45

# INIT
timestamp, file_name, file_path = helper_output(folder_path=folder_path)

# PREPROCESSING
with open(file_path, 'a') as file:
    file.write(f"\nCHAPTER 2: PREPROCESSING\n\n")
    file.write("# This section executes the data retrieval, preprocessing and splitting in a training and dataset.")
    file.write(f"During the whole file, the used dataset is: {dataset}\n\n")

if dataset == "lalonde":
    # for lalaonde
    all_data = processing_get_data_lalonde()
    train_x, test_x, train_y, test_y, train_t, test_t = processing_transform_data_lalonde(all_data)
elif dataset == "twins":
    # for twins
    train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y = preprocessing_get_data_twin()
    train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y = preprocessing_transform_data_twin(train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y)
    #test_ite = test_potential_y["y_t1"]- test_potential_y["y_t0"]
    test_ite = pd.DataFrame({'ite': test_potential_y["y_t1"] - test_potential_y["y_t0"]})

    train_ite = train_potential_y["y_t1"]- train_potential_y["y_t0"]
    # split the data in treated and controlled
    train_treated_x, train_control_x, train_treated_y, train_control_y, test_treated_x, test_control_x, test_treated_y, test_control_y = preprocessing_split_t_c_data(train_x, train_y, train_t, test_x, test_y, test_t)
else:
    pass

# MODEL T-LEARNER
with open(file_path, 'a') as file:
    file.write(f"CHAPTER 3: MODEL TRAINING\n\n")
    file.write("# This section provides details about the model selection, training process, and any hyperparameter tuning.\n")
    file.write(f"The trained ITE model is a T-LEARNER.\n")
    file.write(f"The two individually trained models are: {model_class.__name__}\n\n")

# Training separate models for treated and control groups
treated_model, control_model = predictor_t_model(train_treated_x, train_treated_y, train_control_x, train_control_y, model_class=model_class, max_iter=10000, solver='saga', random_state=42)

# PREDICT
with open(file_path, 'a') as file:
    file.write(f"CHAPTER 4: PREDICT\n\n")
    file.write("# This section applies the trained models to our test_set. \n")
    file.write("# We are able to predict the y_t0, y_t1 and ite \n")

# Training and Testing predictions to evaluate individual models
train_treated_y_pred, train_treated_y_prob, train_control_y_pred, train_control_y_prob = predictor_train_predictions(treated_model, control_model, train_treated_x, train_control_x)
test_treated_y_pred, test_treated_y_prob, test_control_y_pred, test_control_y_prob = predictor_test_predictions(treated_model, control_model, test_treated_x, test_control_x)

# Testing Predictions to evaluate ITE
test_y_t1_pred, test_y_t0_pred, test_y_t1_prob, test_y_t0_prob, test_ite_prob, test_ite_pred = predictor_ite_predictions(treated_model, control_model, test_x)

# EVALUATE INDIVIDUAL MODELS
with open(file_path, 'a') as file:
    file.write(f"CHAPTER 5: EVALUATE INDIVIDUAL LOGISTIC REGRESSION MODELS \n\n")
    file.write("# This section evaluates the individually trained models (two as we used a T-learner). \n")
    file.write(f"The used performance measures are: \n\n")
    file.write(f" - Confusion Matrix \n")
    file.write(f" - Accuracy: overal correctness of the model ((TP + TN) / (TP + TN + FP + FN)) \n")
    file.write(f" - Precision: It measures the accuracy of positive predictions (TP / (TP + FP)) \n")
    file.write(f" - Recall: ability of the model to capture all the relevant cases (TP / (TP + FN)) \n")
    file.write(f" - F1 Score It balances precision and recall, providing a single metric for model evaluation (2 * (Precision * Recall) / (Precision + Recall)) \n")
    file.write(f" - ROC \n\n")

with open(file_path, 'a') as file:
    file.write("Evaluation of the individual models based on the **training data**\n")
evaluation_binary(train_treated_y, train_treated_y_pred, train_treated_y_prob, train_control_y, train_control_y_pred, train_control_y_prob, file_path)

with open(file_path, 'a') as file:
    file.write("\nEvaluation of the individual models based on the **test data**\n")
evaluation_binary(test_treated_y, test_treated_y_pred, test_treated_y_prob, test_control_y, test_control_y_pred, test_control_y_prob, file_path)

# EVALUATE OVERALL ITE MODEL: PERFORMANCE
with open(file_path, 'a') as file:
    file.write(f"CHAPTER 6: EVALUATE OVERALL ITE MODEL: PERFORMANCE \n\n")
    file.write("# This section evaluates the overal performance of the ITE model.\n")
    file.write(f"The used performance measures are: \n\n")
    file.write(f" - Root Mean Squared Error (RMSE) of the ITE \n")
    file.write(f" - Accurate estimate of the ATE \n")
    file.write(f" - Accurancy of ITE\n")

test_set = pd.concat([test_t, test_y_t1_pred, test_y_t1_prob, test_y_t0_pred, test_y_t0_prob, test_ite_pred, test_ite_prob, test_potential_y["y_t0"], test_potential_y["y_t1"], test_ite], axis=1)

# Calculate
rmse = np.sqrt(mean_squared_error(test_set['ite'], test_set['ite_prob'])) # Calculate Root Mean Squared Error (RMSE)
ate_accuracy = np.abs(test_set['ite_pred'].mean() - test_set['ite'].mean()) # Evaluate ATE accuracy

accurancy, rr, micro_tpr, micro_fpr, macro_tpr, macro_fpr, micro_distance_threedroc, macro_distance_threedroc = calculate_crosstab('ite', 'ite_pred', test_set, file_path)

# Log results
with open(file_path, 'a') as file:
    file.write(f"\n\nRoot Mean Squared Error (RMSE) between the ite and ite_prob: {rmse.round(4)}\n\n")
    file.write(f"The Actual Average Treatment Effect (ATE): {test_set['ite'].mean().round(4)}\n")
    file.write(f"The Predicted Average Treatment Effect (ATE): {test_set['ite_pred'].mean().round(4)}\n")
    file.write(f"Accuracy of Average Treatment Effect (ATE): {ate_accuracy.round(4)}\n")

# EVALUATE OVERALL ITE MODEL: COST
with open(file_path, 'a') as file:
    file.write(f"\n\nCHAPTER 7: EVALUATE OVERALL ITE MODEL: COST \n\n")
    file.write("# This section evaluates the overal misclassification costs of the ITE model.\n")

# Apply the categorization function to create the 'Category' column
test_set['category'] = test_set.apply(categorize, axis=1)
test_set['category_pred'] = test_set.apply(categorize_pred, axis=1)

instructions_matrix(file_path)
calculate_crosstab_matrix_names('y_t0', 'y_t1', test_set, file_path)
calculate_crosstab_matrix_names('y_t0_pred', 'y_t1_pred', test_set, file_path)

# Apply the calculate_cost function to each row in the DataFrame
test_set['cost_ite'] = test_set.apply(calculate_cost_ite, axis=1)

# Calculate total misclassification cost
total_cost_ite = test_set['cost_ite'].sum()
with open(file_path, 'a') as file:
    file.write(f"\nTotal Misclassification Cost: {total_cost_ite}\n")

# CHAPTER 8: REJECTION
with open(file_path, 'a') as file:
    file.write(f"\nCHAPTER 8: REJECTION \n\n")
    file.write("# This section executes and reports metrics for ITE models with rejection.\n")
    file.write("# Every indicated change are in comparision to the base ITE model without rejection.\n")
    file.write(f"\nARCHITECTURE TYPE 0: NO REJECTION -- BASELINE MODEL\n")

# ARCHITECTURE TYPE 1: NO REJECTION
test_set['ite_reject'] = test_set.apply(lambda row: row['ite_pred'], axis=1)
print_rejection(file_path, test_set, total_cost_ite, accurancy, micro_distance_threedroc, macro_distance_threedroc)

# ARCHITECTURE TYPE 1: SEPARATED
with open(file_path, 'a') as file:
    file.write(f"\nARCHITECTURE TYPE 1: SEPARATED\n")

# REJECTION OOD
with open(file_path, 'a') as file:
    file.write(f"\nREJECTION TYPE 1A: OUT OF DISRIBUTION\n")

model = nbrs_train(train_x)
d = distance_test_to_train(model, test_x)
test_set['ood'] = d.apply(is_out_of_distribution, threshold_distance=6)
test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['ood'] else row['ite_pred'], axis=1)

print_rejection(file_path, test_set, total_cost_ite, accurancy, micro_distance_threedroc, macro_distance_threedroc)



# REJECTION ONE CLASS CLASSIFICATION MODEL
# Generally, they enclose the dataset into a specific surface and
# flag any example that falls outside such region as novelty. For instance, a typical
# approach is to use a One-Class Support Vector Machine (OCSVM) to encapsulate the training data through a hypersphere (Coenen et al. 2020; Homenda et al.
# 2014). By adjusting the size of the hypersphere, the

with open(file_path, 'a') as file:
    file.write(f"\nREJECTION TYPE 1B: ONE CLASS CLASSIFICATION MODEL\n")

model = nbrs_train(train_x)
d = distance_test_to_train(model, test_x)
test_set['ood'] = d.apply(is_out_of_distribution, threshold_distance=6)
test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['ood'] else row['ite_pred'], axis=1)

print_rejection(file_path, test_set, total_cost_ite, accurancy, micro_distance_threedroc, macro_distance_threedroc)

# REJECTION ONE CLASS CLASSIFICATION MODEL
# Generally, they enclose the dataset into a specific surface and
# flag any example that falls outside such region as novelty. For instance, a typical
# approach is to use a One-Class Support Vector Machine (OCSVM) to encapsulate the training data through a hypersphere (Coenen et al. 2020; Homenda et al.
# 2014). By adjusting the size of the hypersphere, the proportion of non-rejected
# examples can be increased (Wu et al. 2007)

# Rejection using OCSVM
with open(file_path, 'a') as file:
    file.write(f"\nREJECTION TYPE 1B: ONE CLASS CLASSIFICATION MODEL using OCSVM\n")

# Assuming train_x and test_x are your training and test data, replace with actual data
model_ocsvm = train_ocsvm(train_x)
distances_ocsvm = distance_test_to_train_ocsvm(model_ocsvm, test_x)

# Assuming you have a threshold_distance defined
threshold_distance = 4
test_set['ood'] = distances_ocsvm.apply(is_out_of_distribution_ocsvm, threshold=threshold_distance)
test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['ood'] else row['ite_pred'], axis=1)

# Assuming you have functions like 'print_rejection' defined
print_rejection(file_path, test_set, total_cost_ite, accurancy, micro_distance_threedroc, macro_distance_threedroc)

# REJECTION SCORES MODEL
# Alternatively, some models assign scores that represent the degree of novelty
# of each example (i.e., the higher the more novel), such as LOF (Van der Plas et al.
# 2023) or Neural Networks (Hsu et al. 2020). When dealing with these methods,
# one often initially transforms the scores into novelty probabilities using heuristic
# functions, such as sigmoid and squashing (Vercruyssen et al. 2018), or Gaussian
# Processes (Martens et al. 2023). Then, the rejection threshold can be set to reject
# examples with high novelty probability.

with open(file_path, 'a') as file:
    file.write(f"\nREJECTION TYPE 1C: SCORE MODEL\n")
    file.write(f"\n - Not done yet\n")

model = nbrs_train(train_x)
d = distance_test_to_train(model, test_x)
test_set['ood'] = d.apply(is_out_of_distribution, threshold_distance=6)
test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['ood'] else row['ite_pred'], axis=1)

print_rejection(file_path, test_set, total_cost_ite, accurancy, micro_distance_threedroc, macro_distance_threedroc)



# ARCHITECTURE TYPE 2: DEPENDENT
with open(file_path, 'a') as file:
    file.write(f"\nARCHITECTURE TYPE 2: DEPENDENT\n")
    file.write(f"\nREJECTION TYPE 2A: REJECTION BASED ON PROBABILITIES BY MINIMIZING 3DROC \n")
    file.write(f"\nVARIANT TYPE 2A I: OPTIMIZATION OF SINGLE BOUNDARIES BY MINIMIZING 3DROC \n")

# Run the optimization
result = minimize_scalar(calculate_objective_threedroc_single_variable, bounds=(0.5, 1), method='bounded', args=(test_set, file_path), options={'disp': True})
# Get the optimal value
prob_reject_upper_bound = result.x
prob_reject_under_bound = 1 - prob_reject_upper_bound

# Use the optimal value in your code
test_set['y_t1_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t1_prob'] < prob_reject_upper_bound else False, axis=1)
test_set['y_t0_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t0_prob'] < prob_reject_upper_bound else False, axis=1)
test_set['y_reject'] = test_set.apply(lambda row: True if row['y_t0_reject_prob'] and row['y_t1_reject_prob'] else False, axis=1)
test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['y_reject_prob'] else row['ite_pred'], axis=1)

with open(file_path, 'a') as file:
    file.write(f"\nITE values witht a probability between the optimal underbound {prob_reject_under_bound} and the optimal upperbound {prob_reject_upper_bound} are rejected ")
print_rejection(file_path, test_set, total_cost_ite, accurancy, micro_distance_threedroc, macro_distance_threedroc)

## DOUBLE VARIABLE OPTIMIZATION
with open(file_path, 'a') as file:
    file.write(f"\nVARIANT TYPE 2A II: OPTIMIZATION OF DOUBLE BOUNDARIES BY MINIMIZING 3DROC  \n")

# Optimization using minimize function (Multiple variables)
initial_guess = [0.45, 0.55]
result = minimize(calculate_objective_threedroc_double_variable, initial_guess, args=(test_set, file_path), bounds=[(0, 0.5), (0.5, 1)])
# Get the optimal value
prob_reject_under_bound, prob_reject_upper_bound = result.x

# Use the optimal value in your code
test_set['y_t1_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t1_prob'] < prob_reject_upper_bound else False, axis=1)
test_set['y_t0_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t0_prob'] < prob_reject_upper_bound else False, axis=1)
test_set['y_reject'] = test_set.apply(lambda row: True if row['y_t0_reject_prob'] and row['y_t1_reject_prob'] else False, axis=1)
test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['y_reject'] else row['ite_pred'], axis=1)

with open(file_path, 'a') as file:
    file.write(f"\nITE values witht a probability between the optimal underbound {prob_reject_under_bound} and the optimal upperbound {prob_reject_upper_bound} are rejected ")
print_rejection(file_path, test_set, total_cost_ite, accurancy, micro_distance_threedroc, macro_distance_threedroc)

# REJECTION TYPE 2A: REJECTION BASED ON PROBABILITIES BY MINIMIZING MISCLASSIFICATION COSTS
with open(file_path, 'a') as file:
    file.write(f"\nREJECTION TYPE 2A: REJECTION BASED ON PROBABILITIES BY MINIMIZING MISCLASSIFICATION COSTS \n")

# Run the optimization
result = minimize_scalar(calculate_objective_misclassificationcost_single_variable, bounds=(0.5, 1), method='bounded', args=(test_set, file_path), options={'disp': True})
# Get the optimal value
prob_reject_upper_bound = result.x
prob_reject_under_bound = 1 - prob_reject_upper_bound

# Use the optimal value in your code
test_set['y_t1_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t1_prob'] < prob_reject_upper_bound else False, axis=1)
test_set['y_t0_reject_prob'] = test_set.apply(lambda row: True if prob_reject_under_bound < row['y_t0_prob'] < prob_reject_upper_bound else False, axis=1)
test_set['y_reject'] = test_set.apply(lambda row: True if row['y_t0_reject_prob'] and row['y_t1_reject_prob'] else False, axis=1)
test_set['ite_reject'] = test_set.apply(lambda row: "R" if row['y_reject_prob'] else row['ite_pred'], axis=1)

with open(file_path, 'a') as file:
    file.write(f"\nITE values witht a probability between the optimal underbound {prob_reject_under_bound} and the optimal upperbound {prob_reject_upper_bound} are rejected ")
print_rejection(file_path, test_set, total_cost_ite, accurancy, micro_distance_threedroc, macro_distance_threedroc)




with open(file_path, 'a') as file:
    file.write("\n\nTable of test_set (First 20 rows)\n")
    file.write(tabulate(test_set.head(20), headers='keys', tablefmt='pretty', showindex=False))