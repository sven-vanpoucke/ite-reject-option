"""
Table of contents:

"""

# Chapter 0: Imports

## General Packages
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
from datasets.ihdp2 import preprocessing_get_data_ihdp, preprocessing_transform_data_ihdp
from datasets.propensity_score import calculate_propensity_scores, knn_matching

## MODEL T-LEARNER
from models.predictor import predictor_t_model
from sklearn.linear_model import LogisticRegression, LinearRegression
from models.predictor import predictor_train_predictions, predictor_test_predictions, predictor_ite_predictions

# Evaluate
from models.evaluators.cost_evaluator import categorize
from models.evaluators.evaluator import calculate_all_metrics
from models.evaluators.evaluator import calculate_performance_metrics

# Graphs
from models.rejectors.helper import onelinegraph, twolinegraph

# Rejection
from models.rejectors.helper import train_model, novelty_rejection
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import csv

# Chapter 1: Initialization

## Parameters
### To choose
dataset = "TWINSC" # Choose out of TWINS or TWINSC (if you want TWINS to be treated as continuous instead of classification) or LALONDE or IHDP
psm = False
### Rejection
detail_factor = 1 # 1 (no extra detail) or 10 (extra detail)
max_rr = 15 # (number between 1 and 49)
x_scaling = True # True or False

### Not to choose
folder_path = 'output/'
text_folder_path = 'output/text/'
timestamp, file_name, file_path = helper_output(dataset, folder_path=text_folder_path)
metrics_results = {}

### Outdated
rejection_architecture = 'dependent' # dependent_rejector or separated_rejector
prob_reject_upper_bound = 0.55
prob_reject_under_bound = 0.45

# Chapter 2: Preprocessing

## Output to file
with open(file_path, 'a') as file:
    file.write(f"\Chapter 2: Preprocessing\n\n")
    file.write("# This section executes the data retrieval, preprocessing and splitting in a training and dataset.")
    file.write(f"During the whole file, the used dataset is: {dataset}\n\n")

## Retrieval of data
if dataset == "LALONDE":
    all_data = processing_get_data_lalonde()
    train_x, test_x, train_y, test_y, train_t, test_t = processing_transform_data_lalonde(all_data)
elif dataset == "TWINS" or dataset == "TWINSC":
    # Set the model class for the T-learner
    model_class = LogisticRegression # Which two models do we want to generate in the t-models
    model_params = {"max_iter": 10000, "solver": "saga", "random_state": 42}
    train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y = preprocessing_get_data_twin()
    train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y = preprocessing_transform_data_twin(train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y)
elif dataset == "TWINS":
    # Set the model class for the T-learner
    model_class = LogisticRegression # Which two models do we want to generate in the t-models
    model_params = {"max_iter": 10000, "solver": "saga", "random_state": 42}
    train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y = preprocessing_get_data_twin()
    train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y = preprocessing_transform_data_twin(train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y)
elif dataset == "IHDP":
    # Set the model class for the T-learner
    model_class = LinearRegression # Which two models do we want to generate in the t-models
    model_params = {"fit_intercept": True}
    # Step 1: Load and preprocess IHDP data
    train_x, train_t, train_y, train_potential_y, test_x, test_t, test_y, test_potential_y = preprocessing_get_data_ihdp()
    # Step 2: Transform the data
    train_x, train_t, train_y, train_potential_y, test_x, test_t, test_y, test_potential_y = preprocessing_transform_data_ihdp(train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y)

## Editing of the data
train_treated_x, train_control_x, train_treated_y, train_control_y, test_treated_x, test_control_x, test_treated_y, test_control_y = preprocessing_split_t_c_data(train_x, train_y, train_t, test_x, test_y, test_t)
test_ite = pd.DataFrame({'ite': test_potential_y["y_t1"] - test_potential_y["y_t0"]})
train_ite = pd.DataFrame({'ite': train_potential_y["y_t1"] - train_potential_y["y_t0"]})

# Chapter 3: Training of the ITE Model
## Output to file
with open(file_path, 'a') as file:
    file.write(f"CHAPTER 3: Training of the ITE Model\n\n")
    file.write("# This section provides details about the model selection, training process, and any hyperparameter tuning.\n")
    file.write(f"The trained ITE model is a T-LEARNER.\n")
    file.write(f"The two individually trained models are: {model_class.__name__}\n\n")

## Training of the ITE Model (T-learner: This model is trained on the treated and control groups separately)
treated_model, control_model = predictor_t_model(train_treated_x, train_treated_y, train_control_x, train_control_y, model_class, model_params)

# Chapter 4: Predictions
## Training and Testing predictions to evaluate individual models
train_treated_y_pred, train_treated_y_prob, train_control_y_pred, train_control_y_prob = predictor_train_predictions(treated_model, control_model, train_treated_x, train_control_x)
test_treated_y_pred, test_treated_y_prob, test_control_y_pred, test_control_y_prob = predictor_test_predictions(treated_model, control_model, test_treated_x, test_control_x)
## Testing Predictions to evaluate ITE
train_y_t1_pred, train_y_t0_pred, train_y_t1_prob, train_y_t0_prob, train_ite_prob, train_ite_pred = predictor_ite_predictions(treated_model, control_model, train_x)
test_y_t1_pred, test_y_t0_pred, test_y_t1_prob, test_y_t0_prob, test_ite_prob, test_ite_pred = predictor_ite_predictions(treated_model, control_model, test_x)

## Merge the different dataframes
if train_treated_y_prob is not None and not train_treated_y_prob.isna().all():
    test_set = pd.concat([test_t, test_y_t1_pred, test_y_t1_prob, test_y_t0_pred, test_y_t0_prob, test_ite_pred, test_ite_prob, test_potential_y["y_t0"], test_potential_y["y_t1"], test_ite], axis=1)
    train_set = pd.concat([test_t, train_y_t1_pred, train_y_t1_prob, train_y_t0_pred, train_y_t0_prob, train_ite_pred, train_ite_prob, train_potential_y["y_t0"], train_potential_y["y_t1"], train_ite], axis=1)
else:
    test_set = pd.concat([test_t, test_y_t1_pred, test_y_t0_pred, test_ite_pred, test_potential_y["y_t0"], test_potential_y["y_t1"], test_ite], axis=1)
    train_set = pd.concat([test_t, train_y_t1_pred, train_y_t0_pred, train_ite_pred, train_potential_y["y_t0"], train_potential_y["y_t1"], train_ite], axis=1)

## Make TWINS (binary) to TWINSC (continuous)
if dataset == "TWINSC":
    # Delete columns y_t1_pred and y_t0_pred, ite_pred
    test_set = test_set.drop(['y_t1_pred', 'y_t0_pred', 'ite_pred'], axis=1)
    # Rename columns y_t1_prob, y_t0_prob, ite_prob to y_t1_pred, y_t0_pred, ite_pred
    test_set = test_set.rename(columns={'y_t1_prob': 'y_t1_pred', 'y_t0_prob': 'y_t0_pred', 'ite_prob': 'ite_pred'})

    # Delete columns y_t1_pred and y_t0_pred, ite_pred
    train_set = train_set.drop(['y_t1_pred', 'y_t0_pred', 'ite_pred'], axis=1)
    # Rename columns y_t1_prob, y_t0_prob, ite_prob to y_t1_pred, y_t0_pred, ite_pred
    train_set = train_set.rename(columns={'y_t1_prob': 'y_t1_pred', 'y_t0_prob': 'y_t0_pred', 'ite_prob': 'ite_pred'})


# Chapter 6: Evaluate overall ITE Model: Costs
## Apply the categorization function to create the 'Category' column
test_set['category'] = test_set.apply(categorize, axis=1, is_pred=False)
test_set['category_pred'] = test_set.apply(categorize, axis=1)
test_set['category_rej'] = test_set.apply(categorize, axis=1)
test_set['ite_mistake'] = test_set.apply(lambda row: 0 if row['ite_pred']==row['ite'] else 1, axis=1)

#######################################################################################################################

# CHAPTER 7: REJECTION
## Output to file
with open(file_path, 'a') as file:
    file.write(f"\nCHAPTER 7: REJECTION \n\n")
    file.write("# This section executes and reports metrics for ITE models with rejection.\n")

## Merge the test_set with the train_set !!
x = pd.concat([train_x, test_x], ignore_index=True).copy()
all_data = pd.concat([train_set, test_set], ignore_index=True).copy()

#######################################################################################################################
# Baseline Model - No Rejection // Experiment 0
experiment_id = 0
experiment_names = {}
experiment_name = "No Rejector - Baseline Model"
experiment_names.update({experiment_id: f"{experiment_name}"})

# Step 4 Apply rejector to the code
all_data['ite_reject'] = all_data.apply(lambda row: row['ite_pred'], axis=1)

# Step 5 Calculate the performance metrics
metrics_dict = calculate_all_metrics('ite', 'ite_reject', all_data, file_path, metrics_results, append_metrics_results=False, print=False)
metrics_results[experiment_id] = metrics_dict

#######################################################################################################################
# Architecture Type = Separated
architecture="Separated Architecture"


# #######################################################################################################################
# #######################################################################################################################
# #######################################################################################################################
# Ambiguity
# #######################################################################################################################
# #######################################################################################################################
# #######################################################################################################################


# # https://contrib.scikit-learn.org/forest-confidence-interval/index.html
# # pip3 install forestci
# from forestci import random_forest_error


# from sklearn.ensemble import RandomForestRegressor
# forest = RandomForestRegressor(n_estimators=100)
# forest.fit(pd.concat([train_x, train_t]), train_y)

# # returns An array with the unbiased sampling variance (V_IJ_unbiased)
# ci = random_forest_error(forest, train_x, test_x, inbag=None, calibrate=True, memory_constrained=False, memory_limit=None)



# quantile-forest (https://pypi.org/project/quantile-forest/): 
# This package offers a different approach. Instead of directly calculating confidence intervals, 
# it provides a RandomForestQuantileRegressor class that allows you to specify quantiles during training. 
# This enables you to directly estimate the desired quantiles (e.g., 2.5% and 97.5% for a 95% confidence interval) 
# and build your intervals based on those estimates.

from quantile_forest import RandomForestQuantileRegressor
from sklearn import datasets

# X, y = datasets.fetch_california_housing(return_X_y=True)
# qrf = RandomForestQuantileRegressor()
# qrf.fit(X, y)
# y_pred = qrf.predict(X, quantiles=[0.025, 0.5, 0.975])


# Convert column names to strings
train_x.columns = train_x.columns.astype(str)  

treated_x = pd.concat([train_treated_x, test_treated_x], ignore_index=True).copy()
treated_y = pd.concat([train_treated_y, test_treated_y], ignore_index=True).copy()
control_x = pd.concat([train_control_x, test_control_x], ignore_index=True).copy()
control_y = pd.concat([train_control_y, test_control_y], ignore_index=True).copy()
x = pd.concat([train_x, test_x], ignore_index=True).copy()
t = pd.concat([test_t, train_t], ignore_index=True).copy()
y = pd.concat([test_y, train_y], ignore_index=True).copy()
potential_y = pd.concat([test_potential_y, train_potential_y], ignore_index=True).copy()
ite = pd.concat([test_ite, train_ite], ignore_index=True).copy()

# Concatenate and reset indices
xt = pd.concat([x, t], axis=1)
y = pd.DataFrame(y)

# Print or inspect concatenated_data to check for NaN values
# print(train_xt[train_xt.isnull().any(axis=1)])

# Fit the RandomForestQuantileRegressor
forest = RandomForestQuantileRegressor()
forest.fit(xt, y.squeeze())

# Predict using the fitted model
y_pred = forest.predict(xt, quantiles=[0.5])

y_lower = forest.predict(xt, quantiles=[0.025])
y_upper = forest.predict(xt, quantiles=[0.975])

y_lower2 = forest.predict(xt, quantiles=[0.05])
y_upper2 = forest.predict(xt, quantiles=[0.95])

y_lower3 = forest.predict(xt, quantiles=[0.10])
y_upper3 = forest.predict(xt, quantiles=[0.90])

y_lower4 = forest.predict(xt, quantiles=[0.15])
y_upper4 = forest.predict(xt, quantiles=[0.85])


# all_data['size_of_ci'] = y_upper - y_lower # confidence interval

all_data['size_of_ci'] = ((y_upper - y_lower) + (y_upper2 - y_lower2) + (y_upper3 - y_lower3) + (y_upper4 - y_lower4)) /4 # confidence interval

# Rejection based on Random Forest
experiment_id += 1
experiment_name =  "Rejection based on Random Forest (confidence interval)"
abbreviation = "RFCI"
experiment_names.update({experiment_id: f"{experiment_name}"})

# loop over all possible RR
reject_rates = []
rmse_accepted = []
rmse_rejected = []

# all_data.sort_values(by='amount_of_times_rejected', ascending=False)
all_data = all_data.sort_values(by='size_of_ci', ascending=False).copy()
print(all_data)
all_data = all_data.reset_index(drop=True)

detail_factor = 10 # 1 or 10
for rr in range(1, 15*detail_factor):
    num_to_set = int(rr / (100.0*detail_factor) * len(all_data)) # example: 60/100 = 0.6 * length of the data
      
    all_data['ite_reject'] = all_data['ite_pred']
    all_data['ite_reject'] = all_data['ite_reject'].astype(object)  # Change dtype of entire column

    all_data.loc[:num_to_set -1, 'ite_reject'] = 'R'

    metrics_result = calculate_performance_metrics('ite', 'ite_reject', all_data, file_path)

    if metrics_result is not None and 'Rejection Rate' in metrics_result:
        reject_rates.append(metrics_result['Rejection Rate'])
        print(f"RR: {rr / (100*detail_factor) }, RR: {metrics_result['Rejection Rate']}")
    else:
        reject_rates.append(None)

    if metrics_result is not None and 'RMSE' in metrics_result:
        rmse_accepted.append(metrics_result['RMSE'])
    else:
        rmse_accepted.append(None)

    if metrics_result is not None and 'RMSE Rejected' in metrics_result:
        rmse_rejected.append(metrics_result['RMSE Rejected'])
    else:
        rmse_rejected.append(None)

# Graph with reject rate and rmse_accepted & rmse_rejected
        
twolinegraph(reject_rates, "Reject Rate", rmse_accepted, "RMSE of Accepted Samples", "green", rmse_rejected, "RMSE of Rejected Samples", "red", f"Impact of Reject Rate on RMSE for {dataset}", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_rmse.png")
onelinegraph(reject_rates, "Reject Rate", rmse_accepted, "RMSE of Accepted Samples", "green", f"Impact of Reject Rate on RMSE for {dataset}", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_rmse_accepted.png")
onelinegraph(reject_rates, "Reject Rate", rmse_rejected, "RMSE of Rejected Samples", "red", f"Impact of Reject Rate on RMSE for {dataset}", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_rmse_rejected.png")

# optimal model
min_rmse = min(rmse_accepted)  # Find the minimum
min_rmse_index = rmse_accepted.index(min_rmse)  # Find the index of the minimum RMSE
optimal_reject_rate = reject_rates[min_rmse_index]  # Get the rejection rate at the same index

all_data['ite_reject'] = all_data['ite_pred']
all_data['ite_reject'] = all_data['ite_reject'].astype(object)  # Change dtype of entire column
all_data.loc[:num_to_set -1, 'ite_reject'] = 'R'

metrics_dict = calculate_all_metrics('ite', 'ite_reject', all_data, file_path, metrics_results, append_metrics_results=False, print=False)
metrics_dict['2/ Optimal RR (%)'] = round(optimal_reject_rate, 4)*100

original_rmse = metrics_dict.get('Original RMSE', None)
metrics_dict['2/ Original RMSE ()'] = original_rmse
metrics_dict['2/ Minimum RMSE'] = round(min_rmse, 4)

metrics_dict['2/ Change of RMSE (%)'] = (min_rmse - original_rmse) / original_rmse * 100
metrics_dict['2/ Improvement of RMSE (%)'] = -((min_rmse - original_rmse) / original_rmse) * 100

# mistake_from_perfect_column = [perfect - actual for perfect, actual in zip(rmse_accepted_perfect, rmse_accepted)]
# mistake_from_perfect = sum(mistake_from_perfect_column)
# metrics_dict['2/ Mistake from Perfect'] = round(mistake_from_perfect, 4)

metrics_results[experiment_id] = metrics_dict

metrics_results = pd.DataFrame(metrics_results)

# improvement_matrix = metrics_results.copy()
# for col in improvement_matrix.columns[0:]:
#     improvement_matrix[col] = round((improvement_matrix[col] - improvement_matrix[col].iloc[0]) / improvement_matrix[col].iloc[0] * 100, 2)
# improvement_matrix = pd.DataFrame(improvement_matrix)

# Chapter 8: Output to file
with open(file_path, 'a') as file:

    file.write("\n\nTable of all_data (First 5 rows)\n")
    file.write(tabulate(all_data.head(5), headers='keys', tablefmt='pretty', showindex=False))
    
    file.write ("\n")
    for exp_number, description in experiment_names.items():
        file.write(f"# Experiment {exp_number}: {description}\n")

    file.write("\nTable of results of the experiments\n")
    file.write(tabulate(metrics_results, headers='keys', tablefmt='rounded_grid', showindex=True))
    
    # file.write("\nTable of the improvement of the experiments\n")
    # file.write(tabulate(improvement_matrix, headers='keys', tablefmt='rounded_grid', showindex=True))

    # # Category != category_pred
    # file.write("\n\nTable of test_set with wrong classification\n")
    # file.write(tabulate(test_set[test_set['category'] != test_set['category_pred']], headers='keys', tablefmt='pretty', showindex=False))