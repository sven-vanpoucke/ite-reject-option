"""
Table of contents:

Chapter 0: Imports

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
dataset = "IHDP" # Choose out of TWINS or TWINSC (if you want TWINS to be treated as continuous instead of classification) or LALONDE or IHDP
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

## Merge the test_set with the train_set !!
treated_x = pd.concat([train_treated_x, test_treated_x], ignore_index=True).copy()
treated_y = pd.concat([train_treated_y, test_treated_y], ignore_index=True).copy()
control_x = pd.concat([train_control_x, test_control_x], ignore_index=True).copy()
control_y = pd.concat([train_control_y, test_control_y], ignore_index=True).copy()
x = pd.concat([train_x, test_x], ignore_index=True).copy()
t = pd.concat([train_t, test_t], ignore_index=True).copy()
xt = pd.concat([x, t], axis=1)
y = pd.concat([train_y, test_y], ignore_index=True).copy()
y = pd.DataFrame(y)
ite = pd.concat([train_ite, test_ite], ignore_index=True).copy()
potential_y = pd.concat([train_potential_y, test_potential_y], ignore_index=True).copy()

# Chapter 3: Training of the ITE Model

## Training of the ITE Model (T-learner: This model is trained on the treated and control groups separately)
train_treated_model, train_control_model = predictor_t_model(train_treated_x, train_treated_y, train_control_x, train_control_y, model_class, model_params)
treated_model, control_model = predictor_t_model(treated_x, treated_y, control_x, control_y, model_class, model_params)

# Chapter 4: Predictions based on T learner trained on train set
## Training and Testing predictions to evaluate individual models
train_treated_y_pred, train_treated_y_prob, train_control_y_pred, train_control_y_prob = predictor_train_predictions(train_treated_model, train_control_model, train_treated_x, train_control_x)
test_treated_y_pred, test_treated_y_prob, test_control_y_pred, test_control_y_prob = predictor_test_predictions(train_treated_model, train_control_model, test_treated_x, test_control_x)
treated_y_pred, treated_y_prob, control_y_pred, control_y_prob = predictor_train_predictions(treated_model, control_model, treated_x, control_x)

## Testing Predictions to evaluate ITE
train_y_t1_pred, train_y_t0_pred, train_y_t1_prob, train_y_t0_prob, train_ite_prob, train_ite_pred = predictor_ite_predictions(train_treated_model, train_control_model, train_x)
test_y_t1_pred, test_y_t0_pred, test_y_t1_prob, test_y_t0_prob, test_ite_prob, test_ite_pred = predictor_ite_predictions(train_treated_model, train_control_model, test_x)
y_t1_pred, y_t0_pred, y_t1_prob, y_t0_prob, ite_prob, ite_pred = predictor_ite_predictions(treated_model, control_model, x)

## Merge the different dataframes
if train_treated_y_prob is not None and not train_treated_y_prob.isna().all():
    test_set = pd.concat([test_t, test_y_t1_pred, test_y_t1_prob, test_y_t0_pred, test_y_t0_prob, test_ite_pred, test_ite_prob, test_potential_y["y_t0"], test_potential_y["y_t1"], test_ite], axis=1)
    train_set = pd.concat([test_t, train_y_t1_pred, train_y_t1_prob, train_y_t0_pred, train_y_t0_prob, train_ite_pred, train_ite_prob, train_potential_y["y_t0"], train_potential_y["y_t1"], train_ite], axis=1)
    set = pd.concat([t, y_t1_pred, y_t1_prob, y_t0_pred, y_t0_prob, ite_pred, ite_prob, potential_y["y_t0"], potential_y["y_t1"], ite], axis=1)
else:
    test_set = pd.concat([test_t, test_y_t1_pred, test_y_t0_pred, test_ite_pred, test_potential_y["y_t0"], test_potential_y["y_t1"], test_ite], axis=1)
    train_set = pd.concat([test_t, train_y_t1_pred, train_y_t0_pred, train_ite_pred, train_potential_y["y_t0"], train_potential_y["y_t1"], train_ite], axis=1)
    set = pd.concat([t, y_t1_pred, y_t0_pred, ite_pred, potential_y["y_t0"], potential_y["y_t1"], ite], axis=1)
    
## Make TWINS (binary) to TWINSC (continuous)
if dataset == "TWINSC":
    # Delete columns y_t1_pred and y_t0_pred, ite_pred
    test_set = test_set.drop(['y_t1_pred', 'y_t0_pred', 'ite_pred'], axis=1)
    # Rename columns y_t1_prob, y_t0_prob, ite_prob to y_t1_pred, y_t0_pred, ite_pred
    test_set = test_set.rename(columns={'y_t1_prob': 'y_t1_pred', 'y_t0_prob': 'y_t0_pred', 'ite_prob': 'ite_pred'})

    train_set = train_set.drop(['y_t1_pred', 'y_t0_pred', 'ite_pred'], axis=1)
    train_set = train_set.rename(columns={'y_t1_prob': 'y_t1_pred', 'y_t0_prob': 'y_t0_pred', 'ite_prob': 'ite_pred'})

    set = set.drop(['y_t1_pred', 'y_t0_pred', 'ite_pred'], axis=1)
    set = set.rename(columns={'y_t1_prob': 'y_t1_pred', 'y_t0_prob': 'y_t0_pred', 'ite_prob': 'ite_pred'})

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

#######################################################################################################################
# Architecture Type = Separated
architecture="Separated Architecture"

if x_scaling:
    scaler = StandardScaler()
    x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

#######################################################################################################################
# Rejection

# No Rejection
experiment_id = 0
experiment_names = {}
experiment_name = "No Rejector - Baseline Model"
experiment_names.update({experiment_id: f"{experiment_name}"})

all_data = pd.concat([train_set, test_set], ignore_index=True).copy()
all_data['ite_reject'] = all_data.apply(lambda row: row['ite_pred'], axis=1)

# Step 5 Calculate the performance metrics
metrics_dict = calculate_all_metrics('ite', 'ite_reject', all_data, file_path, metrics_results, append_metrics_results=False, print=False)
metrics_results[experiment_id] = metrics_dict

# Type 0
experiment_id += 1
experiment_name =  "Perfect Rejection"
abbreviation = "Perfect"
experiment_names.update({experiment_id: f"{experiment_name}"})
# all_data['se'] = (all_data['ite'] - all_data['ite_pred']) ** 2
# rmse_accepted_perfect, metrics_results[experiment_id] = novelty_rejection(0, max_rr, detail_factor, IsolationForest, x, all_data, file_path, experiment_id, dataset, folder_path, abbreviation)
# loop over all possible RR
reject_rates = []
rmse_accepted = []
rmse_rejected = []
change_rmse = []

all_data['se'] = (all_data['ite'] - all_data['ite_pred']) ** 2
all_data = all_data.sort_values(by='se', ascending=False).copy()
all_data = all_data.reset_index(drop=True)

for rr in range(1, max_rr*detail_factor):
    num_to_set = int(rr / (100.0*detail_factor) * len(all_data)) # example: 60/100 = 0.6 * length of the data

    all_data['ite_reject'] = all_data['ite_pred']
    all_data['ite_reject'] = all_data['ite_reject'].astype(object)  # Change dtype of entire column
    all_data.loc[:num_to_set -1, 'ite_reject'] = 'R'

    metrics_result = calculate_performance_metrics('ite', 'ite_reject', all_data, file_path)

    if metrics_result:
        reject_rates.append(metrics_result.get('Rejection Rate', None))
        rmse_accepted.append(metrics_result.get('RMSE Accepted', None))
        rmse_rejected.append(metrics_result.get('RMSE Rejected', None))
    else:
        reject_rates.append(None)
        rmse_accepted.append(None)
        rmse_rejected.append(None)

rmse_accepted_perfect = rmse_accepted
rmse_rejected_perfect = rmse_rejected
rr_perfect = reject_rates

# Graph with reject rate and rmse_accepted & rmse_rejected
twolinegraph(reject_rates, "Reject Rate", rmse_accepted, "RMSE of Accepted Samples", "green", rmse_rejected, "RMSE of Rejected Samples", "red", f"Impact of Reject Rate on RMSE for {dataset}", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_rmse.png")
onelinegraph(reject_rates, "Reject Rate", rmse_accepted, "RMSE of Accepted Samples", "green", f"Impact of Reject Rate on RMSE for {dataset}", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_rmse_accepted.png")
onelinegraph(reject_rates, "Reject Rate", rmse_rejected, "RMSE of Rejected Samples", "red", f"Impact of Reject Rate on RMSE for {dataset}", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_rmse_rejected.png")

# Optimal Model
min_rmse = min(rmse_accepted)  # Find the minimum
min_rmse_index = rmse_accepted.index(min_rmse)  # Find the index of the minimum RMSE
optimal_reject_rate = reject_rates[min_rmse_index]  # Get the rejection rate at the same index

all_data['ite_reject'] = all_data['ite_pred']
all_data['ite_reject'] = all_data['ite_reject'].astype(object)  # Change dtype of entire column
all_data.loc[:num_to_set -1, 'ite_reject'] = 'R'

metrics_dict = calculate_all_metrics('ite', 'ite_reject', all_data, file_path, metrics_results, append_metrics_results=False, print=False)
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