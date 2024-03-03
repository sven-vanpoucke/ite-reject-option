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

## Train ITE Model
from models.predictor import predictor_t_model
from sklearn.linear_model import LogisticRegression, LinearRegression
from models.predictor import predictor_train_predictions, predictor_test_predictions, predictor_ite_predictions
from quantile_forest import RandomForestQuantileRegressor

# Evaluate
from models.evaluators.cost_evaluator import categorize
from models.evaluators.evaluator import calculate_all_metrics
from models.evaluators.evaluator import calculate_performance_metrics

# Graphs
from models.rejectors.helper import onelinegraph, twolinegraph

# Rejection
from models.rejectors.helper import novelty_rejection, ambiguity_rejection
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM



# Chapter 1: Initialization

## Parameters
### To choose
dataset = "IHDP" # Choose out of TWINS or TWINSC (if you want TWINS to be treated as continuous instead of classification) or LALONDE or IHDP
psm = False
### Rejection
detail_factor = 1 # 1 (no extra detail) or 10 (extra detail)
max_rr = 15 # (number between 1 and 49)
x_scaling = False # True or False

### Not to choose
folder_path = 'output/'
text_folder_path = 'output/text/'
timestamp, file_name, file_path = helper_output(dataset, folder_path=text_folder_path)
metrics_results = {}
experiment_names = {}
experiment_id = -1


# Chapter 2: Preprocessing

## Output
with open(file_path, 'a') as file:
    file.write(f"\Chapter 2: Preprocessing\n\n")
    file.write("# This section executes the data retrieval, preprocessing and splitting in a training and dataset.")
    file.write(f"During the whole file, the used dataset is: {dataset}\n\n")

## Retrieve Data
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

## Edit Data
train_treated_x, train_control_x, train_treated_y, train_control_y, test_treated_x, test_control_x, test_treated_y, test_control_y = preprocessing_split_t_c_data(train_x, train_y, train_t, test_x, test_y, test_t)
test_ite = pd.DataFrame({'ite': test_potential_y["y_t1"] - test_potential_y["y_t0"]})
train_ite = pd.DataFrame({'ite': train_potential_y["y_t1"] - train_potential_y["y_t0"]})

## Merge test_set & the train_set
treated_x = pd.concat([train_treated_x, test_treated_x], ignore_index=True).copy() # Under each other
treated_y = pd.concat([train_treated_y, test_treated_y], ignore_index=True).copy()
control_x = pd.concat([train_control_x, test_control_x], ignore_index=True).copy()
control_y = pd.concat([train_control_y, test_control_y], ignore_index=True).copy()
x = pd.concat([train_x, test_x], ignore_index=True).copy()
t = pd.concat([train_t, test_t], ignore_index=True).copy()
xt = pd.concat([x, t], axis=1).copy() # Left & right from eachother
train_xt = pd.concat([train_x, train_t], ignore_index=True).copy()
test_xt = pd.concat([test_x, test_t], ignore_index=True).copy()
y = pd.concat([train_y, test_y], ignore_index=True).copy()
y = pd.DataFrame(y)
ite = pd.concat([train_ite, test_ite], ignore_index=True).copy()
potential_y = pd.concat([train_potential_y, test_potential_y], ignore_index=True).copy()

if x_scaling:
    scaler = StandardScaler()
    x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

# Chapter 3: Train ITE Model
## T-learner: train_set
train_treated_model, train_control_model = predictor_t_model(train_treated_x, train_treated_y, train_control_x, train_control_y, model_class, model_params)

## T-learner: all_set
treated_model, control_model = predictor_t_model(treated_x, treated_y, control_x, control_y, model_class, model_params)

## RandomForestQuantileRegressor: train_set
train_forest_model = RandomForestQuantileRegressor()
train_forest_model.fit(xt, y.squeeze())

## RandomForestQuantileRegressor: all_set
forest_model = RandomForestQuantileRegressor()
forest_model.fit(xt, y.squeeze())

# Chapter 4: Predict
## T-Learner - train_set
train_treated_y_pred, train_treated_y_prob, train_control_y_pred, train_control_y_prob = predictor_train_predictions(train_treated_model, train_control_model, train_treated_x, train_control_x)
train_y_t1_pred, train_y_t0_pred, train_y_t1_prob, train_y_t0_prob, train_ite_prob, train_ite_pred = predictor_ite_predictions(train_treated_model, train_control_model, train_x)

## T-Learner - test_set
test_treated_y_pred, test_treated_y_prob, test_control_y_pred, test_control_y_prob = predictor_test_predictions(train_treated_model, train_control_model, test_treated_x, test_control_x)
test_y_t1_pred, test_y_t0_pred, test_y_t1_prob, test_y_t0_prob, test_ite_prob, test_ite_pred = predictor_ite_predictions(train_treated_model, train_control_model, test_x)

## T-Learner - set
treated_y_pred, treated_y_prob, control_y_pred, control_y_prob = predictor_train_predictions(treated_model, control_model, treated_x, control_x)
y_t1_pred, y_t0_pred, y_t1_prob, y_t0_prob, ite_prob, ite_pred = predictor_ite_predictions(treated_model, control_model, x)

## RandomForestQuantileRegressor - train_set
train_ite_pred = train_forest_model.predict(train_xt, quantiles=[0.5])
test_ite_pred = train_forest_model.predict(test_xt, quantiles=[0.5])

## RandomForestQuantileRegressor - all_set
ite_pred = forest_model.predict(xt, quantiles=[0.5])

# Chapter 5: Process Data
## Merge Outcomes
if train_treated_y_prob is not None and not train_treated_y_prob.isna().all():
    test_set = pd.concat([test_t, test_y_t1_pred, test_y_t1_prob, test_y_t0_pred, test_y_t0_prob, test_ite_pred, test_ite_prob, test_potential_y["y_t0"], test_potential_y["y_t1"], test_ite], axis=1)
    train_set = pd.concat([test_t, train_y_t1_pred, train_y_t1_prob, train_y_t0_pred, train_y_t0_prob, train_ite_pred, train_ite_prob, train_potential_y["y_t0"], train_potential_y["y_t1"], train_ite], axis=1)
    all_set = pd.concat([t, y_t1_pred, y_t1_prob, y_t0_pred, y_t0_prob, ite_pred, ite_prob, potential_y["y_t0"], potential_y["y_t1"], ite], axis=1).copy()
    
    train_forest_set = pd.concat([train_t, train_ite_pred, train_ite], axis=1).copy()
    test_forest_set = pd.concat([test_t, test_ite_pred, test_ite], axis=1).copy()
    forest_set = pd.concat([t, ite_pred, ite], axis=1).copy()
else:
    test_set = pd.concat([test_t, test_y_t1_pred, test_y_t0_pred, test_ite_pred, test_potential_y["y_t0"], test_potential_y["y_t1"], test_ite], axis=1)
    train_set = pd.concat([test_t, train_y_t1_pred, train_y_t0_pred, train_ite_pred, train_potential_y["y_t0"], train_potential_y["y_t1"], train_ite], axis=1)
    all_set = pd.concat([t, y_t1_pred, y_t0_pred, ite_pred, potential_y["y_t0"], potential_y["y_t1"], ite], axis=1).copy()
    
    train_forest_set = pd.concat([train_t, train_ite_pred, train_ite], axis=1).copy()
    test_forest_set = pd.concat([test_t, test_ite_pred, test_ite], axis=1).copy()
    forest_set = pd.concat([t, ite_pred, ite], axis=1).copy()

## Make TWINS (binary) to TWINSC (continuous)
if dataset == "TWINSC":
    # Delete columns y_t1_pred and y_t0_pred, ite_pred
    test_set = test_set.drop(['y_t1_pred', 'y_t0_pred', 'ite_pred'], axis=1)
    # Rename columns y_t1_prob, y_t0_prob, ite_prob to y_t1_pred, y_t0_pred, ite_pred
    test_set = test_set.rename(columns={'y_t1_prob': 'y_t1_pred', 'y_t0_prob': 'y_t0_pred', 'ite_prob': 'ite_pred'})

    train_set = train_set.drop(['y_t1_pred', 'y_t0_pred', 'ite_pred'], axis=1)
    train_set = train_set.rename(columns={'y_t1_prob': 'y_t1_pred', 'y_t0_prob': 'y_t0_pred', 'ite_prob': 'ite_pred'})

    all_set = all_set.drop(['y_t1_pred', 'y_t0_pred', 'ite_pred'], axis=1)
    all_set = all_set.rename(columns={'y_t1_prob': 'y_t1_pred', 'y_t0_prob': 'y_t0_pred', 'ite_prob': 'ite_pred'})

## Merge train & test
all_data = pd.concat([train_set, test_set], ignore_index=True).copy()
forest_all_data = pd.concat([train_forest_set, test_forest_set], ignore_index=True).copy()

# Chapter 6: Costs Performance
test_set['category'] = test_set.apply(categorize, axis=1, is_pred=False)
test_set['category_pred'] = test_set.apply(categorize, axis=1)
test_set['category_rej'] = test_set.apply(categorize, axis=1)
test_set['ite_mistake'] = test_set.apply(lambda row: 0 if row['ite_pred']==row['ite'] else 1, axis=1)

#######################################################################################################################

# CHAPTER 7: REJECTION
## Output
with open(file_path, 'a') as file:
    file.write(f"\nCHAPTER 7: REJECTION \n\n")
    file.write("# This section executes and reports metrics for ITE models with rejection.\n")

#######################################################################################################################
# Rejection Architecture
architecture="Separated Architecture"

#######################################################################################################################
# No Rejection
experiment_id += 1
experiment_name = "No Rejector - Baseline Model"
experiment_names.update({experiment_id: f"{experiment_name}"})

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
        rmse_accepted.append(metrics_result.get('RMSE', None))
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

# optimal model
min_rmse = min(rmse_accepted)  # Find the minimum
min_rmse_index = rmse_accepted.index(min_rmse)  # Find the index of the minimum RMSE
optimal_reject_rate = reject_rates[min_rmse_index]  # Get the rejection rate at the same index

all_data['ite_reject'] = all_data['ite_pred']
all_data['ite_reject'] = all_data['ite_reject'].astype(object)  # Change dtype of entire column
all_data.loc[:num_to_set -1, 'ite_reject'] = 'R'

metrics_dict = calculate_all_metrics('ite', 'ite_reject', all_data, file_path, metrics_results, append_metrics_results=False, print=False)
metrics_results[experiment_id] = metrics_dict

# Type 1
for model, abbreviation in zip([IsolationForest, OneClassSVM, LocalOutlierFactor], ["IF", "OCSVM", "LOF"]):
    experiment_id += 1
    experiment_names[experiment_id] = f"Rejection based on {model.__name__} (train data) - Novelty Type I"
    metrics_results[experiment_id] = novelty_rejection(1, max_rr, detail_factor, model, x, all_data, file_path, experiment_id, dataset, folder_path, abbreviation, rmse_accepted_perfect)

# Type 1
for model, abbreviation in zip([IsolationForest, OneClassSVM, LocalOutlierFactor], ["IF", "OCSVM", "LOF"]):
    experiment_id += 1
    experiment_names[experiment_id] = f"Rejection based on {model.__name__} (train data) - Novelty Type II"
    metrics_results[experiment_id] = novelty_rejection(2, max_rr, detail_factor, model, x, all_data, file_path, experiment_id, dataset, folder_path, abbreviation, rmse_accepted_perfect)

# Type 3
for model, abbreviation in zip([IsolationForest, OneClassSVM, LocalOutlierFactor], ["IF", "OCSVM", "LOF"]):
    experiment_id += 1
    experiment_names[experiment_id] = f"Rejection based on {model.__name__} (train data) - Novelty Type III"
    metrics_results[experiment_id] = novelty_rejection(3, max_rr, detail_factor, model, x, all_data, file_path, experiment_id, dataset, folder_path, abbreviation, rmse_accepted_perfect)

#######################################################################################################################
# T-Learner trained on All Data
#######################################################################################################################

# No Rejection
experiment_id += 1
experiment_name = "No Rejector - Baseline Model"
experiment_names.update({experiment_id: f"{experiment_name}"})

all_set['ite_reject'] = all_set.apply(lambda row: row['ite_pred'], axis=1)

# Step 5 Calculate the performance metrics
metrics_dict = calculate_all_metrics('ite', 'ite_reject', all_set, file_path, metrics_results, append_metrics_results=False, print=False)
metrics_results[experiment_id] = metrics_dict

# Type 0
experiment_id += 1
experiment_name =  "Perfect Rejection"
abbreviation = "Perfect"
experiment_names.update({experiment_id: f"{experiment_name}"})
# all_set['se'] = (all_set['ite'] - all_set['ite_pred']) ** 2
# rmse_accepted_perfect, metrics_results[experiment_id] = novelty_rejection(0, max_rr, detail_factor, IsolationForest, x, all_set, file_path, experiment_id, dataset, folder_path, abbreviation)
# loop over all possible RR

reject_rates = []
rmse_accepted = []
rmse_rejected = []
change_rmse = []

all_set['se'] = (all_set['ite'] - all_set['ite_pred']) ** 2
all_set = all_set.sort_values(by='se', ascending=False).copy()
all_set = all_set.reset_index(drop=True)

for rr in range(1, max_rr*detail_factor):
    num_to_set = int(rr / (100.0*detail_factor) * len(all_set)) # example: 60/100 = 0.6 * length of the data

    all_set['ite_reject'] = all_set['ite_pred']
    all_set['ite_reject'] = all_set['ite_reject'].astype(object)  # Change dtype of entire column
    all_set.loc[:num_to_set -1, 'ite_reject'] = 'R'

    metrics_result = calculate_performance_metrics('ite', 'ite_reject', all_set, file_path)

    if metrics_result:
        reject_rates.append(metrics_result.get('Rejection Rate', None))
        rmse_accepted.append(metrics_result.get('RMSE', None))
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

# optimal model
min_rmse = min(rmse_accepted)  # Find the minimum
min_rmse_index = rmse_accepted.index(min_rmse)  # Find the index of the minimum RMSE
optimal_reject_rate = reject_rates[min_rmse_index]  # Get the rejection rate at the same index

all_set['ite_reject'] = all_set['ite_pred']
all_set['ite_reject'] = all_set['ite_reject'].astype(object)  # Change dtype of entire column
all_set.loc[:num_to_set -1, 'ite_reject'] = 'R'

metrics_dict = calculate_all_metrics('ite', 'ite_reject', all_set, file_path, metrics_results, append_metrics_results=False, print=False)
metrics_results[experiment_id] = metrics_dict

# Type 1
for model, abbreviation in zip([IsolationForest, OneClassSVM, LocalOutlierFactor], ["IF", "OCSVM", "LOF"]):
    experiment_id += 1
    experiment_names[experiment_id] = f"Rejection based on {model.__name__} (train data) - Novelty Type I"
    metrics_results[experiment_id] = novelty_rejection(1, max_rr, detail_factor, model, x, all_set, file_path, experiment_id, dataset, folder_path, abbreviation, rmse_accepted_perfect)

# Type 1
for model, abbreviation in zip([IsolationForest, OneClassSVM, LocalOutlierFactor], ["IF", "OCSVM", "LOF"]):
    experiment_id += 1
    experiment_names[experiment_id] = f"Rejection based on {model.__name__} (train data) - Novelty Type II"
    metrics_results[experiment_id] = novelty_rejection(2, max_rr, detail_factor, model, x, all_set, file_path, experiment_id, dataset, folder_path, abbreviation, rmse_accepted_perfect)

# Type 3
for model, abbreviation in zip([IsolationForest, OneClassSVM, LocalOutlierFactor], ["IF", "OCSVM", "LOF"]):
    experiment_id += 1
    experiment_names[experiment_id] = f"Rejection based on {model.__name__} (train data) - Novelty Type III"
    metrics_results[experiment_id] = novelty_rejection(3, max_rr, detail_factor, model, x, all_set, file_path, experiment_id, dataset, folder_path, abbreviation, rmse_accepted_perfect)

# #######################################################################################################################
# #######################################################################################################################
# Ambiguity
# #######################################################################################################################
# #######################################################################################################################


# Confidence Interval for Ambiguity Rejection
def confidence_interval(xt, forest_model):
    y_lower = forest_model.predict(xt, quantiles=[0.025])
    y_upper = forest_model.predict(xt, quantiles=[0.975])

    y_lower2 = forest_model.predict(xt, quantiles=[0.05])
    y_upper2 = forest_model.predict(xt, quantiles=[0.95])

    y_lower3 = forest_model.predict(xt, quantiles=[0.10])
    y_upper3 = forest_model.predict(xt, quantiles=[0.90])

    y_lower4 = forest_model.predict(xt, quantiles=[0.15])
    y_upper4 = forest_model.predict(xt, quantiles=[0.85])

    size_of_ci = ((y_upper - y_lower) + (y_upper2 - y_lower2) + (y_upper3 - y_lower3) + (y_upper4 - y_lower4)) /4 # confidence interval

    return size_of_ci


#######################################################################################################################
# Regression Forest
# quantile-forest (https://pypi.org/project/quantile-forest/): 
# This package offers a different approach. Instead of directly calculating confidence intervals, 
# it provides a RandomForestQuantileRegressor class that allows you to specify quantiles during training. 
# This enables you to directly estimate the desired quantiles (e.g., 2.5% and 97.5% for a 95% confidence interval) 
# and build your intervals based on those estimates.

#######################################################################################################################

# No Rejection
experiment_id += 1
experiment_name = "No Regression - Rejection based on Regression Forest (C.I.)"
experiment_names.update({experiment_id: f"{experiment_name}"})

forest_set['ite_reject'] = forest_set.apply(lambda row: row['ite_pred'], axis=1)

# Step 5 Calculate the performance metrics
metrics_dict = calculate_all_metrics('ite', 'ite_reject', forest_set, file_path, metrics_results, append_metrics_results=False, print=False)
metrics_results[experiment_id] = metrics_dict

# Type 0
experiment_id += 1
experiment_name =  "Perfect Rejection"
abbreviation = "Perfect"
experiment_names.update({experiment_id: f"{experiment_name}"})
# forest_set['se'] = (forest_set['ite'] - forest_set['ite_pred']) ** 2
# rmse_accepted_perfect, metrics_results[experiment_id] = novelty_rejection(0, max_rr, detail_factor, IsolationForest, x, forest_set, file_path, experiment_id, dataset, folder_path, abbreviation)
# loop over all possible RR

reject_rates = []
rmse_accepted = []
rmse_rejected = []
change_rmse = []

forest_set['se'] = (forest_set['ite'] - forest_set['ite_pred']) ** 2
forest_set = forest_set.sort_values(by='se', ascending=False).copy()
forest_set = forest_set.reset_index(drop=True)

for rr in range(1, max_rr*detail_factor):
    num_to_set = int(rr / (100.0*detail_factor) * len(forest_set)) # example: 60/100 = 0.6 * length of the data

    forest_set['ite_reject'] = forest_set['ite_pred']
    forest_set['ite_reject'] = forest_set['ite_reject'].astype(object)  # Change dtype of entire column
    forest_set.loc[:num_to_set -1, 'ite_reject'] = 'R'

    metrics_result = calculate_performance_metrics('ite', 'ite_reject', forest_set, file_path)

    if metrics_result:
        reject_rates.append(metrics_result.get('Rejection Rate', None))
        rmse_accepted.append(metrics_result.get('RMSE', None))
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

# optimal model
min_rmse = min(rmse_accepted)  # Find the minimum
min_rmse_index = rmse_accepted.index(min_rmse)  # Find the index of the minimum RMSE
optimal_reject_rate = reject_rates[min_rmse_index]  # Get the rejection rate at the same index

forest_set['ite_reject'] = forest_set['ite_pred']
forest_set['ite_reject'] = forest_set['ite_reject'].astype(object)  # Change dtype of entire column
forest_set.loc[:num_to_set -1, 'ite_reject'] = 'R'

metrics_dict = calculate_all_metrics('ite', 'ite_reject', forest_set, file_path, metrics_results, append_metrics_results=False, print=False)
metrics_results[experiment_id] = metrics_dict

# Type 1
experiment_id += 1
model = "RandomForestQuantileRegressor"
abbreviation = "RFQR"
experiment_names[experiment_id] = f"Rejection based on RandomForestQuantileRegressor - Ambiguity Type I"
metrics_results[experiment_id] = ambiguity_rejection(1, max_rr, detail_factor, model, x, forest_set, file_path, experiment_id, dataset, folder_path, abbreviation, rmse_accepted_perfect)

# #######################################################################################################################

# No Rejection
experiment_id += 1
experiment_name = "No Regression (Trained DATA) - Rejection based on Regression Forest (C.I.)"
experiment_names.update({experiment_id: f"{experiment_name}"})

forest_set['ite_reject'] = forest_set.apply(lambda row: row['ite_pred'], axis=1)

# Step 5 Calculate the performance metrics
metrics_dict = calculate_all_metrics('ite', 'ite_reject', forest_set, file_path, metrics_results, append_metrics_results=False, print=False)
metrics_results[experiment_id] = metrics_dict

# Type 0
experiment_id += 1
experiment_name =  "Perfect Rejection"
abbreviation = "Perfect"
experiment_names.update({experiment_id: f"{experiment_name}"})
# forest_all_data['se'] = (forest_all_data['ite'] - forest_all_data['ite_pred']) ** 2
# rmse_accepted_perfect, metrics_results[experiment_id] = novelty_rejection(0, max_rr, detail_factor, IsolationForest, x, forest_all_data, file_path, experiment_id, dataset, folder_path, abbreviation)
# loop over all possible RR

reject_rates = []
rmse_accepted = []
rmse_rejected = []
change_rmse = []

forest_all_data['se'] = (forest_all_data['ite'] - forest_all_data['ite_pred']) ** 2
forest_all_data = forest_all_data.sort_values(by='se', ascending=False).copy()
forest_all_data = forest_all_data.reset_index(drop=True)

for rr in range(1, max_rr*detail_factor):
    num_to_set = int(rr / (100.0*detail_factor) * len(forest_all_data)) # example: 60/100 = 0.6 * length of the data

    forest_all_data['ite_reject'] = forest_all_data['ite_pred']
    forest_all_data['ite_reject'] = forest_all_data['ite_reject'].astype(object)  # Change dtype of entire column
    forest_all_data.loc[:num_to_set -1, 'ite_reject'] = 'R'

    metrics_result = calculate_performance_metrics('ite', 'ite_reject', forest_all_data, file_path)

    if metrics_result:
        reject_rates.append(metrics_result.get('Rejection Rate', None))
        rmse_accepted.append(metrics_result.get('RMSE', None))
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

# optimal model
min_rmse = min(rmse_accepted)  # Find the minimum
min_rmse_index = rmse_accepted.index(min_rmse)  # Find the index of the minimum RMSE
optimal_reject_rate = reject_rates[min_rmse_index]  # Get the rejection rate at the same index

forest_all_data['ite_reject'] = forest_all_data['ite_pred']
forest_all_data['ite_reject'] = forest_all_data['ite_reject'].astype(object)  # Change dtype of entire column
forest_all_data.loc[:num_to_set -1, 'ite_reject'] = 'R'

metrics_dict = calculate_all_metrics('ite', 'ite_reject', forest_all_data, file_path, metrics_results, append_metrics_results=False, print=False)
metrics_results[experiment_id] = metrics_dict

# Type 1
experiment_id += 1
model = "RandomForestQuantileRegressor"
abbreviation = "RFQR"
experiment_names[experiment_id] = f"Rejection based on RandomForestQuantileRegressor - Ambiguity Type I"
metrics_results[experiment_id] = ambiguity_rejection(1, max_rr, detail_factor, model, x, forest_all_data, file_path, experiment_id, dataset, folder_path, abbreviation, rmse_accepted_perfect)

# #######################################################################################################################
# # # https://contrib.scikit-learn.org/forest-confidence-interval/index.html
# # # pip3 install forestci
# from forestci import random_forest_error
# #######################################################################################################################

# from sklearn.ensemble import RandomForestRegressor
# forest = RandomForestRegressor(n_estimators=100)
# forest.fit(pd.concat([train_x, train_t]), train_y)

# # # returns An array with the unbiased sampling variance (V_IJ_unbiased)
# ci = random_forest_error(forest, train_x, test_x, inbag=None, calibrate=True, memory_constrained=False, memory_limit=None)


# #######################################################################################################################
# #######################################################################################################################
# #######################################################################################################################
# Propensity score matching
# #######################################################################################################################
# #######################################################################################################################
# #######################################################################################################################

# Calculate propensity scores for training data
train_propensity_scores = calculate_propensity_scores(train_x, train_t)
test_propensity_scores = calculate_propensity_scores(test_x, test_t)

# Perform KNN matching on training data
matched_control_x, matched_control_y = knn_matching(train_treated_x, train_control_x, train_control_y, n_neighbors=5)  # Modify n_neighbors as needed

# Chapter 3: Training of the ITE Model
## Output to file
with open(file_path, 'a') as file:
    file.write(f"CHAPTER 3: Training of the ITE Model\n\n")
    file.write("# This section provides details about the model selection, training process, and any hyperparameter tuning.\n")
    file.write(f"The trained ITE model is a T-LEARNER.\n")
    file.write(f"The two individually trained models are: {model_class.__name__}\n\n")

# ## Training of the ITE Model (T-learner: This model is trained on the treated and control groups separately)
# model = LinearRegression()
# model.fit(matched_control_x, matched_control_y)
# train_y_t1_pred, train_y_t0_pred, 
# predicted_y = model.predict(test_x) # outcome

treated_model, control_model = predictor_t_model(train_treated_x, train_treated_y, matched_control_x, matched_control_y, model_class, model_params)

# Chapter 4: Predictions
## Training and Testing predictions to evaluate individual models
train_treated_y_pred, train_treated_y_prob, train_control_y_pred, train_control_y_prob = predictor_train_predictions(treated_model, control_model, train_treated_x, train_control_x)
test_treated_y_pred, test_treated_y_prob, test_control_y_pred, test_control_y_prob = predictor_test_predictions(treated_model, control_model, test_treated_x, test_control_x)
## Testing Predictions to evaluate ITE
train_y_t1_pred, train_y_t0_pred, train_y_t1_prob, train_y_t0_prob, train_ite_prob, train_ite_pred = predictor_ite_predictions(treated_model, control_model, train_x)
test_y_t1_pred, test_y_t0_pred, test_y_t1_prob, test_y_t0_prob, test_ite_prob, test_ite_pred = predictor_ite_predictions(treated_model, control_model, test_x)

## Merge the different dataframes
if train_treated_y_prob is not None and not train_treated_y_prob.isna().all():
    test_set = pd.concat([test_t, test_y_t1_pred, test_y_t1_prob, test_y_t0_pred, test_y_t0_prob, test_ite_pred, test_ite_prob, test_potential_y["y_t0"], test_potential_y["y_t1"], test_ite], axis=1).copy()
    train_set = pd.concat([test_t, train_y_t1_pred, train_y_t1_prob, train_y_t0_pred, train_y_t0_prob, train_ite_pred, train_ite_prob, train_potential_y["y_t0"], train_potential_y["y_t1"], train_ite], axis=1).copy()
else:
    test_set = pd.concat([test_t, test_y_t1_pred, test_y_t0_pred, test_ite_pred, test_potential_y["y_t0"], test_potential_y["y_t1"], test_ite], axis=1).copy()
    train_set = pd.concat([test_t, train_y_t1_pred, train_y_t0_pred, train_ite_pred, train_potential_y["y_t0"], train_potential_y["y_t1"], train_ite], axis=1).copy()

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

#######################################################################################################################
# Baseline Model - No Rejection // Experiment 0
experiment_id += 1
experiment_name = "No Rejector - Baseline Model (PSM)"
experiment_names.update({experiment_id: f"{experiment_name}"})

# Step 4 Apply rejector to the code
all_data['ite_reject'] = all_data.apply(lambda row: row['ite_pred'], axis=1)

# Step 5 Calculate the performance metrics
metrics_dict = calculate_all_metrics('ite', 'ite_reject', all_data, file_path, metrics_results, append_metrics_results=False, print=False)
metrics_results[experiment_id] = metrics_dict

#######################################################################################################################
# Type 0
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
        rmse_accepted.append(metrics_result.get('RMSE', None))
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

# Type 1
for model, abbreviation in zip([IsolationForest, OneClassSVM, LocalOutlierFactor], ["IF", "OCSVM", "LOF"]):
    experiment_id += 1
    experiment_names[experiment_id] = f"Rejection based on {model.__name__} - Novelty Type I"
    metrics_results[experiment_id] = novelty_rejection(1, max_rr, detail_factor, model, x, all_data, file_path, experiment_id, dataset, folder_path, abbreviation, rmse_accepted_perfect)

# Type 1
for model, abbreviation in zip([IsolationForest, OneClassSVM, LocalOutlierFactor], ["IF", "OCSVM", "LOF"]):
    experiment_id += 1
    experiment_names[experiment_id] = f"Rejection based on {model.__name__} - Novelty Type II"
    metrics_results[experiment_id] = novelty_rejection(2, max_rr, detail_factor, model, x, all_data, file_path, experiment_id, dataset, folder_path, abbreviation, rmse_accepted_perfect)

# Type 3
for model, abbreviation in zip([IsolationForest, OneClassSVM, LocalOutlierFactor], ["IF", "OCSVM", "LOF"]):
    experiment_id += 1
    experiment_names[experiment_id] = f"Rejection based on {model.__name__} - Novelty Type III"
    metrics_results[experiment_id] = novelty_rejection(3, max_rr, detail_factor, model, x, all_data, file_path, experiment_id, dataset, folder_path, abbreviation, rmse_accepted_perfect)


# # Chapter 3: Training of the ITE Model
# treated_model, control_model = predictor_t_model(train_treated_x, train_treated_y, train_control_x, train_control_y, model_class, model_params)

# ## Chapter 3C: Predicting the ITE and related variables (y_t0 and y_t1)
# train_treated_y_pred, train_treated_y_prob, train_control_y_pred, train_control_y_prob = predictor_train_predictions(treated_model, control_model, train_treated_x, train_control_x)
# test_treated_y_pred, test_treated_y_prob, test_control_y_pred, test_control_y_prob = predictor_test_predictions(treated_model, control_model, test_treated_x, test_control_x)
# train_y_t1_pred, train_y_t0_pred, train_y_t1_prob, train_y_t0_prob, train_ite_prob, train_ite_pred = predictor_ite_predictions(treated_model, control_model, train_x)
# test_y_t1_pred, test_y_t0_pred, test_y_t1_prob, test_y_t0_prob, test_ite_prob, test_ite_pred = predictor_ite_predictions(treated_model, control_model, test_x)

# ## Chapter 3B: Training of the ITE Model
# ## create bootstraps
# treated_x = pd.concat([train_treated_x, test_treated_x], ignore_index=True).copy()
# treated_y = pd.concat([train_treated_y, test_treated_y], ignore_index=True).copy()
# control_x = pd.concat([train_control_x, test_control_x], ignore_index=True).copy()
# control_y = pd.concat([train_control_y, test_control_y], ignore_index=True).copy()
# x = pd.concat([train_x, test_x], ignore_index=True).copy()
# t = pd.concat([test_t, train_t], ignore_index=True).copy()
# potential_y = pd.concat([test_potential_y, train_potential_y], ignore_index=True).copy()
# ite = pd.concat([test_ite, train_ite], ignore_index=True).copy()

# # create bootstraps
# num_bootstraps = 200  # Replace with the desired number

# # List to store trained models
# treated_models = []
# control_models = []

# # Bootstrap loop
# for _ in range(num_bootstraps) :
#     print(f"Bootstrap: {_}")
#     # Sample with replacement from the treated and control groups
#     bootstrap_treated_x = treated_x.sample(n=len(treated_x), replace=True)
#     bootstrap_treated_y = treated_y.loc[bootstrap_treated_x.index]

#     bootstrap_control_x = control_x.sample(n=len(control_x), replace=True)
#     bootstrap_control_y = control_y.loc[bootstrap_control_x.index]

#     # Train ITE models for the current bootstrap sample
#     treated_model, control_model = predictor_t_model(
#         bootstrap_treated_x, bootstrap_treated_y,
#         bootstrap_control_x, bootstrap_control_y,
#         model_class, model_params
#     )

#     # Append trained models to the list
#     treated_models.append(treated_model)
#     control_models.append(control_model)

# ## Chapter 3C: Predicting the ITE and related variables (y_t0 and y_t1)
# ## Training and Testing predictions to evaluate individual models
# # Assuming you have the functions predictor_train_predictions and predictor_ite_predictions defined

# # List to store predictions and probabilities
# treated_y_preds, treated_y_probs, control_y_preds, control_y_probs = [], [], [], []
# y_t1_preds, y_t0_preds, y_t1_probs, y_t0_probs, ite_probs, ite_preds = [], [], [], [], [], []

# # Loop through the trained models
# for treated_model, control_model in zip(treated_models, control_models):
#     # Predictions
#     treated_y_pred, treated_y_prob, control_y_pred, control_y_prob = predictor_train_predictions(
#         treated_model, control_model, treated_x, control_x
#     )
    
#     treated_y_preds.append(treated_y_pred)
#     treated_y_probs.append(treated_y_prob)
#     control_y_preds.append(control_y_pred)
#     control_y_probs.append(control_y_prob)

#     # ITE predictions
#     y_t1_pred, y_t0_pred, y_t1_prob, y_t0_prob, ite_prob, ite_pred = predictor_ite_predictions(
#         treated_model, control_model, train_x
#     )

#     y_t1_preds.append(y_t1_pred)
#     y_t0_preds.append(y_t0_pred)
#     y_t1_probs.append(y_t1_prob)
#     y_t0_probs.append(y_t0_prob)
#     ite_probs.append(ite_prob)
#     ite_preds.append(ite_pred)

# # treated_y_pred = np.mean(treated_y_preds, axis=0)
# # treated_y_prob = np.mean(treated_y_probs, axis=0)
# # control_y_pred = np.mean(control_y_preds, axis=0)
# # control_y_prob = np.mean(control_y_probs, axis=0)

# # y_t1_pred = np.mean(y_t1_preds, axis=0)
# # y_t0_pred = np.mean(y_t0_preds, axis=0)
# # y_t1_prob = np.mean(y_t1_probs, axis=0)
# # y_t0_prob = np.mean(y_t0_probs, axis=0)
# # ite_prob = np.mean(ite_probs, axis=0)
# # ite_pred = np.mean(ite_preds, axis=0)


# # Chapter 5: Evaluate overall ITE Model: Performance

# ## Chapter 5B: Preprocessing of the test_set
# if treated_y_prob is not None and not treated_y_prob.isna().all():
#     # Calculate the difference between the max and min values for each array
#     y_t1_pred_diff = np.ptp(y_t1_preds, axis=0)
#     y_t0_pred_diff = np.ptp(y_t0_preds, axis=0)
#     y_t1_prob_diff = np.ptp(y_t1_probs, axis=0)
#     y_t0_prob_diff = np.ptp(y_t0_probs, axis=0)
#     ite_prob_diff = pd.DataFrame({'size_of_ci': np.ptp(ite_probs, axis=0)})
#     ite_pred_diff = pd.DataFrame({'Difference': np.ptp(ite_preds, axis=0)})

#     all_data = pd.concat([t, y_t1_pred, y_t1_prob, y_t0_pred, y_t0_prob, ite_pred, ite_prob, potential_y['y_t0'], potential_y['y_t1'], ite, ite_pred_diff, ite_prob_diff], axis=1)
#     test_set = pd.concat([test_t, test_y_t1_pred, test_y_t1_prob, test_y_t0_pred, test_y_t0_prob, test_ite_pred, test_ite_prob, test_potential_y["y_t0"], test_potential_y["y_t1"], test_ite], axis=1)
#     train_set = pd.concat([test_t, train_y_t1_pred, train_y_t1_prob, train_y_t0_pred, train_y_t0_prob, train_ite_pred, train_ite_prob, train_potential_y["y_t0"], train_potential_y["y_t1"], train_ite], axis=1)
# else:
#     ite_pred_diff = pd.DataFrame({'size_of_ci': np.ptp(ite_preds, axis=0)})

#     all_data = pd.concat([t, y_t1_pred, y_t0_pred, ite_pred, ite_prob, potential_y['y_t0'], potential_y['y_t1'], ite, ite_pred_diff], axis=1)
#     test_set = pd.concat([test_t, test_y_t1_pred, test_y_t0_pred, test_ite_pred, test_potential_y["y_t0"], test_potential_y["y_t1"], test_ite], axis=1)
#     train_set = pd.concat([test_t, train_y_t1_pred, train_y_t0_pred, train_ite_pred, train_potential_y["y_t0"], train_potential_y["y_t1"], train_ite], axis=1)

# if dataset == "TWINSC":
#     # Delete columns y_t1_pred and y_t0_pred, ite_pred
#     test_set = test_set.drop(['y_t1_pred', 'y_t0_pred', 'ite_pred'], axis=1)
#     # Rename columns y_t1_prob, y_t0_prob, ite_prob to y_t1_pred, y_t0_pred, ite_pred
#     test_set = test_set.rename(columns={'y_t1_prob': 'y_t1_pred', 'y_t0_prob': 'y_t0_pred', 'ite_prob': 'ite_pred'})

#     # Delete columns y_t1_pred and y_t0_pred, ite_pred
#     train_set = train_set.drop(['y_t1_pred', 'y_t0_pred', 'ite_pred'], axis=1)
#     # Rename columns y_t1_prob, y_t0_prob, ite_prob to y_t1_pred, y_t0_pred, ite_pred
#     train_set = train_set.rename(columns={'y_t1_prob': 'y_t1_pred', 'y_t0_prob': 'y_t0_pred', 'ite_prob': 'ite_pred'})

#     # Delete columns y_t1_pred and y_t0_pred, ite_pred
#     all_data = all_data.drop(['y_t1_pred', 'y_t0_pred', 'ite_pred'], axis=1)
#     # Rename columns y_t1_prob, y_t0_prob, ite_prob to y_t1_pred, y_t0_pred, ite_pred
#     all_data = all_data.rename(columns={'y_t1_prob': 'y_t1_pred', 'y_t0_prob': 'y_t0_pred', 'ite_prob': 'ite_pred'})


# #######################################################################################################################

# CHAPTER 7: REJECTION

# #######################################################################################################################
# # Rejection based on Isolation Forest (comparing T to UT and UT to T)
# experiment_id += 1
# experiment_name =  "Rejection based bootstrapping (confidence interval)"
# abbreviation = "BSCI"
# experiment_names.update({experiment_id: f"{experiment_name}"})

# # loop over all possible RR
# reject_rates = []
# rmse_accepted = []
# rmse_rejected = []

# # all_data.sort_values(by='amount_of_times_rejected', ascending=False)
# all_data = all_data.sort_values(by='size_of_ci', ascending=False).copy()
# all_data = all_data.reset_index(drop=True)

# detail_factor = 10 # 1 or 10
# for rr in range(1, 6*detail_factor):
#     num_to_set = int(rr / (100.0*detail_factor) * len(all_data)) # example: 60/100 = 0.6 * length of the data
      
#     all_data['ite_reject'] = all_data['ite_pred']
#     all_data['ite_reject'] = all_data['ite_reject'].astype(object)  # Change dtype of entire column

#     metrics_result = calculate_performance_metrics('ite', 'ite_reject', all_data, file_path)

#     if metrics_result is not None and 'Rejection Rate' in metrics_result:
#         reject_rates.append(metrics_result['Rejection Rate'])
#         print(f"RR: {rr / (100*detail_factor) }, RR: {metrics_result['Rejection Rate']}")
#     else:
#         reject_rates.append(None)

#     if metrics_result is not None and 'RMSE' in metrics_result:
#         rmse_accepted.append(metrics_result['RMSE'])
#     else:
#         rmse_accepted.append(None)

#     if metrics_result is not None and 'RMSE Rejected' in metrics_result:
#         rmse_rejected.append(metrics_result['RMSE Rejected'])
#     else:
#         rmse_rejected.append(None)

# # Graph with reject rate and rmse_accepted & rmse_rejected
        
# twolinegraph(reject_rates, "Reject Rate", rmse_accepted, "RMSE of Accepted Samples", "green", rmse_rejected, "RMSE of Rejected Samples", "red", f"Impact of Reject Rate on RMSE for {dataset}", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_rmse.png")
# onelinegraph(reject_rates, "Reject Rate", rmse_accepted, "RMSE of Accepted Samples", "green", f"Impact of Reject Rate on RMSE for {dataset}", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_rmse_accepted.png")
# onelinegraph(reject_rates, "Reject Rate", rmse_rejected, "RMSE of Rejected Samples", "red", f"Impact of Reject Rate on RMSE for {dataset}", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_rmse_rejected.png")

# # optimal model
# min_rmse = min(rmse_accepted)  # Find the minimum
# min_rmse_index = rmse_accepted.index(min_rmse)  # Find the index of the minimum RMSE
# optimal_reject_rate = max(reject_rates[min_rmse_index], 0.01)  # Get the rejection rate at the same index

# model = train_model(x, IsolationForest, contamination=optimal_reject_rate, random_state=42)

# all_data['ood'] = pd.Series(model.predict(x), name='ood')
# all_data['y_reject'] = all_data.apply(lambda row: True if row['ood'] == -1 else False, axis=1)
# all_data['ite_reject'] = all_data.apply(lambda row: "R" if row['y_reject'] else row['ite_pred'], axis=1)

# metrics_dict = calculate_all_metrics('ite', 'ite_reject', all_data, file_path, metrics_results, append_metrics_results=False, print=False)
# metrics_results[experiment_id] = metrics_dict

# #######################################################################################################################



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