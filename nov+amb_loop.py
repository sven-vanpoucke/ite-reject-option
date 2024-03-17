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
import time

## INIT
from models.helper import helper_output, helper_output_loop

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

def merge_test_train(train_treated_x, train_treated_y, train_control_x, train_control_y, test_treated_x, test_treated_y, test_control_x, test_control_y, train_x, train_t, train_y, test_x, test_t, test_y, train_ite, test_ite, train_potential_y, test_potential_y, x_scaling):
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
    return treated_x, treated_y, control_x, control_y, x, t, xt, train_xt, test_xt, y, ite, potential_y

def plot_summary(reject_rates_list, rmse_rank_accepted_list, experiment_ids_list, dataset, folder_path, plot_title, file_name):
    plt.figure(figsize=(10, 6))

    for i in range(1,10):
        plt.plot(reject_rates_list[i], rmse_rank_accepted_list[i], label=f"Experiment {experiment_ids_list[i]}")

    plt.xlabel('Reject Rate')
    plt.ylabel(f'{file_name}')
    plt.title(f'{plot_title} for {dataset}')
    plt.legend()
    plt.grid(True)

    plt.savefig(f"{folder_path}graph/{dataset}_All_{file_name}.png")
    plt.close()
    plt.cla()

def plot_canvas(reject_rates_list, rmse_rank_accepted_list, experiment_ids_list, dataset, folder_path, plot_title, file_name):
    plt.figure(figsize=(15, 15))  # Increase the figure size for a 3x3 grid

    # Create a 3x3 grid of subplots
    for i in range(1, 10):
        plt.subplot(3, 3, i)

        # Plot the corresponding graph
        plt.plot(reject_rates_list[i], rmse_rank_accepted_list[i], label=f"Experiment {experiment_ids_list[i]}")
        plt.xlabel('Reject Rate')
        plt.ylabel(f'{file_name}')
        plt.title(f'Experiment {experiment_ids_list[i]}')
        plt.legend()
        plt.grid(True)

    plt.suptitle(f'{plot_title} for {dataset}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust subplot layout

    # Save the combined plot as an image
    plt.savefig(f"{folder_path}graph/{dataset}_All_{file_name}.png")
    plt.close()
    plt.cla()

def canvas_change(reject_rates_list, metric_list, metric_list2, experiment_ids_list, dataset, folder_path, heuristic_cutoff_list, xlabel, ylabel, folder, y_min, y_max, title):
    plt.figure(figsize=(15, 15))  # Increase the figure size for a 3x3 grid
    
    # Create a 3x3 grid of subplots
    for i in range(1, 10):
        plt.subplot(3, 3, i)

        # Plot the corresponding graph
        plt.plot([rate * 100 for rate in reject_rates_list[i]], metric_list[i], color="green", label=f"Accepted Observations")

        plt.plot([rate * 100 for rate in reject_rates_list[i]], metric_list2[i], color="red", label=f"Rejected Observations")

        plt.ylim(y_min, y_max)  # Set x-axis range from 0 to 6
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
        
        if heuristic_cutoff_list[i]*100 < 15:
            plt.axvline(x=heuristic_cutoff_list[i]*100, color='green', linestyle='-', linewidth=1, label='Heuristic Optimal RR') # this line is the heuristical cut-off point
            # Add text label for the vertical line
            plt.text(heuristic_cutoff_list[i]*100+0.25, -4, 'Heuristic Optimal RR', rotation=90, color='green', verticalalignment='center')
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        plt.title(f'Experiment {experiment_ids_list[i]}')
        # plt.legend()
        # plt.grid(True)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust subplot layout

    # Save the combined plot as an image
    plt.savefig(f"{folder_path}overleaf/{folder}/{dataset}_All.png")
    plt.close()
    plt.cla()
    

def canvas_change_loop(reject_rates_list, metric_list, metric_name, experiment_ids_list, dataset, folder_path, heuristic_cutoff_list, xlabel, ylabel, folder, y_min, y_max, title, datasets):
    plt.figure(figsize=(15, 15))  # Increase the figure size for a 3x3 grid
    
    # Create a 3x3 grid of subplots
    for i in range(1, 10):
        plt.subplot(3, 3, i)
        plt.ylim(y_min, y_max)  # Set x-axis range from 0 to 6
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
        
        for dataset in datasets:
            # Plot the graph for the green color
            if dataset=="TWINSC":
                color = "green"
            else:
                color = "blue"
            plt.plot([rate * 100 for rate in reject_rates_list[dataset][i]], [result.get(metric_name, None) for result in metric_list[dataset][i]], color=color, label=f"{dataset}")

            # Check if heuristic cutoff is less than 15
            if heuristic_cutoff_list[dataset][i] * 100 < 15:
                plt.axvline(x=heuristic_cutoff_list[dataset][i]*100, color=color, linestyle=':', linewidth=1)
                plt.text(heuristic_cutoff_list[dataset][i]*100 + 0.25, y_min+(y_max-y_min)*0.25, 'Heuristic Optimal RR', rotation=90, color=color, verticalalignment='center')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        plt.title(f'Experiment {experiment_ids_list[dataset][i]}')
        # plt.grid(True)
        plt.legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust subplot layout

    # Save the combined plot as an image
    plt.savefig(f"{folder_path}overleaf/{folder}/All_All.png")
    plt.close()
    plt.cla()

    plt.figure(figsize=(15, 15))  # Increase the figure size for a 2x1 grid

    # Create a 2x1 grid of subplots
    x = 0
    for i in [5,6,7,10]:
        x += 1
        plt.subplot(2, 2, x)  # Configure subplot as 2x1
    
        plt.ylim(y_min, y_max)  # Set x-axis range from 0 to 6
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
        
        for dataset in datasets:
            # Plot the graph for the green color
            if dataset=="TWINSC":
                color = "green"
            else:
                color = "blue"
            plt.plot([rate * 100 for rate in reject_rates_list[dataset][i]], [result.get(metric_name, None) for result in metric_list[dataset][i]], color=color, label=f"{dataset}")

            # Check if heuristic cutoff is less than 15
            if heuristic_cutoff_list[dataset][i] * 100 < 15:
                plt.axvline(x=heuristic_cutoff_list[dataset][i]*100, color=color, linestyle=':', linewidth=1)
                plt.text(heuristic_cutoff_list[dataset][i]*100 + 0.25, -4, 'Heuristic Optimal RR', rotation=90, color=color, verticalalignment='center')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        plt.title(f'Experiment {experiment_ids_list[dataset][i]}')
        # plt.grid(True)
        plt.legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust subplot layout

    # Save the combined plot as an image
    plt.savefig(f"{folder_path}overleaf/{folder}/Nov_vs_Amb.png")
    plt.close()
    plt.cla()


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

## Parameters
### To choose
datasets = ["TWINSC", "IHDP"] # Choose out of TWINS or TWINSC (if you want TWINS to be treated as continuous instead of classification) or LALONDE or IHDP
# datasets = ["IHDP"]
psm = False
### Rejection
detail_factor = 1 # 1 (no extra detail) or 10 (extra detail)
max_rr = 15 # (number between 1 and 49)
x_scaling = False # True or False

### Not to choose
folder_path = 'output/'
text_folder_path = 'output/text/'
metrics_results = {}
experiment_names = {}
timestamp, file_name, file_path = helper_output_loop(folder_path=text_folder_path)
all_data_list = []
xt_list = []
x_list = []
train_forest_model_list = []
start_time = time.time()


# Chapter 1: Initialization
for dataset in datasets:
    ## Parameters
    ### To choose
    dataset = dataset 

    # Chapter 2: Preprocessing
    ## Retrieve Data
    if dataset == "TWINS" or dataset == "TWINSC":
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
    
    treated_x, treated_y, control_x, control_y, x, t, xt, train_xt, test_xt, y, ite, potential_y = merge_test_train(train_treated_x, train_treated_y, train_control_x, train_control_y, test_treated_x, test_treated_y, test_control_x, test_control_y, train_x, train_t, train_y, test_x, test_t, test_y, train_ite, test_ite, train_potential_y, test_potential_y, x_scaling)

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
    train_ite_pred_forest = pd.Series(train_forest_model.predict(train_xt, quantiles=[0.5]), name='ite_pred')
    test_ite_pred_forest = pd.Series(train_forest_model.predict(test_xt, quantiles=[0.5]), name='ite_pred')

    ## RandomForestQuantileRegressor - all_set
    ite_pred_forest = pd.Series(forest_model.predict(xt, quantiles=[0.5]), name='ite_pred')

    # Chapter 5: Process Data
    ## Merge Outcomes
    if train_treated_y_prob is not None and not train_treated_y_prob.isna().all():
        test_set = pd.concat([test_t, test_y_t1_pred, test_y_t1_prob, test_y_t0_pred, test_y_t0_prob, test_ite_pred, test_ite_prob, test_potential_y["y_t0"], test_potential_y["y_t1"], test_ite], axis=1)
        train_set = pd.concat([test_t, train_y_t1_pred, train_y_t1_prob, train_y_t0_pred, train_y_t0_prob, train_ite_pred, train_ite_prob, train_potential_y["y_t0"], train_potential_y["y_t1"], train_ite], axis=1)
        all_set = pd.concat([t, y_t1_pred, y_t1_prob, y_t0_pred, y_t0_prob, ite_pred, ite_prob, potential_y["y_t0"], potential_y["y_t1"], ite], axis=1).copy()
        
        train_forest_set = pd.concat([train_t, train_ite_pred_forest, train_ite], axis=1).copy()
        test_forest_set = pd.concat([test_t, test_ite_pred_forest, test_ite], axis=1).copy()
        forest_set = pd.concat([t, ite_pred, ite], axis=1).copy()
    else:
        test_set = pd.concat([test_t, test_y_t1_pred, test_y_t0_pred, test_ite_pred, test_potential_y["y_t0"], test_potential_y["y_t1"], test_ite], axis=1)
        train_set = pd.concat([test_t, train_y_t1_pred, train_y_t0_pred, train_ite_pred, train_potential_y["y_t0"], train_potential_y["y_t1"], train_ite], axis=1)
        all_set = pd.concat([t, y_t1_pred, y_t0_pred, ite_pred, potential_y["y_t0"], potential_y["y_t1"], ite], axis=1).copy()
        
        train_forest_set = pd.concat([train_t, train_ite_pred, train_ite], axis=1).copy()
        test_forest_set = pd.concat([test_t, test_ite_pred, test_ite], axis=1).copy()
        forest_set = pd.concat([t, ite_pred_forest, ite], axis=1).copy()

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
    
    # Rejection
    all_data['ite_reject'] = all_data.apply(lambda row: row['ite_pred'], axis=1)
    all_data['se'] = (all_data['ite'] - all_data['ite_pred']) ** 2

    all_data_list.append(all_data)
    x_list.append(x)
    xt_list.append(xt)
    train_forest_model_list.append(train_forest_model)

#######################################################################################################################
# CHAPTER 7: REJECTION
i = -1 # (dataset 0 first)
reject_rates_list = {}
rmse_rank_accepted_list = {}
rmse_rank_weighted_accepted_list = {}
sign_error_accepted_list = {}
signerror_weighted_accepted_list = {}
experiment_ids_list = {}
rmse_accepted_list = {}
rmse_change_accepted_list = {}
heuristic_cutoff_list = {}
sign_error_change_accepted_list = {}
rmse_rank_change_accepted_list = {}
rmse_rank_weighted_change_accepted_list = {}
metrics_results_list_global = {}

for dataset in datasets:
    experiment_id = -2 # reset experiment_id for each dataset
    metrics_results[dataset] = {}
    i += 1
    ## Output
    with open(file_path, 'a') as file:
        file.write(f"\nREJECTION for {dataset}\n\n")

    #######################################################################################################################
    # Rejection Architecture
    architecture="Separated Architecture"

    #######################################################################################################################
    # No Rejection
    experiment_id += 1
    experiment_name = "No Rejector - Baseline Model"
    experiment_names.update({experiment_id: f"{experiment_name}"})

    # Calculate the performance metrics
    metrics_dict = calculate_all_metrics('ite', 'ite_reject', all_data_list[i], file_path, metrics_results, append_metrics_results=False, print=False)        
    metrics_results[dataset].update({experiment_id: metrics_dict})

    #######################################################################################################################
    # Type 0 - Perfect Rejection
    experiment_id += 1
    experiment_name =  "Perfect Rejection"
    abbreviation = "Perfect"
    experiment_names.update({experiment_id: f"{experiment_name}"})

    rr_perfect = []
    rmse_accepted_perfect = []
    rmse_rejected_perfect = []
    change_rmse = []
    
    all_data_list[i] = all_data_list[i].sort_values(by='se', ascending=False).copy()
    all_data_list[i] = all_data_list[i].reset_index(drop=True)

    for rr in range(1, max_rr*detail_factor):
        num_to_set = int(rr / (100.0*detail_factor) * len(all_data_list[i])) # example: 60/100 = 0.6 * length of the data

        all_data_list[i]['ite_reject'] = all_data_list[i]['ite_pred']
        all_data_list[i]['ite_reject'] = all_data_list[i]['ite_reject'].astype(object)  # Change dtype of entire column
        all_data_list[i].loc[:num_to_set -1, 'ite_reject'] = 'R'

        metrics_result = calculate_performance_metrics('ite', 'ite_reject', all_data_list[i], file_path)

        if metrics_result:
            rr_perfect.append(metrics_result.get('Rejection Rate', None))
            rmse_accepted_perfect.append(metrics_result.get('RMSE Accepted', None))
            rmse_rejected_perfect.append(metrics_result.get('RMSE Rejected', None))
        else:
            rr_perfect.append(None)
            rmse_accepted_perfect.append(None)
            rmse_rejected_perfect.append(None)

    # Graph with reject rate and rmse_accepted & rmse_rejected
    twolinegraph(rr_perfect, "Reject Rate", rmse_accepted_perfect, "RMSE of Accepted Samples", "green", rmse_rejected_perfect, "RMSE of Rejected Samples", "red", f"Impact of Reject Rate on RMSE for {dataset}", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_rmse.png")
    onelinegraph(rr_perfect, "Reject Rate", rmse_accepted_perfect, "RMSE of Accepted Samples", "green", f"Impact of Reject Rate on RMSE for {dataset}", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_rmse_accepted.png")
    onelinegraph(rr_perfect, "Reject Rate", rmse_rejected_perfect, "RMSE of Rejected Samples", "red", f"Impact of Reject Rate on RMSE for {dataset}", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_rmse_rejected.png")

    # optimal model
    min_rmse = min(rmse_accepted_perfect)  # Find the minimum
    min_rmse_index = rmse_accepted_perfect.index(min_rmse)  # Find the index of the minimum RMSE
    optimal_reject_rate = rr_perfect[min_rmse_index]  # Get the rejection rate at the same index

    all_data_list[i]['ite_reject'] = all_data_list[i]['ite_pred']
    all_data_list[i]['ite_reject'] = all_data_list[i]['ite_reject'].astype(object)  # Change dtype of entire column
    all_data_list[i].loc[:num_to_set -1, 'ite_reject'] = 'R'

    metrics_dict = calculate_all_metrics('ite', 'ite_reject', all_data_list[i], file_path, metrics_results, append_metrics_results=False, print=False)
    metrics_results[experiment_id] = metrics_dict
    list_results = []
    
    reject_rates_list[dataset] = {}
    rmse_rank_accepted_list[dataset] = {}
    rmse_rank_weighted_accepted_list[dataset] = {}
    sign_error_accepted_list[dataset] = {}
    signerror_weighted_accepted_list[dataset] = {}
    experiment_ids_list[dataset] = {}
    rmse_accepted_list[dataset] = {}
    rmse_change_accepted_list[dataset] = {}
    heuristic_cutoff_list[dataset] = {}
    sign_error_change_accepted_list[dataset] = {}
    rmse_rank_change_accepted_list[dataset] = {}
    rmse_rank_weighted_change_accepted_list[dataset] = {}
    metrics_results_list_global[dataset] = {}
    # Type 1
    for model, abbreviation in zip([IsolationForest, OneClassSVM, LocalOutlierFactor], ["IF", "OCSVM", "LOF"]):
        experiment_id += 1
        experiment_names[experiment_id] = f"Rejection based on {model.__name__} (train data) - Novelty Type I"
        metrics_dict, reject_rates, heuristic_cutoff, metrics_results_list = novelty_rejection(1, max_rr, detail_factor, model, x_list[i], all_data_list[i], file_path, experiment_id, dataset, folder_path, abbreviation, rmse_accepted_perfect, give_details=True)
        metrics_results[dataset].update({experiment_id: metrics_dict})
        reject_rates_list[dataset].update({experiment_id: reject_rates})
        heuristic_cutoff_list[dataset].update({experiment_id: heuristic_cutoff})
        metrics_results_list_global[dataset].update({experiment_id: metrics_results_list})
        experiment_ids_list[dataset].update({experiment_id: experiment_id})
    
    # Type 2
    for model, abbreviation in zip([IsolationForest, OneClassSVM, LocalOutlierFactor], ["IF", "OCSVM", "LOF"]):
        experiment_id += 1
        experiment_names[experiment_id] = f"Rejection based on {model.__name__} (train data) - Novelty Type II"
        metrics_dict, reject_rates, heuristic_cutoff, metrics_results_list = novelty_rejection(2, max_rr, detail_factor, model, x_list[i], all_data_list[i], file_path, experiment_id, dataset, folder_path, abbreviation, rmse_accepted_perfect, give_details=True)
        metrics_results[dataset].update({experiment_id: metrics_dict})
        reject_rates_list[dataset].update({experiment_id: reject_rates})
        heuristic_cutoff_list[dataset].update({experiment_id: heuristic_cutoff})
        metrics_results_list_global[dataset].update({experiment_id: metrics_results_list})
        experiment_ids_list[dataset].update({experiment_id: experiment_id})


    # Type 3
    for model, abbreviation in zip([IsolationForest, OneClassSVM, LocalOutlierFactor], ["IF", "OCSVM", "LOF"]):
        experiment_id += 1
        experiment_names[experiment_id] = f"Rejection based on {model.__name__} (train data) - Novelty Type III"
        metrics_dict, reject_rates, heuristic_cutoff, metrics_results_list = novelty_rejection(3, max_rr, detail_factor, model, x_list[i], all_data_list[i], file_path, experiment_id, dataset, folder_path, abbreviation, rmse_accepted_perfect, give_details=True)
        metrics_results[dataset].update({experiment_id: metrics_dict})
        reject_rates_list[dataset].update({experiment_id: reject_rates})
        heuristic_cutoff_list[dataset].update({experiment_id: heuristic_cutoff})
        metrics_results_list_global[dataset].update({experiment_id: metrics_results_list})
        experiment_ids_list[dataset].update({experiment_id: experiment_id})


    #######################################################################################################################
    # Ambiguity
    #######################################################################################################################

    # Type 1
    experiment_id += 1
    model = "RandomForestQuantileRegressor"
    abbreviation = "RFQR"
    experiment_names[experiment_id] = f"Rejection based on RandomForestQuantileRegressor - Ambiguity Type I"
    # metrics_dict, reject_rates, rmse_accepted, rmse_rank_accepted, sign_error_accepted, rmse_rank_weighted_accepted, rmse_change_accepted, heuristic_cutoff, sign_error_change_accepted, rmse_rank_weighted_change_accepted, rmse_rank_change_accepted = ambiguity_rejection(1, max_rr, detail_factor, train_forest_model_list[i], xt_list[i], all_data_list[i], file_path, experiment_id, dataset, folder_path, abbreviation, rmse_accepted_perfect, give_details=True)
    metrics_dict, reject_rates, heuristic_cutoff, metrics_results_list = ambiguity_rejection(1, max_rr, detail_factor, train_forest_model_list[i], xt_list[i], all_data_list[i], file_path, experiment_id, dataset, folder_path, abbreviation, rmse_accepted_perfect, give_details=True)
    
    metrics_results[dataset].update({experiment_id: metrics_dict})
    reject_rates_list[dataset].update({experiment_id: reject_rates})
    heuristic_cutoff_list[dataset].update({experiment_id: heuristic_cutoff})
    metrics_results_list_global[dataset].update({experiment_id: metrics_results_list})
    experiment_ids_list[dataset].update({experiment_id: experiment_id})


#######################################################################################################################
canvas_change_loop(reject_rates_list, metrics_results_list_global, "RMSE Change (%)", experiment_ids_list, dataset, folder_path, heuristic_cutoff_list, 'Reject Rate (%)','RMSE Deviation from No-Rejection (%)', 'rmse', -11, 3, f'Impact of Rejection on the RMSE of the TE', datasets)
canvas_change_loop(reject_rates_list, metrics_results_list_global, "Similarity 50% Improved (%)", experiment_ids_list, dataset, folder_path, heuristic_cutoff_list, 'Reject Rate (%)','Similarity Deviation from No-Rejection (%)', 'similarity', -10, 6, f'Impact of Rejection on the similarity of the TE', datasets)

i = -1
for dataset in datasets:
    i += 1
    # plot_summary(reject_rates_list[dataset], rmse_accepted_list[dataset], experiment_ids_list[dataset], dataset, folder_path, "Impact RR on RMSE Accepted", "RMSEAccepted")
    # plot_summary(reject_rates_list[dataset], [result.get('RMSE Accepted', None) for result in metrics_results_list_global[dataset][2]], experiment_ids_list[dataset], dataset, folder_path, "Impact RR on RMSE Accepted", "RMSEAccepted")
    # plot_summary(reject_rates_list[dataset], rmse_rank_accepted_list[dataset], experiment_ids_list[dataset], dataset, folder_path, "Impact RR on RMSE Rank Accepted", "RMSERankAccepted")
    # plot_summary(reject_rates_list[dataset], sign_error_accepted_list[dataset], experiment_ids_list[dataset], dataset, folder_path, "Impact RR on Sign Error Accepted", "SignErrorAccepted")
    # plot_summary(reject_rates_list[dataset], rmse_rank_weighted_accepted_list[dataset], experiment_ids_list[dataset], dataset, folder_path, "Impact RR on RMSE Rank Weighted Accepted", "RMSERankWeightedAccepted")
    # plot_canvas(reject_rates_list[dataset], rmse_accepted_list[dataset], experiment_ids_list[dataset], dataset, folder_path, "Impact RR on RMSE Accepted", "RMSEAccepted")
    # # 9x9 plots:
    # canvas_change(reject_rates_list[dataset], [result.get('RMSE Change (%)', None) for result in metrics_results_list_global[dataset]], [result.get('RMSE Change (%)', None) for result in metrics_results_list_global[dataset]], experiment_ids_list[dataset], dataset, folder_path, heuristic_cutoff_list[dataset], 'Reject Rate (%)','RMSE Deviation from No-Rejection (%)', 'rmse', -9, 3, f'Impact of Rejection on the RMSE of the TE ({dataset})')
    
    canvas_change(reject_rates_list[dataset], [result.get('Similarity 50% Accepted (%)', None) for result in metrics_results_list_global[dataset]], [result.get('Similarity 50% Rejected (%)', None) for result in metrics_results_list_global[dataset]], experiment_ids_list[dataset], dataset, folder_path, heuristic_cutoff_list[dataset], 'Reject Rate (%)','RMSE Deviation from No-Rejection (%)', 'signaccuracy', 0, 100, f'Impact of Rejection on the RMSE of the TE ({dataset})')
    canvas_change(reject_rates_list[dataset], [result.get('Sign Accuracy Accepted (%)', None) for result in metrics_results_list_global[dataset]], [result.get('Sign Accuracy Rejected (%)', None) for result in metrics_results_list_global[dataset]], experiment_ids_list[dataset], dataset, folder_path, heuristic_cutoff_list[dataset], 'Reject Rate (%)','RMSE Deviation from No-Rejection (%)', 'similarity', 0, 100, f'Impact of Rejection on the RMSE of the TE ({dataset})')
    canvas_change(reject_rates_list[dataset], [result.get('Weighted Sign Accuracy Accepted (%)', None) for result in metrics_results_list_global[dataset]],[result.get('Weighted Sign Accuracy Rejected (%)', None) for result in metrics_results_list_global[dataset]], experiment_ids_list[dataset], dataset, folder_path, heuristic_cutoff_list[dataset], 'Reject Rate (%)','RMSE Deviation from No-Rejection (%)', 'weightedsignaccuracy', 0, 100, f'Impact of Rejection on the RMSE of the TE ({dataset})')
    canvas_change(reject_rates_list[dataset], [result.get('Adverse Effect Accuracy Accepted (%)', None) for result in metrics_results_list_global[dataset]],[result.get('Adverse Effect Accuracy Rejected (%)', None) for result in metrics_results_list_global[dataset]], experiment_ids_list[dataset], dataset, folder_path, heuristic_cutoff_list[dataset], 'Reject Rate (%)','RMSE Deviation from No-Rejection (%)', 'adverseeffect', 0, 100, f'Impact of Rejection on the RMSE of the TE ({dataset})')
    canvas_change(reject_rates_list[dataset], [result.get('Positive Potential Accuracy Accepted (%)', None) for result in metrics_results_list_global[dataset]],[result.get('Positive Potential Accuracy Rejected (%)', None) for result in metrics_results_list_global[dataset]], experiment_ids_list[dataset], dataset, folder_path, heuristic_cutoff_list[dataset], 'Reject Rate (%)','RMSE Deviation from No-Rejection (%)', 'positivepotential', 0, 100, f'Impact of Rejection on the RMSE of the TE ({dataset})')





    uplift_scores = all_data_list[i]['ite']
    # Sort the data
    sorted_indices = np.argsort(uplift_scores)
    sorted_uplifts = uplift_scores[sorted_indices]
    # Divide into quantiles (e.g., 10 quantiles)
    quantiles = np.linspace(0, 1, 11)
    quantile_values = np.percentile(sorted_uplifts, quantiles * 100)
    # Calculate cumulative uplift
    cumulative_uplift = np.cumsum(quantile_values)

    # Plot the uplift curve
    plt.plot(quantiles, cumulative_uplift, marker='o', label="Reality")
    
    uplift_scores = all_data_list[i]['ite_pred']
    # Sort the data
    sorted_indices = np.argsort(uplift_scores)
    sorted_uplifts = uplift_scores[sorted_indices]
    # Divide into quantiles (e.g., 10 quantiles)
    quantiles = np.linspace(0, 1, 11)
    quantile_values = np.percentile(sorted_uplifts, quantiles * 100)
    # Calculate cumulative uplift
    cumulative_uplift = np.cumsum(quantile_values)

    plt.plot(quantiles, cumulative_uplift, linestyle='--', marker='o', label="Estimated")

    plt.xlabel('Proportion of Population')
    plt.ylabel('Cumulative Uplift')
    plt.title('Uplift Curve')
    plt.grid(True)
    plt.legend()

    plt.savefig(f"{folder_path}graph/uplift/{dataset}_real_uplift.png")
    plt.close()
    plt.cla()


    #######################################################################################################################
    metrics_results[dataset] = pd.DataFrame.from_dict(metrics_results[dataset], orient='index')

    # Chapter 8: Output to file
    with open(file_path, 'a') as file:

        file.write("\n\nTable of all_data (First 5 rows)\n")
        file.write(tabulate(all_data_list[i].head(5), headers='keys', tablefmt='pretty', showindex=False))
        
        file.write ("\n")
        for exp_number, description in experiment_names.items():
            file.write(f"# Experiment {exp_number}: {description}\n")

        file.write("\nTable of results of the experiments\n")
        file.write(tabulate(metrics_results[dataset].transpose(), headers='keys', tablefmt='rounded_grid', showindex=True))



end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime} seconds")