from models.evaluators.performance_evaluator import calculate_performance_metrics
from models.evaluators.evaluator import calculate_all_metrics
from math import sqrt
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import pandas as pd

# Function to train a model equal to the one in one_class_classification_rejector.py
def train_model(x, model_class, **model_params):
    # Train the model
    model = model_class(**model_params)
    model.fit(x)
    return model

# Function to determine if a point is out of distribution based on a threshold
def is_too_far(distance, threshold):
    return distance > threshold


# Objective function
def calculate_objective(threshold_distance, *args):
    data = args[0]
    file_path = args[1]
    distances = args[2]
    key_metric = args[3]
    minmax = args[4]
    
    data['ood'] = distances.apply(is_too_far, threshold=threshold_distance)
    data['ite_reject'] = data.apply(lambda row: "R" if row['ood'] else row['ite_pred'], axis=1)
    metrics_dict = calculate_performance_metrics('ite', 'ite_reject', data, file_path)
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
        file.write(f"\n Current value for the metric: {round(metric,2)} with threshold: {round(threshold_distance,2)} and rejection rate: {round(metrics_dict['Rejection Rate'],2)}")
        # file.write("\n")
        # file.write(tabulate(data.head(50), headers='keys', tablefmt='pretty', showindex=False))
    return metric

def onelinegraph(x, x_label, y, y_label, color, title, folder, xmin=0, xmax=0, ymin=0, ymax=0):
    # Graph with reject rate and RMSE of Accepted Samples
    plt.plot(x, y, color=color, label=y_label)
    plt.xlabel(x_label)
    plt.title(title)
    if xmin != xmax:
        plt.xlim(xmin, xmax)  # Set x-axis range from 0 to 6
    if ymin != ymax:
        plt.ylim(ymin, ymax)  # Set y-axis range from 0 to 25
    plt.legend()
    plt.savefig(folder)
    plt.close()
    plt.cla()

def onelinechangegraph(x, x_label, y, y_label, color, title, folder, xmin=0, xmax=0, ymin=0, ymax=0):
    # Graph with reject rate and RMSE of Accepted Samples
    plt.plot(x, y, color=color, label=y_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title(title)
    plt.ylim(-9, 3)  # Set x-axis range from 0 to 6
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    # plt.xlim(xmin, xmax)  # Set y-axis range from 0 to 25
    # plt.legend()
    plt.savefig(folder)
    plt.close()
    plt.cla()


def twolinegraph(x, x_label, y, y_label, color, y2, y2_label, color2, title, folder):
    # Graph with reject rate and RMSE of Accepted Samples
    plt.plot(x, y, color=color, label=y_label)
    plt.plot(x, y2, color=color2, label=y2_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend()
    plt.savefig(folder)
    plt.close()
    plt.cla()

def histogram(values, xlabel, ylabel, title, folder, lowest_rejected_value, mean, plusstd, plus2std):
    plt.hist(values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.axvline(mean, color='blue', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(lowest_rejected_value, color='red', linestyle='dashed', linewidth=2, label='Lowest R')
    plt.axvline(plusstd, color='green', linestyle='dashed', linewidth=2, label='Mean + 1 Std Dev')
    # plt.axvline(plus2std, color='green', linestyle='dashed', linewidth=2, label='Mean + 2 Std Dev')
    plt.legend()
    plt.title(title)
    plt.savefig(folder)
    plt.close()
    plt.cla()

def f(type, contamination, t_x, ut_x, t_data, ut_data, detail_factor, model_name, all_data):
    contamination /= (100 * detail_factor)  # max of 0.5
    if model_name == IsolationForest:
        t_model = train_model(t_x, IsolationForest, contamination=contamination, random_state=42)
        ut_model = train_model(ut_x, IsolationForest, contamination=contamination, random_state=42)
    elif model_name == OneClassSVM:
        t_model = train_model(t_x, OneClassSVM, nu=contamination)
        ut_model = train_model(ut_x, OneClassSVM, nu=contamination)
    elif model_name == LocalOutlierFactor:
        t_model = train_model(t_x, LocalOutlierFactor, contamination=contamination, novelty=True)
        ut_model = train_model(ut_x, LocalOutlierFactor, contamination=contamination, novelty=True)

    if type == 2:
        t_data['ood'] = pd.Series(ut_model.predict(t_x), name='ood').copy()
        ut_data['ood'] = pd.Series(t_model.predict(ut_x), name='ood').copy()
        all_data['amount_of_times_rejected_new'] = all_data.apply(lambda row: 1 if row['ood'] == -1 else 0, axis=1)
    if type == 3:
        ut_data['ood-ut'] = pd.Series(t_model.predict(ut_x), name='ood').copy()
        ut_data['ood-t'] = pd.Series(t_model.predict(ut_x), name='ood').copy()
        # ut_data['ood'] = (ut_data['ood-ut'] + ut_data['ood-t']) / 2
        ut_data['ood'] = ut_data[['ood-ut', 'ood-t']].max(axis=1)

    all_data = pd.concat([t_data, ut_data], ignore_index=True).copy()
    all_data['amount_of_times_rejected_new'] = all_data.apply(lambda row: 1 if row['ood'] == -1 else 0, axis=1)
    return all_data['amount_of_times_rejected_new']

def ambiguity_rejection(type_nr, max_rr, detail_factor, model, xt, all_data, file_path, experiment_id, dataset, folder_path, abbreviation, rmse_accepted_perfect=[], give_details=False):
    reject_rates = []
    rmse_accepted = []
    rmse_rejected = []
    rmse_change_accepted = []
    optimal_reject_rate = None
    metrics_results_list = []
    min_rmse = float('inf')  # Set to positive infinity initially

    optimal_model = None

    # sign_error_accepted = []
    # rmse_rank_accepted = []
    # rmse_rank_weighted_accepted = []
    # sign_error_change_accepted = []
    # rmse_rank_weighted_change_accepted = []
    # rmse_rank_change_accepted = []
    if type_nr == 1:

        # y_lower = model.predict(xt, quantiles=[0.025])
        # y_upper = model.predict(xt, quantiles=[0.975])

        # y_lower2 = model.predict(xt, quantiles=[0.05])
        # y_upper2 = model.predict(xt, quantiles=[0.95])

        # y_lower3 = model.predict(xt, quantiles=[0.10])
        # y_upper3 = model.predict(xt, quantiles=[0.90])

        # y_lower4 = model.predict(xt, quantiles=[0.15])
        # y_upper4 = model.predict(xt, quantiles=[0.85])

        # size_of_ci = ((y_upper - y_lower) + (y_upper2 - y_lower2) + (y_upper3 - y_lower3) + (y_upper4 - y_lower4)) /4 # confidence interval
        
        # all_data['size_of_ci'] = size_of_ci
        # all_data = all_data.sort_values(by='size_of_ci', ascending=False).copy()
        # all_data = all_data.reset_index(drop=True)

        quantiles = [0.025, 0.05, 0.10, 0.15]
        y_values = [model.predict(xt, quantiles=[q]) for q in quantiles]
        size_of_ci = sum([(y[1] - y[0]) for y in y_values]) / len(quantiles)
        all_data['size_of_ci'] = size_of_ci
        all_data = all_data.sort_values(by='size_of_ci', ascending=False).reset_index(drop=True)

        all_data['Ambiguity Score'] = all_data['size_of_ci']
        all_data['Ambiguity Score Normalized'] = (all_data['Ambiguity Score'] - all_data['Ambiguity Score'].min()) / (all_data['Ambiguity Score'].max() - all_data['Ambiguity Score'].min())

        for rr in range(1, max_rr * detail_factor):
            num_to_set = int(rr / (100.0 * detail_factor) * len(all_data)) # example: 60/100 = 0.6 * length of the data
            all_data['ite_reject'] = all_data['ite_pred'].astype(object)
            if num_to_set != 0:
                all_data.loc[:num_to_set - 1, 'ite_reject'] = 'R'

            metrics_result = calculate_performance_metrics('ite', 'ite_reject', all_data, file_path)
            if metrics_result:
                reject_rates.append(metrics_result.get('Rejection Rate', None))
                rmse_accepted.append(metrics_result.get('RMSE Accepted', None))
                rmse_rejected.append(metrics_result.get('RMSE Rejected', None))
                rmse_original = metrics_result.get('RMSE Original', None)
                improvement = (metrics_result.get('RMSE Accepted', None) - rmse_original) / rmse_original * 100
                rmse_change_accepted.append(improvement)

                # sign_error_accepted.append(metrics_result.get('Sign Error Accepted (%)', None))
                # sign_error_change_accepted.append(metrics_result.get('Sign Error Change (%)', 0))
                # rmse_rank_accepted.append(metrics_result.get('RMSE Rank Accepted', None))
                # rmse_rank_weighted_accepted.append(metrics_result.get('RMSE Rank Weighted Accepted', None))
                # rmse_rank_change_accepted.append(metrics_result.get('RMSE Rank Change (%)', 0))
                # rmse_rank_weighted_change_accepted.append(metrics_result.get('RMSE Rank Weighted Change (%)', 0))
                metrics_results_list.append(metrics_result)

            else:
                reject_rates.append(None)
                rmse_accepted.append(None)
                rmse_rejected.append(None)
    
    # Graphs
    twolinegraph(reject_rates, "Reject Rate", rmse_accepted, "RMSE of Accepted Samples", "green", rmse_rejected, "RMSE of Rejected Samples", "red", f"Impact of Reject Rate on RMSE for {dataset}", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_rmse_accepted.png")
    # twolinegraph(reject_rates, "Reject Rate", rmse_accepted, f"Experiment {experiment_id} ", "blue", rmse_accepted_perfect, "Perfect Rejection", "green", f"RMSE of Accepted Samples ({dataset})", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_PerfRMSEe.png")
    # onelinegraph(reject_rates, "Reject Rate", rmse_change_accepted, "Change of RMSE (%)", "green", f"Impact of Reject Rate on change of RMSE for {dataset}", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_ChangeRMSE.png")
    onelinegraph(reject_rates, "Reject Rate", rmse_accepted, "RMSE of Accepted Samples", "green", f"Impact of Reject Rate on RMSE for {dataset}", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_rmse_accepted.png")
    onelinegraph(reject_rates, "Reject Rate", rmse_rejected, "RMSE of Rejected Samples", "red", f"Impact of Reject Rate on RMSE for {dataset}", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_rmse_rejected.png")
    onelinegraph(reject_rates, "Reject Rate", [result.get('Sign Error Accepted (%)', None) for result in metrics_results_list], "Sign Error Accepted (%)", "green", f"Impact of Reject Rate on Sign Error {dataset} ({experiment_id})", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_SignError_accepted.png")
    onelinegraph(reject_rates, "Reject Rate", [result.get('RMSE Rank Accepted', None) for result in metrics_results_list], "RMSE Rank Accepted", "green", f"Impact of Reject Rate on Rank Error {dataset} ({experiment_id})", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_RankError_accepted.png")
    onelinegraph(reject_rates, "Reject Rate", [result.get('RMSE Rank Weighted Accepted', None) for result in metrics_results_list], "RMSE Rank Weighted Accepted", "green", f"Impact of Reject Rate on Weighted Rank Error {dataset} ({experiment_id})", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_RankWeightedError_accepted.png")

    # optimal model 
    min_rmse = min(rmse_accepted)  # Find the minimum
    min_rmse_index = rmse_accepted.index(min_rmse)  # Find the index of the minimum RMSE
    optimal_reject_rate = reject_rates[min_rmse_index]  # Get the rejection rate at the same index

    filtered_data = all_data[all_data['Ambiguity Score'] != 0]
    std_dev = filtered_data['Ambiguity Score'].std()
    mean = filtered_data['Ambiguity Score'].mean()
    plusstd = filtered_data['Ambiguity Score'].mean() + std_dev
    plus2std = filtered_data['Ambiguity Score'].mean() + 2* std_dev
    heuristic_cutoff = len(all_data[all_data['Ambiguity Score'] > plusstd]) / len(all_data)

    lowest_rejected_value = all_data.loc[all_data['ite_reject'] == 'R', 'Ambiguity Score Normalized'].iloc[-1]
    std_dev_normalized = filtered_data['Ambiguity Score'].std()
    mean_normalized = filtered_data['Ambiguity Score Normalized'].mean()
    plusstd_normalized = filtered_data['Ambiguity Score Normalized'].mean() + std_dev_normalized
    plus2std_normalized = filtered_data['Ambiguity Score Normalized'].mean() + 2* std_dev_normalized

    if type_nr==1:
        num_to_set = int(optimal_reject_rate * len(all_data)) # example: 60/100 = 0.6 * length of the data
        all_data['ite_reject'] = all_data['ite_pred']
        all_data['ite_reject'] = all_data['ite_reject'].astype(object)  # Change dtype of entire column
        all_data.loc[:num_to_set -1, 'ite_reject'] = 'R'

        lowest_rejected_value = all_data.loc[all_data['ite_reject'] == 'R', 'size_of_ci'].iloc[-1]
        count_lowest_rejected_value = len(all_data[all_data['size_of_ci'] == lowest_rejected_value])


        # histogram(filtered_data['size_of_ci'], 'Ambiguity Scores', 'Frequency', 'Histogram of Frequency by Ambiguity Scores', f"{folder_path}histogram/{dataset}_{experiment_id}_{abbreviation}_histogram.png", lowest_rejected_value, mean, plusstd, plus2std)
        # histogram(filtered_data['Ambiguity Score Normalized'], 'Ambiguity Scores Normalized', 'Frequency', 'Histogram of Frequency by Ambiguity Scores (Normalized)', f"{folder_path}histogram/{dataset}_{experiment_id}_{abbreviation}_histogram_normalized.png", lowest_rejected_value,mean_normalized, plusstd_normalized, plus2std_normalized)

        all_data['ite_reject'] = all_data['ite_pred']
        all_data['ite_reject'] = all_data['ite_reject'].astype(object)  # Change dtype of entire column
        # all_data = all_data.sort_values(by='amount_of_times_rejected', ascending=False).copy()
        all_data[all_data['Ambiguity Score Normalized'] >= plusstd_normalized]

    metrics_dict = calculate_all_metrics('ite', 'ite_reject', all_data, file_path, {}, append_metrics_results=False, print=False)
    metrics_dict['2/ Optimal RR (%)'] = round(optimal_reject_rate * 100, 4)
    metrics_dict['2/ Original RMSE'] = metrics_dict.get('RMSE Original', None)
    metrics_dict['2/ Minimum RMSE'] = round(min_rmse, 4)
    metrics_dict['2/ Change of RMSE (%)'] = (min_rmse - metrics_dict.get('RMSE Original', None)) / metrics_dict.get('RMSE Original', None) * 100
    metrics_dict['2/ Improvement of RMSE (%)'] = -((min_rmse - metrics_dict.get('RMSE Original', None)) / metrics_dict.get('RMSE Original', None)) * 100
    metrics_dict['2/ Mistake from Perfect'] = round(sum([perfect - actual for perfect, actual in zip(rmse_accepted_perfect, rmse_accepted)]), 4)

    if type_nr == 1:
        metrics_dict['3/ Optimal Amount of times Rejected'] = lowest_rejected_value
        metrics_dict['3/ Count of this Optimal'] = count_lowest_rejected_value
    
    if give_details==True:
        # return metrics_dict, reject_rates, rmse_accepted, rmse_rank_accepted, sign_error_accepted, rmse_rank_weighted_accepted, rmse_change_accepted, heuristic_cutoff, sign_error_change_accepted, rmse_rank_weighted_change_accepted, rmse_rank_change_accepted
    
        return metrics_dict, reject_rates, heuristic_cutoff, metrics_results_list
    else:
        return metrics_dict
    
def novelty_rejection(type_nr, max_rr, detail_factor, model_name, x, all_data, file_path, experiment_id, dataset, folder_path, abbreviation, rmse_accepted_perfect=[], give_details=False):
    reject_rates = []
    rmse_accepted = []
    rmse_rejected = []
    rmse_change_accepted = []
    metrics_results_list = []

    # sign_error_accepted = []
    # rmse_rank_accepted = []
    # rmse_rank_weighted_accepted = []
    # sign_error_change_accepted = []
    # rmse_rank_weighted_change_accepted = []
    # rmse_rank_change_accepted = []

    min_rmse = float('inf')  # Set to positive infinity initially
    optimal_model = None

    if type_nr == 1:
        for contamination in range(int(1*detail_factor), int(max_rr*detail_factor)):
            contamination /= (100 * detail_factor) # max of 0.5

            if model_name == IsolationForest:
                model = train_model(x, IsolationForest, contamination=contamination, random_state=42) # lower contamination, less outliers
                all_data['ood'] = pd.Series(model.predict(x), name='ood')
            elif model_name == OneClassSVM:
                model = train_model(x, OneClassSVM, nu=contamination) # lower contamination, less outliers
                all_data['ood'] = pd.Series(model.predict(x), name='ood')
            elif model_name == LocalOutlierFactor:
                model = train_model(x, LocalOutlierFactor, contamination=contamination, novelty=True)
                all_data['ood'] = pd.Series(model.predict(x), name='ood')

            all_data['ite_reject'] = all_data.apply(lambda row: "R" if row['ood'] else row['ite_pred'], axis=1)

            all_data['y_reject'] = all_data.apply(lambda row: True if row['ood'] == -1 else False, axis=1)

            set_rejected = all_data.copy()
            set_accepted = all_data.copy()
            set_rejected = set_rejected[set_rejected['y_reject'] == True]
            set_accepted = set_accepted[set_accepted['y_reject'] == False]

            all_data['ite_reject'] = all_data.apply(lambda row: "R" if row['y_reject'] else row['ite_pred'], axis=1)

            metrics_result = calculate_performance_metrics('ite', 'ite_reject', all_data, file_path)

            if metrics_result:
                reject_rates.append(metrics_result.get('Rejection Rate', None))
                current_rmse = metrics_result.get('RMSE Accepted', None)
                rmse_accepted.append(metrics_result.get('RMSE Accepted', None))
                rmse_rejected.append(metrics_result.get('RMSE Rejected', None))
                improvement = ( metrics_result.get('RMSE Accepted', None) - metrics_result.get('RMSE Original', None) ) / metrics_result.get('RMSE Original', None) * 100
                
                metrics_results_list.append(metrics_result)
                # sign_error_accepted.append(metrics_result.get('Sign Error Accepted (%)', None))
                # sign_error_change_accepted.append(metrics_result.get('Sign Error Change (%)', 0))

                # rmse_rank_accepted.append(metrics_result.get('RMSE Rank Accepted', None))
                # rmse_rank_weighted_accepted.append(metrics_result.get('RMSE Rank Weighted Accepted', None))

                # rmse_rank_change_accepted.append(metrics_result.get('RMSE Rank Change (%)', 0))
                # rmse_rank_weighted_change_accepted.append(metrics_result.get('RMSE Rank Weighted Change (%)', 0))

                rmse_change_accepted.append(improvement)
            else:
                reject_rates.append(None)
                rmse_accepted.append(None)
                rmse_rejected.append(None)

            # Update minimum RMSE and optimal model if needed
            if current_rmse < min_rmse:
                min_rmse = current_rmse
                optimal_model = model
    
    if type_nr == 2 or type_nr == 3:
        # split the data
        t_data = all_data[all_data['treatment'] == 1].copy()
        ut_data = all_data[all_data['treatment'] == 0].copy()
        t_x = x[all_data['treatment'] == 1].copy()
        ut_x = x[all_data['treatment'] == 0].copy()

        t_data['amount_of_times_rejected'] = 0
        ut_data['amount_of_times_rejected'] = 0
        all_data['amount_of_times_rejected'] = 0

        for contamination in range(int(1 * detail_factor), int(49 * detail_factor)):
            amount_of_times_rejected_new = f(type_nr, contamination, t_x, ut_x, t_data, ut_data, detail_factor, model_name, all_data)
            all_data['amount_of_times_rejected'] += amount_of_times_rejected_new
            all_data['amount_of_times_rejected'].fillna(0, inplace=True)
            all_data['amount_of_times_rejected'] = all_data['amount_of_times_rejected'].astype(int)

        all_data = all_data.sort_values(by='amount_of_times_rejected', ascending=False).copy()
        all_data = all_data.reset_index(drop=True)

        for rr in range(1, max_rr*detail_factor):
            num_to_set = int(rr / (100.0*detail_factor) * len(all_data)) # example: 60/100 = 0.6 * length of the data

            all_data['ite_reject'] = all_data['ite_pred']
            all_data['ite_reject'] = all_data['ite_reject'].astype(object)  # Change dtype of entire column
            if num_to_set != 0:
                all_data.loc[:num_to_set -1, 'ite_reject'] = 'R'

            metrics_result = calculate_performance_metrics('ite', 'ite_reject', all_data, file_path)

            if metrics_result:
                reject_rates.append(metrics_result.get('Rejection Rate', None))
                rmse_accepted.append(metrics_result.get('RMSE Accepted', None))
                rmse_rejected.append(metrics_result.get('RMSE Rejected', None))
                improvement = ( metrics_result.get('RMSE Accepted', None) - metrics_result.get('RMSE Original', None) ) / metrics_result.get('RMSE Original', None) * 100
                rmse_change_accepted.append(improvement)
                metrics_results_list.append(metrics_result)

                # sign_error_accepted.append(metrics_result.get('Sign Error Accepted (%)', None))
                # rmse_rank_accepted.append(metrics_result.get('RMSE Rank Accepted', None))
                # rmse_rank_weighted_accepted.append(metrics_result.get('RMSE Rank Weighted Accepted', None))
                # sign_error_change_accepted.append(metrics_result.get('Sign Error Change (%)', 0))

                # rmse_rank_change_accepted.append(metrics_result.get('RMSE Rank Change (%)', 0))
                # rmse_rank_weighted_change_accepted.append(metrics_result.get('RMSE Rank Weighted Change (%)', 0))

            else:
                reject_rates.append(None)
                rmse_accepted.append(None)
                rmse_rejected.append(None)

    # Graph with reject rate and rmse_accepted & rmse_rejected
    twolinegraph(reject_rates, "Reject Rate", rmse_accepted, "RMSE of Accepted Samples", "green", rmse_rejected, "RMSE of Rejected Samples", "red", f"Impact of Reject Rate on RMSE for {dataset}", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_rmse.png")
    # twolinegraph(reject_rates, "Reject Rate", rmse_accepted, f"Experiment {experiment_id} ", "blue", rmse_accepted_perfect, "Perfect Rejection", "green", f"RMSE of Accepted Samples ({dataset})", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_PerfRMSEe.png")
    onelinechangegraph([rate * 100 for rate in reject_rates], "Reject Rate (%)", rmse_change_accepted, "RMSE Deviation from No-Rejection (%)", "green", f"Impact of Rejection on the RMSE of the TE ({dataset} - {experiment_id})", f"{folder_path}overleaf/rmse/{dataset}_{experiment_id}_{abbreviation}_ChangeRMSE.png")
    onelinegraph(reject_rates, "Reject Rate", rmse_accepted, "RMSE of Accepted Samples", "green", f"Impact of Reject Rate on RMSE for {dataset}", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_rmse_accepted.png")
    onelinegraph(reject_rates, "Reject Rate", rmse_rejected, "RMSE of Rejected Samples", "red", f"Impact of Reject Rate on RMSE for {dataset}", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_rmse_rejected.png")
    
    # Graph with reject rate and rmse_accepted & rmse_rejected
    onelinegraph(reject_rates, "Reject Rate", [result.get('Sign Error Accepted (%)', None) for result in metrics_results_list], "Sign Error Accepted (%)", "green", f"Impact of Reject Rate on Sign Error {dataset} ({experiment_id})", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_SignError_accepted.png")
    # onelinegraph(reject_rates, "Reject Rate", rmse_rank_accepted, "RMSE Rank Accepted", "green", f"Impact of Reject Rate on Rank Error {dataset} ({experiment_id})", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_RankError_accepted.png")
    # onelinegraph(reject_rates, "Reject Rate", rmse_rank_weighted_accepted, "RMSE Rank Weighted Accepted", "green", f"Impact of Reject Rate on Weighted Rank Error {dataset} ({experiment_id})", f"{folder_path}graph/{dataset}_{experiment_id}_{abbreviation}_RankWeightedError_accepted.png")

    # optimal model 
    min_rmse = min(rmse_accepted)  # Find the minimum
    min_rmse_index = rmse_accepted.index(min_rmse)  # Find the index of the minimum RMSE
    optimal_reject_rate = reject_rates[min_rmse_index]  # Get the rejection rate at the same index
    
    if type_nr==0:
        all_data['ite_reject'] = all_data['ite_pred']
        all_data['ite_reject'] = all_data['ite_reject'].astype(object)  # Change dtype of entire column
        all_data.loc[:num_to_set -1, 'ite_reject'] = 'R'
    elif type_nr==1:
        all_data['ood'] = pd.Series(optimal_model.predict(x), name='ood')
        all_data['y_reject'] = all_data.apply(lambda row: True if row['ood'] == -1 else False, axis=1)
        all_data['ite_reject'] = all_data.apply(lambda row: "R" if row['y_reject'] else row['ite_pred'], axis=1)
        heuristic_cutoff = len(all_data[all_data['ood'] > 0]) / len(all_data)
        # heuristic_cutoff = 0.25

    elif type_nr==2 or type_nr==3:
        num_to_set = int(optimal_reject_rate * len(all_data)) # example: 60/100 = 0.6 * length of the data
        all_data['ite_reject'] = all_data['ite_pred']
        all_data['ite_reject'] = all_data['ite_reject'].astype(object)  # Change dtype of entire column
        # all_data = all_data.sort_values(by='amount_of_times_rejected', ascending=False).copy()
        all_data.loc[:num_to_set -1, 'ite_reject'] = 'R'

        lowest_rejected_value = all_data.loc[all_data['ite_reject'] == 'R', 'amount_of_times_rejected'].iloc[-1]
        count_lowest_rejected_value = len(all_data[all_data['amount_of_times_rejected'] == lowest_rejected_value])

        all_data['Novelty Score'] = all_data['amount_of_times_rejected']

        # filtered_data = all_data[all_data['amount_of_times_rejected'] != 0]

        histogram(all_data['Novelty Score'], 'Novelty Scores', 'Frequency', 'Histogram of Frequency by Novelty Scores', f"{folder_path}histogram/{dataset}_{experiment_id}_{abbreviation}_histogram.png", lowest_rejected_value,0,0,0)
        all_data['Novelty Score Normalized'] = (all_data['Novelty Score'] - all_data['Novelty Score'].min()) / (all_data['Novelty Score'].max() - all_data['Novelty Score'].min())
        lowest_rejected_value = all_data.loc[all_data['ite_reject'] == 'R', 'Novelty Score Normalized'].iloc[-1]

        histogram(all_data['Novelty Score Normalized'], 'Novelty Scores Normalized', 'Frequency', 'Histogram of Frequency by Novelty Scores (Normalized)', f"{folder_path}histogram/{dataset}_{experiment_id}_{abbreviation}_histogram_normalized.png", lowest_rejected_value,0,0,0)
        heuristic_cutoff = len(all_data[all_data['Novelty Score'] > 0]) / len(all_data)
    
    metrics_dict = calculate_all_metrics('ite', 'ite_reject', all_data, file_path, {}, append_metrics_results=False, print=False)

    metrics_dict['2/ Optimal RR (%)'] = round(optimal_reject_rate, 4)*100

    original_rmse = metrics_dict.get('RMSE Original', None)
    metrics_dict['2/ Original RMSE ()'] = original_rmse
    metrics_dict['2/ Minimum RMSE'] = round(min_rmse, 4)

    metrics_dict['2/ Change of RMSE (%)'] = (min_rmse - original_rmse) / original_rmse * 100
    metrics_dict['2/ Improvement of RMSE (%)'] = -((min_rmse - original_rmse) / original_rmse) * 100

    mistake_from_perfect_column = [perfect - actual for perfect, actual in zip(rmse_accepted_perfect, rmse_accepted)]
    mistake_from_perfect = sum(mistake_from_perfect_column)
    metrics_dict['2/ Mistake from Perfect'] = round(mistake_from_perfect, 4)
    
    if type_nr == 2 or type_nr ==3:
        metrics_dict['3/ Optimal Amount of times Rejected'] = lowest_rejected_value
        metrics_dict['3/ Count of this Optimal'] = count_lowest_rejected_value
    if give_details==True:
        # return metrics_dict, reject_rates, rmse_accepted, rmse_rank_accepted, sign_error_accepted, rmse_rank_weighted_accepted, rmse_change_accepted, heuristic_cutoff, sign_error_change_accepted, rmse_rank_weighted_change_accepted, rmse_rank_change_accepted
        return metrics_dict, reject_rates, heuristic_cutoff, metrics_results_list

    else:
        return metrics_dict