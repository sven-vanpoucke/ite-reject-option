import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from propensity_score_matching import perform_matching

# Function to evaluate the performance based on thresholds
def evaluate_thresholds(df, y_true, propensity_scores, lower_threshold, upper_threshold):
    subset_data = df[(df['propensity_score'] >= lower_threshold) & (df['propensity_score'] <= upper_threshold)].copy().reset_index(drop=True)

    matched_dataset, df = perform_matching(subset_data, min_treatment_count=3, k_neighbors=10)
    t_true = matched_dataset['treatment']
    t_pred =matched_dataset['t_predicted']


    accuracy = f1_score(t_true, t_pred)
    return accuracy

# Function to perform grid search for ambiguity rejection thresholds
def optimize_thresholds(df, y_true, propensity_scores, lower_range=(0.4,0.8, 0.05), upper_range=(0.6, 1.0, 0.05)):
    param_grid = {
        'lower_threshold': np.arange(*lower_range),
        'upper_threshold': np.arange(*upper_range)
    }

    grid = ParameterGrid(param_grid)

    best_f1 = 0
    best_thresholds = None

    for params in grid:
        current_f1 = evaluate_thresholds(df,y_true, propensity_scores, params['lower_threshold'], params['upper_threshold'])

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thresholds = params

    return best_thresholds, best_f1

from data_loader import load_dataset
from propensity_score_matching import estimate_propensity_scores
df, df_data = load_dataset()

t_true = df['treatment']

df = estimate_propensity_scores(df,df_data,cv= 5,treatment="treatment")
propensity_scores =  df['propensity_score']

best_thresholds, best_accuracy = optimize_thresholds(df, t_true, propensity_scores)

print("Best Thresholds:", best_thresholds)
print("Best Accuracy:", best_accuracy)


