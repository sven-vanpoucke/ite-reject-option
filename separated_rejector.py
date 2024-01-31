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
# REJECTION OOD
from models.rejector import distance_test_to_train, is_out_of_distribution, nbrs_train
from models.helper import improvement
# REJECTION PROBABILITIES
# ...
# PARAMETERS
dataset = "twins" # Choose out of twins or lalonde
model_class = LogisticRegression # Which two models do we want to generate in the t-models
rejection_type = "ood" # Choose out of ood or prob
prob_reject_upper_bound = 0.55
prob_reject_under_bound = 0.45


# INIT
timestamp, file_name, file_path = helper_output()

# PREPROCESSING
with open(file_path, 'a') as file:
    file.write(f"CHAPTER 2: PREPROCESSING\n\n")
    file.write(f"The used dataset is: {dataset}\n\n")

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
    file.write(f"CHAPTER 3: MODEL T-LEARNER\n\n")
    file.write(f"The used model is: {model_class.__name__}\n\n")

# Training separate models for treated and control groups
treated_model, control_model = predictor_t_model(train_treated_x, train_treated_y, train_control_x, train_control_y, model_class=model_class, max_iter=10000, solver='saga', random_state=42)

# PREDICT
with open(file_path, 'a') as file:
    file.write(f"CHAPTER 4: PREDICT\n\n")

# Training and Testing predictions to evaluate individual models
train_treated_y_pred, train_treated_y_prob, train_control_y_pred, train_control_y_prob = predictor_train_predictions(treated_model, control_model, train_treated_x, train_control_x)
test_treated_y_pred, test_treated_y_prob, test_control_y_pred, test_control_y_prob = predictor_test_predictions(treated_model, control_model, test_treated_x, test_control_x)

# Testing Predictions to evaluate ITE
test_y_t1_pred, test_y_t0_pred, test_y_t1_prob, test_y_t0_prob, test_ite_prob, test_ite_pred = predictor_ite_predictions(treated_model, control_model, test_x)

# EVALUATE INDIVIDUAL MODELS
with open(file_path, 'a') as file:
    file.write(f"CHAPTER 5: EVALUATE INDIVIDUAL LOGISTIC REGRESSION MODELS \n\n")
    file.write(f"Performance measurement: \n\n")
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
    file.write(f"Performance measurement: \n")
    file.write(f" - Root Mean Squared Error (RMSE) of the ITE \n")
    file.write(f" - Accurate estimate of the ATE \n")
    file.write(f" - Accurancy of ITE \n\n")


test_set = pd.concat([test_t, test_y_t1_pred, test_y_t1_prob, test_y_t0_pred, test_y_t0_prob, test_ite_pred, test_ite_prob, test_potential_y["y_t0"], test_potential_y["y_t1"], test_ite], axis=1)

# Calculate
rmse = np.sqrt(mean_squared_error(test_set['ite'], test_set['ite_prob'])) # Calculate Root Mean Squared Error (RMSE)
ate_accuracy = np.abs(test_set['ite_pred'].mean() - test_set['ite'].mean()) # Evaluate ATE accuracy

accurancy_ite = calculate_crosstab('ite', 'ite_pred', test_set, file_path)

# Log results
with open(file_path, 'a') as file:
    file.write(f"Root Mean Squared Error (RMSE) between the ite and ite_prob: {rmse.round(4)}\n\n")
    file.write(f"The Actual Average Treatment Effect (ATE): {test_set['ite'].mean().round(4)}\n")
    file.write(f"The Predicted Average Treatment Effect (ATE): {test_set['ite_pred'].mean().round(4)}\n")
    file.write(f"Accuracy of Average Treatment Effect (ATE): {ate_accuracy.round(4)}\n")

# EVALUATE OVERALL ITE MODEL: COST
with open(file_path, 'a') as file:
    file.write(f"\n\nCHAPTER 7: EVALUATE OVERALL ITE MODEL: COST \n\n")

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

# REJECTION
with open(file_path, 'a') as file:
    file.write(f"\nCHAPTER 8: REJECTION \n\n")
    file.write(f"The used type of rejection is: {rejection_type}\n\n")

if rejection_type == "ood":
    # REJECTION OOD
    model = nbrs_train(train_x)
    d = distance_test_to_train(model, test_x)
    test_set['ood'] = d.apply(is_out_of_distribution, threshold_distance=6)
    percentage_rejected = (test_set['ood'].sum() / test_set['ood'].count()).round(4)*100

    # Calculate total misclassification cost
    test_set['cost_ite_reject_ood'] = test_set.apply(lambda row: 0 if row['ood'] else row['cost_ite'], axis=1)
    total_cost_ite_reject_ood = test_set['cost_ite_reject_ood'].sum()
    improvement_cost_reject_ood = improvement(total_cost_ite, total_cost_ite_reject_ood)

    test_set['ite_rej'] = test_set.apply(lambda row: "R" if row['ood'] else row['ite_pred'], axis=1)
    # Create a cross-tabulation between 'ite' and 'ite_rej'
    accurancy_ite_reject_ood = calculate_crosstab('ite', 'ite_rej', test_set, file_path)
    improvement_ite_reject_ood = improvement(accurancy_ite, accurancy_ite_reject_ood)

    with open(file_path, 'a') as file:
        # Write the count of occurrences where 'ood' is true
        file.write(f"Count of 'ood' being true: {test_set['ood'].sum()}\n")
        # Write the total misclassification cost
        file.write(f'Total Misclassification Cost after ood rejection: {total_cost_ite_reject_ood}\n')
        file.write(f'Change of the misclassification cost after ood rejection: {improvement_cost_reject_ood}%\n')
        file.write(f'Change of the ITE Accurancy: {improvement_ite_reject_ood}%\n')
        file.write(f'Rejection rate: {percentage_rejected}%\n')

    # Calculate new TP, TN, FP, FN from the confusion matrices

elif rejection_type == "prob":
    # REJECTION PROBABILITIES
    test_set['y_t1_reject_prob'] = test_set.apply(lambda row: True if row['y_t1_prob'] < prob_reject_upper_bound and row['y_t1_prob'] > prob_reject_under_bound else False, axis=1)
    test_set['y_t0_reject_prob'] = test_set.apply(lambda row: True if row['y_t0_prob'] < prob_reject_upper_bound and row['y_t0_prob'] > prob_reject_under_bound else False, axis=1)
    test_set['y_reject_prob'] = test_set.apply(lambda row: True if row['y_t0_reject_prob'] and row['y_t1_reject_prob'] else False, axis=1)
    test_set['cost_ite_reject_prob'] = test_set.apply(lambda row: 0 if row['y_reject_prob'] else row['cost_ite'], axis=1)
    
    #  count of occurrences where 'ood' is true
    total_cost_ite_reject_prob = test_set['cost_ite_reject_prob'].sum()
    with open(file_path, 'a') as file:
        # Write the count of occurrences where 'ood' is true after probability rejection
        file.write(f"Count of 'probability rejection' being true: {test_set['y_reject_prob'].sum()}\n")

        # Write the total misclassification cost after probability rejection
        file.write(f'Total Misclassification Cost after probability rejection: {total_cost_ite_reject_prob}\n')

with open(file_path, 'a') as file:
    file.write("\n\nTable of test_set (First 20 rows)\n")
    file.write(tabulate(test_set.head(20), headers='keys', tablefmt='pretty', showindex=False))