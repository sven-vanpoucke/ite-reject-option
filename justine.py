# This file has been created by Justine

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
from datasets.testIHDP import processing_get_data_ihdp, processing_transform_data_ihdp
from datasets.twins import preprocessing_get_data_twin, preprocessing_transform_data_twin
from datasets.processing import preprocessing_split_t_c_data
# MODEL T-LEARNER
from models.predictor import predictor_t_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# PREDICT 
from models.predictor import predictor_train_predictions, predictor_test_predictions, predictor_ite_predictions
# EVALUATE INDIVIDUAL MODELS
from models.model_evaluator import evaluation_binary
# EVALUATE OVERALL ITE MODEL: PERFORMANCE
# REJECTION OOD
from models.rejectors.ood_rejector import distance_test_to_train
#from models.rejectors.helper import is_out_of_distribution, nbrs_train
from models.helper import improvement
# REJECTION PROBABILITIES
# ...
# PARAMETERS
dataset = "ihdp" # Choose out of twins or lalonde
model_class = LinearRegression # Which two models do we want to generate in the t-models, for ihdp, we cannot use logistic regression!
rejection_type = "ood" # Choose out of ood or prob
prob_reject_upper_bound = 0.55
prob_reject_under_bound = 0.45

'''
# INIT
timestamp, file_name, file_path = helper_output()

# PREPROCESSING
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(f"CHAPTER 2: PREPROCESSING\n\n")
    file.write(f"The used dataset is: {dataset}\n\n")
'''
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
elif dataset == "ihdp":
    all_data = processing_get_data_ihdp()
    train_x, test_x, train_y, test_y, train_t, test_t = processing_transform_data_ihdp(all_data)
    test_t = pd.DataFrame(test_t)
    test_t.columns = ['treatment']
    train_t = pd.DataFrame(train_t)
    train_t.columns = ['treatment']
else:
    pass
'''
# MODEL T-LEARNER
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(f"CHAPTER 3: MODEL T-LEARNER\n\n")
    #file.write(f"The used model is: {model_class._name_}\n\n")
'''
# Training separate models for treated and control groups
    #first: separate treated vs control using treatment == 0 or 1
#then us the processing function to get train_treated_x and so forth
'''
def preprocessing_split_t_c_data(train_x, train_y, train_t, test_x, test_y, test_t):
  
  # for training data
  train_treated_x = train_x[train_t['treatment']  == 1]
  train_control_x = train_x[train_t['treatment'] == 0] # no treatment given...
  train_treated_y = train_y[train_t['treatment'] == 1]
  train_control_y = train_y[train_t['treatment'] == 0]
  
  # for test data
  test_treated_x = test_x[test_t['treatment'] == 1]
  test_control_x = test_x[test_t['treatment'] == 0] #no treatment given...
  test_treated_y = test_y[test_t['treatment'] == 1]
  test_control_y = test_y[test_t['treatment'] == 0]

  return train_treated_x, train_control_x, train_treated_y, train_control_y, test_treated_x, test_control_x, test_treated_y, test_control_y
'''
train_treated_x, train_control_x, train_treated_y, train_control_y, test_treated_x, test_control_x, test_treated_y, test_control_y= preprocessing_split_t_c_data(train_x, train_y, train_t, test_x, test_y, test_t)
print("train treated x")
print(train_treated_x)
print("train_treated_y")
print(train_treated_y)


'''
    make use of the t model below: 
    def predictor_t_model(train_treated_x, train_treated_y, train_control_x, train_control_y, model_class=LogisticRegression, **model_params):
    # Training separate models for treated and control groups
    # treated_model = LogisticRegression(max_iter=10000, solver='saga', random_state=42))
    treated_model = model_class(**model_params)
    treated_model.fit(train_treated_x, train_treated_y.values.flatten())

    control_model = model_class(**model_params)
    control_model.fit(train_control_x, train_control_y.values.flatten())
    
    return treated_model, control_model
 '''
train_treated_y = train_y[train_t['treatment'] == 1]['y_factual']  # select 'y_factual' for treated
train_control_y = train_y[train_t['treatment'] == 0]['y_cfactual']  # select 'y_cfactual' for control
train_treated_y = pd.DataFrame(train_treated_y)
train_treated_y.columns = ['y_factual']
train_control_y = pd.DataFrame(train_control_y)
train_control_y.columns = ['y_cfactual']

print("train_treated_y")
print(train_treated_y)

treated_model, control_model = predictor_t_model(train_treated_x, train_treated_y, train_control_x, train_control_y, model_class=model_class)
print(treated_model)
print(control_model)

#Propensity Score Matching:
#This technique aims to balance treated and control groups based on their propensity to receive treatment, estimated using logistic regression. This reduces bias compared to directly comparing treated and untreated groups:

propensity_model = LogisticRegression()
propensity_model.fit(train_x,train_t)
propensity_scores = propensity_model.predict_proba(train_x)[:, 1]  # Probability of receiving treatment

print(max(propensity_scores)) #only 0.5515906372071074

# Use the trained model to predict treatment assignment for the entire test set
predicted_treatments = propensity_model.predict(test_x)
predicted_treatments =np.array(predicted_treatments)

# Create a confusion matrix
conf_matrix = confusion_matrix(test_t, predicted_treatments)

# Calculate performance metrics
accuracy = accuracy_score(test_t, predicted_treatments)
precision = precision_score(test_t, predicted_treatments)
recall = recall_score(test_t, predicted_treatments)
f1 = f1_score(test_t, predicted_treatments)

print("Confusion Matrix:")
print(conf_matrix)
# Extract TP, TN, FP, FN from the confusion matrices
TP, FP, FN, TN = conf_matrix.ravel()

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
# sklearn uses threshold of 0.5 to classify, therefore, a treatment is not often assigned when the max is only 0,55

ATE= all_data["y_factual"]-all_data["y_cfactual"]
ATE = np.array(ATE)  # Example ATE array

average_ATE = np.mean(ATE)
print("Average ATE:", average_ATE)




#this was treatment allocation, without reject options, now we will include reject options, this can be done using 2 techniques: ood and probability based. 
# reject model:
'''
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(f"\nCHAPTER 8: REJECTION \n\n")
    file.write(f"The used type of rejection is: {rejection_type}\n\n")
'''

#creating test_set
df = pd.DataFrame({'predicted_treatments': predicted_treatments})
df.reset_index(drop=True, inplace=True)
test_t_df = pd.DataFrame(test_t, columns=['treatment']).reset_index(drop=True)
test_ITE= test_y["y_factual"]- test_y["y_cfactual"]
test_ITE_df = pd.DataFrame(test_ITE, columns=['test ITE']).reset_index(drop=True)
test_y_df =pd.DataFrame(test_y, columns=['y_factual', 'y_cfactual']).reset_index(drop=True)

test_set = pd.concat([test_t_df, df,test_ITE_df,test_y_df], axis=1)


print(test_set.iloc[1,2])

false=0
ITE_pred=[]
# when treatment is given, y_factual is observed , whereas y_cfactual is an estimation of the other effect, therefore when we dont predict the same, these values must be switched in the estimation of the predicted ITE
for index in test_set.index:
    print(index)
    if test_set.iloc[index,0]==test_set.iloc[index,1]:
        
        ITE_pred.append(test_set.iloc[index, 3]-test_set.iloc[index,4])
    else:
         false+=1
         ITE_pred.append(test_set.iloc[index, 4]-test_set.iloc[index,3])


print("amount of inequal predictions to treatment assignment= ",false)

ITE_pred_series = pd.Series(ITE_pred, name='ITE_pred')
test_set = pd.concat([test_t_df, df,test_ITE_df,test_y_df,ITE_pred_series], axis=1)
print(test_set)

'''
# Calculate
#rmse = np.sqrt(mean_squared_error(test_set['ite'], test_set['ite_prob'])) # Calculate Root Mean Squared Error (RMSE)
ate_accuracy = np.abs(test_set['ite_pred'].mean() - test_set['ite'].mean()) # Evaluate ATE accuracy

'''
ate_accuracy = np.abs(test_set['ITE_pred'].mean() - test_set['test ITE'].mean()) # Evaluate ATE accuracy
print('average predicted ITE is: ',test_set['ITE_pred'].mean())
print('average test ITE is: ',test_set['test ITE'].mean())
print('ate accuracy is: ',ate_accuracy)

'''
#printing the cases where the assigned treatment in the data is not the predicted treatment and their corresponding ITEn and predicted_ITE
wrong_ITE=0
for index in test_set.index:
    if test_set.iloc[index,2]==test_set.iloc[index,5]:
        pass
        
    else:
         wrong_ITE+=1
         print("test ITE",test_set.iloc[index,2])
         print("pred_ITE= ",test_set.iloc[index,5])
         print("y_fact= ",test_set.iloc[index,3])
         print("y_cfact= ",test_set.iloc[index,4])

print(wrong_ITE)
'''

"""
if rejection_type == "ood":
    # REJECTION OOD
    model = nbrs_train(train_x)
    d = distance_test_to_train(model, test_x)
    test_set['ood'] = d.apply(is_out_of_distribution, threshold_distance=6)
    percentage_rejected = (test_set['ood'].sum() / test_set['ood'].count()).round(4)*100
    print(test_set)
    print(percentage_rejected)
"""

"""
#RMSE (Root Mean Squared Error) is typically used for evaluating the accuracy of a predictive model, and it's calculated based on the differences between the actual and predicted values.
rmse = np.sqrt(mean_squared_error(test_set['test ITE'], test_set['ITE_pred'])) # Calculate Root Mean Squared Error (RMSE)
print(rmse)
"""

#TODO
#costs calculation - predict
#costs calculation - rejcect
#


'''
# REJECTION
with open(file_path, 'w', encoding='utf-8') as file:
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

    with open(file_path, 'w', encoding='utf-8') as file:
        # Write the count of occurrences where 'ood' is true
        file.write(f"Count of 'ood' being true: {test_set['ood'].sum()}\n")
        # Write the total misclassification cost
        file.write(f'Total Misclassification Cost after ood rejection: {total_cost_ite_reject_ood}\n')
        file.write(f'Change of the misclassification cost after ood rejection: {improvement_cost_reject_ood}%\n')
        file.write(f'Change of the ITE Accurancy: {improvement_ite_reject_ood}%\n')
        file.write(f'Rejection rate: {percentage_rejected}%\n')

    # Calculate new TP, TN, FP, FN from the confusion matrices
'''




'''

# PREDICT
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(f"CHAPTER 4: PREDICT\n\n")

# Training and Testing predictions to evaluate individual models
train_treated_y_pred, train_treated_y_prob, train_control_y_pred, train_control_y_prob = predictor_train_predictions(treated_model, control_model, train_treated_x, train_control_x)
test_treated_y_pred, test_treated_y_prob, test_control_y_pred, test_control_y_prob = predictor_test_predictions(treated_model, control_model, test_treated_x, test_control_x)

# Testing Predictions to evaluate ITE
test_y_t1_pred, test_y_t0_pred, test_y_t1_prob, test_y_t0_prob, test_ite_prob, test_ite_pred = predictor_ite_predictions(treated_model, control_model, test_x)

# EVALUATE INDIVIDUAL MODELS
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(f"CHAPTER 5: EVALUATE INDIVIDUAL LOGISTIC REGRESSION MODELS \n\n")
    file.write(f"Performance measurement: \n\n")
    file.write(f" - Confusion Matrix \n")
    file.write(f" - Accuracy: overal correctness of the model ((TP + TN) / (TP + TN + FP + FN)) \n")
    file.write(f" - Precision: It measures the accuracy of positive predictions (TP / (TP + FP)) \n")
    file.write(f" - Recall: ability of the model to capture all the relevant cases (TP / (TP + FN)) \n")
    file.write(f" - F1 Score It balances precision and recall, providing a single metric for model evaluation (2 * (Precision * Recall) / (Precision + Recall)) \n")
    file.write(f" - ROC \n\n")

with open(file_path, 'w', encoding='utf-8') as file:
    file.write("Evaluation of the individual models based on the *training data*\n")
evaluation_binary(train_treated_y, train_treated_y_pred, train_treated_y_prob, train_control_y, train_control_y_pred, train_control_y_prob, file_path)

with open(file_path, 'w', encoding='utf-8') as file:
    file.write("\nEvaluation of the individual models based on the *test data*\n")
evaluation_binary(test_treated_y, test_treated_y_pred, test_treated_y_prob, test_control_y, test_control_y_pred, test_control_y_prob, file_path)

# EVALUATE OVERALL ITE MODEL: PERFORMANCE
with open(file_path, 'w', encoding='utf-8') as file:
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
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(f"Root Mean Squared Error (RMSE) between the ite and ite_prob: {rmse.round(4)}\n\n")
    file.write(f"The Actual Average Treatment Effect (ATE): {test_set['ite'].mean().round(4)}\n")
    file.write(f"The Predicted Average Treatment Effect (ATE): {test_set['ite_pred'].mean().round(4)}\n")
    file.write(f"Accuracy of Average Treatment Effect (ATE): {ate_accuracy.round(4)}\n")

# EVALUATE OVERALL ITE MODEL: COST
with open(file_path, 'w', encoding='utf-8') as file:
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
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(f"\nTotal Misclassification Cost: {total_cost_ite}\n")

# REJECTION
with open(file_path, 'w', encoding='utf-8') as file:
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

    with open(file_path, 'w', encoding='utf-8') as file:
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
    with open(file_path, 'w', encoding='utf-8') as file:
        # Write the count of occurrences where 'ood' is true after probability rejection
        file.write(f"Count of 'probability rejection' being true: {test_set['y_reject_prob'].sum()}\n")

        # Write the total misclassification cost after probability rejection
        file.write(f'Total Misclassification Cost after probability rejection: {total_cost_ite_reject_prob}\n')

with open(file_path, 'w', encoding='utf-8') as file:
    file.write("\n\nTable of test_set (First 20 rows)\n")
    file.write(tabulate(test_set.head(20), headers='keys', tablefmt='pretty', showindex=False))




# Training separate models for treated and control groups
# treated_model, control_model = predictor_t_model(train_treated_x, train_treated_y, train_control_x, train_control_y, model_class=model_class, max_iter=10000, solver='saga', random_state=42)

# PREDICT
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(f"CHAPTER 4: PREDICT\n\n")

# Training and Testing predictions to evaluate individual models
train_treated_y_pred, train_treated_y_prob, train_control_y_pred, train_control_y_prob = predictor_train_predictions(treated_model, control_model, train_treated_x, train_control_x)
test_treated_y_pred, test_treated_y_prob, test_control_y_pred, test_control_y_prob = predictor_test_predictions(treated_model, control_model, test_treated_x, test_control_x)

# Testing Predictions to evaluate ITE
test_y_t1_pred, test_y_t0_pred, test_y_t1_prob, test_y_t0_prob, test_ite_prob, test_ite_pred = predictor_ite_predictions(treated_model, control_model, test_x)

# EVALUATE INDIVIDUAL MODELS
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(f"CHAPTER 5: EVALUATE INDIVIDUAL LOGISTIC REGRESSION MODELS \n\n")
    file.write(f"Performance measurement: \n\n")
    file.write(f" - Confusion Matrix \n")
    file.write(f" - Accuracy: overal correctness of the model ((TP + TN) / (TP + TN + FP + FN)) \n")
    file.write(f" - Precision: It measures the accuracy of positive predictions (TP / (TP + FP)) \n")
    file.write(f" - Recall: ability of the model to capture all the relevant cases (TP / (TP + FN)) \n")
    file.write(f" - F1 Score It balances precision and recall, providing a single metric for model evaluation (2 * (Precision * Recall) / (Precision + Recall)) \n")
    file.write(f" - ROC \n\n")

with open(file_path, 'w', encoding='utf-8') as file:
    file.write("Evaluation of the individual models based on the *training data*\n")
evaluation_binary(train_treated_y, train_treated_y_pred, train_treated_y_prob, train_control_y, train_control_y_pred, train_control_y_prob, file_path)

with open(file_path, 'w', encoding='utf-8') as file:
    file.write("\nEvaluation of the individual models based on the *test data*\n")
evaluation_binary(test_treated_y, test_treated_y_pred, test_treated_y_prob, test_control_y, test_control_y_pred, test_control_y_prob, file_path)

# EVALUATE OVERALL ITE MODEL: PERFORMANCE
with open(file_path, 'w', encoding='utf-8') as file:
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
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(f"Root Mean Squared Error (RMSE) between the ite and ite_prob: {rmse.round(4)}\n\n")
    file.write(f"The Actual Average Treatment Effect (ATE): {test_set['ite'].mean().round(4)}\n")
    file.write(f"The Predicted Average Treatment Effect (ATE): {test_set['ite_pred'].mean().round(4)}\n")
    file.write(f"Accuracy of Average Treatment Effect (ATE): {ate_accuracy.round(4)}\n")

# EVALUATE OVERALL ITE MODEL: COST
with open(file_path, 'w', encoding='utf-8') as file:
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
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(f"\nTotal Misclassification Cost: {total_cost_ite}\n")

# REJECTION
with open(file_path, 'w', encoding='utf-8') as file:
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

    with open(file_path, 'w', encoding='utf-8') as file:
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
    with open(file_path, 'w', encoding='utf-8') as file:
        # Write the count of occurrences where 'ood' is true after probability rejection
        file.write(f"Count of 'probability rejection' being true: {test_set['y_reject_prob'].sum()}\n")

        # Write the total misclassification cost after probability rejection
        file.write(f'Total Misclassification Cost after probability rejection: {total_cost_ite_reject_prob}\n')

with open(file_path, 'w', encoding='utf-8') as file:
    file.write("\n\nTable of test_set (First 20 rows)\n")
    file.write(tabulate(test_set.head(20), headers='keys', tablefmt='pretty', showindex=False))
'''