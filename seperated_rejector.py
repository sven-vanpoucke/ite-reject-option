import pandas as pd

# PREPROCESSING
from datasets.lalonde import processing_get_data_lalonde, processing_transform_data_lalonde
from datasets.twins import preprocessing_get_data_twin, preprocessing_transform_data_twin
from datasets.processing import preprocessing_split_t_c_data

# for lalaonde
# all_data = processing_get_data_lalonde()
# train_x, test_x, train_y, test_y, train_t, test_t = processing_transform_data_lalonde(all_data)

# for twins
train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y = preprocessing_get_data_twin()
train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y = preprocessing_transform_data_twin(train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y)
test_ite = test_potential_y["y_t1"]- test_potential_y["y_t0"]
train_ite = train_potential_y["y_t1"]- train_potential_y["y_t0"]
# split the data in treated and controlled
train_treated_x, train_control_x, train_treated_y, train_control_y, test_treated_x, test_control_x, test_treated_y, test_control_y = preprocessing_split_t_c_data(train_x, train_y, train_t, test_x, test_y, test_t)

# MODEL

from models.predictor import predictor_t_model
from sklearn.linear_model import LogisticRegression

# Training separate models for treated and control groups
treated_model, control_model = predictor_t_model(train_treated_x, train_treated_y, train_control_x, train_control_y, model_class=LogisticRegression, max_iter=10000, solver='saga', random_state=42)

# REJECT

# Here should come the code to reject certain predictions in the test_set...

# PREDICT

"""# Predictions for treated groups
train_treated_y_pred = treated_model.predict(train_treated_x)
train_treated_y_prob = treated_model.predict_proba(train_treated_x)[:, 1]
test_treated_y_pred = treated_model.predict(test_treated_x)
test_treated_y_prob = treated_model.predict_proba(test_treated_x)[:, 1]

# Predictions for control groups (T = 0)
train_control_y_pred = control_model.predict(train_control_x)
train_control_y_prob = control_model.predict_proba(train_control_x)[:, 1]
test_control_y_pred = control_model.predict(test_control_x)
test_control_y_prob = control_model.predict_proba(test_control_x)[:, 1]
"""
train_treated_y_pred = pd.Series(treated_model.predict(train_treated_x), name='y_pred')
train_treated_y_prob = pd.Series(treated_model.predict_proba(train_treated_x)[:, 1], name='y_prob')
test_treated_y_pred = pd.Series(treated_model.predict(test_treated_x), name='y_pred')
test_treated_y_prob = pd.Series(treated_model.predict_proba(test_treated_x)[:, 1], name='y_prob')

# Predictions for control groups (T = 0)
train_control_y_pred = pd.Series(control_model.predict(train_control_x), name='y_pred')
train_control_y_prob = pd.Series(control_model.predict_proba(train_control_x)[:, 1], name='y_prob')
test_control_y_pred = pd.Series(control_model.predict(test_control_x), name='y_pred')
test_control_y_prob = pd.Series(control_model.predict_proba(test_control_x)[:, 1], name='y_prob')

# Predictions for test set
test_y_t1_pred = pd.Series(treated_model.predict(test_x), name='y_t1_pred')
test_y_t0_pred = pd.Series(control_model.predict(test_x), name='y_t0_pred')

test_y_t1_prob = pd.Series(treated_model.predict_proba(test_x)[:, 1], name='y_t1_prob')
test_y_t0_prob = pd.Series(control_model.predict_proba(test_x)[:, 1], name='y_t0_prob')
test_ite_prob = pd.Series(test_y_t1_prob-test_y_t0_prob, name='ite_prob')

# EVALUATE MODELS
from models.model_evaluator import evaluation_binary

print("training set:")
#evaluation_binary(train_treated_y, train_treated_y_pred, train_treated_y_prob, train_control_y, train_control_y_pred, train_control_y_prob)

print("test set:")
#evaluation_binary(test_treated_y, test_treated_y_pred, test_treated_y_prob, test_control_y, test_control_y_pred, test_control_y_prob)

# EVALUATE ITE

from models.evaluator import categorize, categorize_pred

test_set_prob = pd.concat([test_t, test_y_t1_prob, test_y_t0_prob, test_ite_prob, test_potential_y["y_t0"], test_potential_y["y_t1"], test_ite], axis=1)
test_set = pd.concat([test_t, test_y_t1_pred, test_y_t1_prob, test_y_t0_pred, test_y_t0_prob, test_ite_prob, test_potential_y["y_t0"], test_potential_y["y_t1"], test_ite], axis=1)


# Apply the categorization function to create the 'Category' column
test_set['category'] = test_set.apply(categorize, axis=1)
test_set['category_pred'] = test_set.apply(categorize_pred, axis=1)

#count_matrix = pd.crosstab(test_set['y_t0'], test_set['y_t1'], margins=True, margins_name='Total')
count_matrix = pd.crosstab(test_set['y_t0'], test_set['y_t1'], margins=False)
print(count_matrix)

count_matrix = pd.crosstab(test_y_t0_pred, test_y_t1_pred, margins=False)
print(count_matrix)

# COSTS
from models.cost import calculate_cost_ite

# Apply the calculate_cost function to each row in the DataFrame
test_set['cost_ite'] = test_set.apply(calculate_cost_ite, axis=1)

# test_set['cost_cb'] = test_set.apply(calculate_cost_cb, axis=1)
print(test_set)



# Rejection ood

from models.rejector import distance_test_to_train, is_out_of_distribution, nbrs_train

# Create a new column 'ood' in test_set based on the calculated distances
model = nbrs_train(train_x)
d = distance_test_to_train(model, test_x)
test_set['ood'] = d.apply(is_out_of_distribution, threshold_distance=6)



# Rejection probabilities
test_set['y_t1_reject_prob'] = test_set.apply(lambda row: True if row['y_t1_prob'] < 0.55 and row['y_t1_prob'] > 0.45 else False, axis=1)
test_set['y_t0_reject_prob'] = test_set.apply(lambda row: True if row['y_t0_prob'] < 0.55 and row['y_t0_prob'] > 0.45 else False, axis=1)
test_set['y_reject_prob'] = test_set.apply(lambda row: True if row['y_t0_reject_prob'] and row['y_t1_reject_prob'] else False, axis=1)

test_set['cost_ite_reject_prob'] = test_set.apply(lambda row: 0 if y_reject_prob else row['cost_ite'], axis=1)


print(test_set)

# Calculate total misclassification cost
total_cost = test_set['cost_ite'].sum()
print(f'Total Misclassification Cost: {total_cost}')

# Print the count of occurrences where 'ood' is true
print("Count of 'ood' being true:", test_set['ood'].sum())

# Calculate total misclassification cost
test_set['cost_ite_reject_ood'] = test_set.apply(lambda row: 0 if row['ood'] else row['cost_ite'], axis=1)
total_cost = test_set['cost_ite_reject_ood'].sum()
print(f'Total Misclassification Cost after ood rejection: {total_cost}')

# Print the count of occurrences where 'ood' is true
print("Count of 'probability rejection' being true:", test_set['y_reject_prob'].sum())
total_cost = test_set['cost_ite_reject_prob'].sum()
print(f'Total Misclassification Cost after probability rejection: {total_cost}')
