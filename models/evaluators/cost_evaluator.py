"""
Table of Contents:
1. define_cost_matrix: Defines the cost matrix for different categories and predicted categories.
2. calculate_cost_ite: Calculates the cost of misclassification for a given row.
3. calculate_misclassification_cost: Calculates the total misclassification cost for a given test set.

Description:
This file contains functions related to misclassification costs. The cost matrix is defined to assign costs for different categories and predicted categories. The calculate_cost_ite function calculates the cost of misclassification for a given row based on the cost matrix. The calculate_misclassification_cost function calculates the total misclassification cost for a given test set by considering rejection cost.

Comments:
- The cost matrix is defined using a dictionary of dictionaries.
- The cost matrix is used to determine the cost of misclassification based on the true category and predicted category.
- The calculate_cost_ite function uses the cost matrix to calculate the cost of misclassification for a given row.
- The calculate_misclassification_cost function calculates the total misclassification cost by considering rejection cost for rows marked as 'R' in the 'ite_reject' column.

This file contains everything related to misclassification costs. These amount are expressed in a valuta (p.e. EUR).
"""
def categorize(row, is_pred=True):
    """
    Categorizes a row based on the values of y_t0 and y_t1.

    Parameters:
    - row: A dictionary representing a row of data.
    - is_pred: A boolean indicating whether the values are predictions or not.

    Returns:
    - A string representing the category of the row.
    """
    if is_pred:
        y_t0 = row['y_t0_pred']
        y_t1 = row['y_t1_pred']
    else:
        y_t0 = row['y_t0']
        y_t1 = row['y_t1']

    if y_t0 == 0 and y_t1 == 0:
        return 'Lost Cause'
    elif y_t0 == 1 and y_t1 == 0:
        return 'Sleeping Dog'
    elif y_t0 == 0 and y_t1 == 1:
        return 'Persuadable' # (can be rescued)
    elif y_t0 == 1 and y_t1 == 1:
        return 'Sure Thing'

def define_cost_matrix(cost_correct=0, cost_same_treatment=0, cost_wasted_treatment=5, cost_potential_improvement=10):
    cost_matrix = {
        'Lost Cause': {
            'Lost Cause': cost_correct, # Correct
            'Sleeping Dog': cost_same_treatment,     # Cost of predicting 'Sleeping Dog' when true category is 'Lost Cause' = we'll not take any action ==> cost = 0
            'Persuadable': cost_wasted_treatment,     # Cost of predicting 'Persuadable' when true category is 'Lost Cause' = we'll give a treatment for nothing
            'Sure Thing': cost_same_treatment,      # Cost of predicting 'Sure Thing' when true category is 'Lost Cause'
        },
        # define costs for other predicted categories when true category is 'Sleeping Dog'
        'Sleeping Dog': {
            'Lost Cause': cost_same_treatment,       # Cost of predicting 'Lost Cause' when true category is 'Sleeping Dog' = we'll not take any action ==> cost = potential improvement
            'Sleeping Dog': cost_correct,     # Correct
            'Persuadable': cost_wasted_treatment+10,      # Cost of predicting 'Persuadable' when true category is 'Sleeping Dog' = wasted T and negative consequenc
            'Sure Thing': cost_same_treatment,      # Cost of predicting 'Sure Thing' when true category is 'Sleeping Dog'
        },
        # define costs for other predicted categories when true category is 'Persuadable'
        'Persuadable': {
            'Lost Cause': cost_potential_improvement,       # Cost of predicting 'Lost Cause' when true category is 'Persuadable' = we'll not take any action ==> cost = potential improvement
            'Sleeping Dog': cost_potential_improvement,    # Cost of predicting 'Sleeping Dog' when true category is 'Persuadable'  = we'll not take any action ==> cost = potential improvement
            'Persuadable': cost_correct,      # Correct
            'Sure Thing': cost_potential_improvement,      # Cost of predicting 'Sure Thing' when true category is 'Persuadable'
        },
        'Sure Thing': {
            'Lost Cause': cost_same_treatment,      # Cost of predicting 'Lost Cause' when true category is 'Sure Thing' = we'll not take any action ==> cost = 0
            'Sleeping Dog': cost_same_treatment,    # Cost of predicting 'Sleeping Dog' when true category is 'Sure Thing' = we'll not take any action ==> cost = 0
            'Persuadable': cost_wasted_treatment,      # Cost of predicting 'Persuadable' when true category is 'Sure Thing'
            'Sure Thing': cost_correct,       # Correct
        },
    }
    return cost_matrix

def calculate_cost_ite(row):
    cost_matrix = define_cost_matrix()
    true_category = row['category'] # What happened in reality
    pred_category = row['category_rej'] # What we predicted
    cost = cost_matrix[true_category][pred_category]
    return cost

def calculate_misclassification_cost(data, rejection_cost=2):
    data['category_rej'] = data.apply(categorize, axis=1)
    data['cost_ite'] = data.apply(calculate_cost_ite, axis=1)
    data['category_reject'] = data.apply(lambda row: rejection_cost if row['ite_reject']=="R" else row['category_rej'], axis=1)

    data['cost_ite_reject'] = data.apply(lambda row: rejection_cost if row['ite_reject']=="R" else row['cost_ite'], axis=1)
    total_cost_ite = data['cost_ite_reject'].sum()
    return total_cost_ite

def calculate_cost_metrics(value, value_pred, data, file_path, print=False):

    total_cost_ite = calculate_misclassification_cost(data)

    correct = 0
    same_treatment = 0
    lost_potential = 0
    wasted_treatment = 0
    opposite_effect = 0
    for index, row in data.iterrows():
        if row['category'] == row['category_reject']:
            correct += 1
        elif row['category'] == 'Lost Cause' and row['category_reject'] == 'Persuadable':
            wasted_treatment += 1
        elif row['category'] == 'Sleeping Dog' and row['category_reject'] == 'Persuadable':
            opposite_effect += 1
        elif row['category'] == 'Persuadable' and row['category'] != row['category_reject']:
            lost_potential += 1
        elif row['category'] == 'Sure Thing' and row['category_reject'] == 'Persuadable':
            wasted_treatment += 1
        else:
            same_treatment += 1

    metrics_dict = {
        'Misclassification Cost': total_cost_ite,
        'Correct Prediction': correct,
        'Same Treatment Given': same_treatment,
        'Lost Potential': lost_potential,
        'Wasted Treatment': wasted_treatment,
        'Opposite Effect': opposite_effect

    }

    return metrics_dict





    metrics_dict = {
        'Misclassification Cost': total_cost_ite,
    }

    return metrics_dict
