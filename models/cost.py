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
            'Lost Cause': cost_potential_improvement,       # Cost of predicting 'Lost Cause' when true category is 'Sleeping Dog' = we'll not take any action ==> cost = potential improvement
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
    true_category = row['category']
    pred_category = row['category_pred']
    return cost_matrix[true_category][pred_category]

def calculate_misclassification_cost(test_set, rejection_cost=2):
    test_set['cost_ite_reject'] = test_set.apply(lambda row: rejection_cost if row['ite_reject']=="R" else row['cost_ite'], axis=1)
    total_cost_ite_2 = test_set['cost_ite_reject'].sum()
    return total_cost_ite_2
