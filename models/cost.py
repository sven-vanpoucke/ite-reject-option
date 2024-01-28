correct = 0
same_action_taken = 0
wasted_treatment = 5 # Giving treatment when it shouldn't have been given
potential_improvement = 10 # Not giving treatment when it should have been given

cost_matrix = {
    'Lost Cause': {
        'Lost Cause': correct, # Correct
        'Sleeping Dog': same_action_taken,     # Cost of predicting 'Sleeping Dog' when true category is 'Lost Cause' = we'll not take any action ==> cost = 0
        'Persuadable': wasted_treatment,     # Cost of predicting 'Persuadable' when true category is 'Lost Cause' = we'll give a treatment for nothing
        'Sure Thing': same_action_taken,      # Cost of predicting 'Sure Thing' when true category is 'Lost Cause'
    },
    # define costs for other predicted categories when true category is 'Sleeping Dog'
    'Sleeping Dog': {
        'Lost Cause': potential_improvement,       # Cost of predicting 'Lost Cause' when true category is 'Sleeping Dog' = we'll not take any action ==> cost = potential improvement
        'Sleeping Dog': correct,     # Correct
        'Persuadable': wasted_treatment+10,      # Cost of predicting 'Persuadable' when true category is 'Sleeping Dog' = wasted T and negative consequenc
        'Sure Thing': same_action_taken,      # Cost of predicting 'Sure Thing' when true category is 'Sleeping Dog'
    },
    # define costs for other predicted categories when true category is 'Persuadable'
    'Persuadable': {
        'Lost Cause': potential_improvement,       # Cost of predicting 'Lost Cause' when true category is 'Persuadable' = we'll not take any action ==> cost = potential improvement
        'Sleeping Dog': potential_improvement,    # Cost of predicting 'Sleeping Dog' when true category is 'Persuadable'  = we'll not take any action ==> cost = potential improvement
        'Persuadable': correct,      # Correct
        'Sure Thing': potential_improvement,      # Cost of predicting 'Sure Thing' when true category is 'Persuadable'
    },
    'Sure Thing': {
        'Lost Cause': same_action_taken,      # Cost of predicting 'Lost Cause' when true category is 'Sure Thing' = we'll not take any action ==> cost = 0
        'Sleeping Dog': same_action_taken,    # Cost of predicting 'Sleeping Dog' when true category is 'Sure Thing' = we'll not take any action ==> cost = 0
        'Persuadable': wasted_treatment,      # Cost of predicting 'Persuadable' when true category is 'Sure Thing'
        'Sure Thing': correct,       # Correct
    },
}

def calculate_cost_ite(row):
    true_category = row['category']
    pred_category = row['category_pred']
    return cost_matrix[true_category][pred_category]

def calculate_cost_cb(row):
    pass