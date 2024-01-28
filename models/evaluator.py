# Define a function to apply categories based on the conditions
def categorize(row):
    if row['y_t0'] == 0 and row['y_t1'] == 0:
        return 'Lost Cause'
    elif row['y_t0'] == 1 and row['y_t1'] == 0:
        return 'Sleeping Dog'
    elif row['y_t0'] == 0 and row['y_t1'] == 1:
        return 'Persuadable' # (can be rescued)
    elif row['y_t0'] == 1 and row['y_t1'] == 1:
        return 'Sure Thing'

def categorize_pred(row):
    if row['y_t0_pred'] == 0 and row['y_t1_pred'] == 0:
        return 'Lost Cause'
    elif row['y_t0_pred'] == 1 and row['y_t1_pred'] == 0:
        return 'Sleeping Dog'
    elif row['y_t0_pred'] == 0 and row['y_t1_pred'] == 1:
        return 'Persuadable' # (can be rescued)
    elif row['y_t0_pred'] == 1 and row['y_t1_pred'] == 1:
        return 'Sure Thing'
