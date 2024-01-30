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

def instructions_matrix(file_path):
    with open(file_path, 'a') as file:
        file.write(f"We make the matrix: Lost Cause, Sleeping Dog, Persuadable, Sure Thing \n")
        file.write(f"Comment: \n")
        file.write(f"Upper left cell: amount of cases that have outcome 0: no matter if you would treat or not \n")
        file.write(f"If treat, they stay alive, if no treat they also stay alive. \n\n")

        file.write(f"Under right cell: amount of cases that have outcome 1: no matter if you would treat or not \n")
        file.write(f"If treat, they die, if no treat they also die. \n\n")

        file.write(f"Upper right cell: amount of cases that have outcome 1 if treated, but outcome 0 if not treated \n")
        file.write(f"If treat, they die, if no treat they stay alive. \n\n")

        file.write(f"Under left cell: amount of cases that have outcome 0 if treated, but outcome 1 if not treated \n")
        file.write(f"If treat, they stay alive, if no treat they die. \n\n")

def calculate_crosstab(t0, t1, data, file_path):
    # Apply the categorization function to create the 'Category' column
    data['category'] = data.apply(categorize, axis=1)
    data['category_pred'] = data.apply(categorize_pred, axis=1)

    # Calculate and write crosstab to the file
    with open(file_path, 'a') as file:
        file.write(f"\nCrosstab for {t0} and {t1}:\n")
        count_matrix = pd.crosstab(data[t0], data[t1], margins=False)
        file.write(tabulate(count_matrix, headers='keys', tablefmt='simple_grid'))
