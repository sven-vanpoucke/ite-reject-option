# Define a function to apply categories based on the conditions
import pandas as pd
from tabulate import tabulate
import numpy as np

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
        file.write(f" - Upper left cell: amount of cases that have outcome 0: no matter if you would treat or not \n")
        file.write(f"   If treat, they stay alive, if no treat they also stay alive. \n")

        file.write(f" - Under right cell: amount of cases that have outcome 1: no matter if you would treat or not \n")
        file.write(f"   If treat, they die, if no treat they also die. \n")

        file.write(f" - Upper right cell: amount of cases that have outcome 1 if treated, but outcome 0 if not treated \n")
        file.write(f"   If treat, they die, if no treat they stay alive. \n")

        file.write(f" - Under left cell: amount of cases that have outcome 0 if treated, but outcome 1 if not treated \n")
        file.write(f"   If treat, they stay alive, if no treat they die. \n\n")

def calculate_crosstab_matrix_names(t0, t1, data, file_path):

    count_matrix = pd.crosstab(data[t0], data[t1], margins=False)

    # Calculate and write crosstab to the file
    with open(file_path, 'a') as file:
        file.write(f"\nCrosstab for {t0} and {t1}:\n")
        file.write(tabulate(count_matrix, headers='keys', tablefmt='simple_grid'))
        file.write(f"\n")
    
def calculate_crosstab(t0, t1, data, file_path):
    # Apply the categorization function to create the 'Category' column
    #data['category'] = data.apply(categorize, axis=1)
    #data['category_pred'] = data.apply(categorize_pred, axis=1)

    count_matrix = pd.crosstab(data[t0], data[t1], margins=False)

    diagonal_elements = np.diag(count_matrix)
    diagonal_sum = diagonal_elements.sum()

    # Assuming count_matrix is your non-square matrix
    rows, columns= count_matrix.shape
    
    # Keep only the first 'rows' columns
    count_matrix_square = count_matrix.iloc[:, :rows]

    count_matrix_square = count_matrix_square.to_numpy()

    # Count elements above the main diagonal
    above_diagonal_sum = np.sum(np.triu(count_matrix_square, k=1))
    below_diagonal_sum = np.sum(np.tril(count_matrix_square, k=-1))

    accurancy = diagonal_sum / (diagonal_sum+above_diagonal_sum+below_diagonal_sum)*100
    accurancy = accurancy.round(2)
    # Calculate and write crosstab to the file
    with open(file_path, 'a') as file:
        file.write(f"\nCrosstab for {t0} and {t1}:\n")
        file.write(tabulate(count_matrix, headers='keys', tablefmt='simple_grid'))
        file.write(f"Sum of the main diagonal: {diagonal_sum}\n")
        file.write(f"Sum above the main diagonal: {above_diagonal_sum}\n")
        file.write(f"Sum below the main diagonal: {below_diagonal_sum}\n")
        file.write(f"\nAccurancy: {accurancy}%\n")
        file.write(f"\n")
    
    return accurancy

"""
┌───────┬───────┬───────┬───────┐
│ ite   │   -1  │   0   │   1   │
├───────┼───────┼───────┼───────┤
│ -1    │   TP  │   FN  │   FN  │
├───────┼───────┼───────┼───────┤
│  0    │   FP  │   TN  │   FP  │
├───────┼───────┼───────┼───────┤
│  1    │   FN  │   FN  │   TP  │
└───────┴───────┴───────┴───────┘
"""