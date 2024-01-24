# Import of the packages.

# Data processing
import pandas as pd # for data processing
import numpy as np

# Packages from "sklearn"
from sklearn.model_selection import train_test_split # to split the data into test and train
from sklearn.metrics import mean_squared_error # to calulcate MSE
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR

# Packages for Data Visualization
from matplotlib import pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Message will be shown if executed succesfully.
print("Finished imports")
from scipy.special import expit

def data_loading_twin(train_rate=0.8):
    """Load twins data.

    Args:
        - train_rate: the ratio of training data

    Returns:
        - train_x: features in training data
        - train_t: treatments in training data
        - train_y: observed outcomes in training data
        - train_potential_y: potential outcomes in training data
        - test_x: features in testing data
        - test_y: observed outcomes in testing data
        - test_t: treatments in testing data
        - test_potential_y: potential outcomes in testing data
    """
    # Change the url to where you can find the data in csv format.
    url = 'https://raw.githubusercontent.com/sven-vanpoucke/Thesis-Data/main/twins/twin_data.csv'

    # Read the CSV data into a pandas DataFrame and remove the quotes from the column names.
    # all_data = pd.read_csv(url, quotechar="'", header=0)  # Uncomment this line if you want to use pandas

    # Load original data (11400 patients, 30 features, 2-dimensional potential outcomes)
    ori_data = np.loadtxt(url, delimiter=",", skiprows=1)

    # Define features
    x = ori_data[:, :30]
    no, dim = x.shape

    # Define potential outcomes
    potential_y = ori_data[:, 30:]
    # Die within 1 year = 1, otherwise = 0
    potential_y = np.array(potential_y < 9999, dtype=float)

    ## Assign treatment
    coef = np.random.uniform(-0.01, 0.01, size=[dim, 1])
    prob_temp = expit(np.matmul(x, coef) + np.random.normal(0, 0.01, size=[no, 1]))

    prob_t = prob_temp / (2 * np.mean(prob_temp))
    prob_t[prob_t > 1] = 1

    t = np.random.binomial(1, prob_t, [no, 1])
    t = t.reshape([no, ])

    ## Define observable outcomes
    y = np.zeros([no, 1])
    y = np.transpose(t) * potential_y[:, 1] + np.transpose(1 - t) * potential_y[:, 0]
    y = np.reshape(np.transpose(y), [no, ])

    ## Train/test division
    idx = np.random.permutation(no)
    train_idx = idx[:int(train_rate * no)]
    test_idx = idx[int(train_rate * no):]

    train_x = x[train_idx, :]
    train_t = t[train_idx]
    train_y = y[train_idx]
    train_potential_y = potential_y[train_idx, :]

    test_x = x[test_idx, :]
    test_y = y[test_idx]
    test_t = t[test_idx]
    test_potential_y = potential_y[test_idx, :]

    return train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y

# Now you can call the function to load the data
train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y = data_loading_twin()

print(train_potential_y)
# Convert NumPy arrays to pandas DataFrames
train_x_df = pd.DataFrame(train_x)
train_y_df = pd.DataFrame(train_y)
train_potential_y_df = pd.DataFrame(train_potential_y)
train_t_df = pd.DataFrame(train_t)
test_x_df = pd.DataFrame(test_x)
test_y_df = pd.DataFrame(test_y)
test_t_df = pd.DataFrame(test_t)

# Display the heads of DataFrames using tabulate
print("\n train_x")
print(tabulate(train_x_df.head(), headers='keys', tablefmt='psql'))

print("\n train_y")
print(tabulate(train_y_df.head(), headers='keys', tablefmt='psql'))

print("\n train__potential_y")
print(tabulate(train_potential_y_df.head(), headers='keys', tablefmt='psql'))

# Filtering rows where column 1 is not equal to column 2
unequal_rows = train_potential_y_df[train_potential_y_df.iloc[:, 0] != train_potential_y_df.iloc[:, 1]]

# Displaying the filtered DataFrame
print(unequal_rows)


print("\n train_t")
print(tabulate(train_t_df.head(), headers='keys', tablefmt='psql'))

print("\n test_x")
print(tabulate(test_x_df.head(), headers='keys', tablefmt='psql'))

print("\n test_y")
print(tabulate(test_y_df.head(), headers='keys', tablefmt='psql'))

print("\n test_t")
print(tabulate(test_t_df.head(), headers='keys', tablefmt='psql'))



import numpy as np

# Calculate the ATE for the training set
N_treated_train = np.sum(train_t == 1)  # Number of treated individuals in training set
N_control_train = np.sum(train_t == 0)  # Number of control individuals in training set

# Calculate the average outcomes for treated and control groups in the training set
ATE_train = (np.mean(train_y[train_t == 1]) - np.mean(train_y[train_t == 0]))

# Calculate the ATE for the test set
N_treated_test = np.sum(test_t == 1)  # Number of treated individuals in test set
N_control_test = np.sum(test_t == 0)  # Number of control individuals in test set

# Calculate the average outcomes for treated and control groups in the test set
ATE_test = (np.mean(test_y[test_t == 1]) - np.mean(test_y[test_t == 0]))

# Display the calculated ATEs for training and test sets
print("ATE for training set:", ATE_train)
print("ATE for test set:", ATE_test)


# 3,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,0,0,20,1,31,25,13,1,2,1,12,1,9999,9999
# train_x: |  0 |   3 |   1 |   2 |   2 |   2 |   2 |   2 |   2 |   2 |   2 |    2 |    2 |    2 |    2 |    2 |    2 |    2 |    1 |    0 |    0 |   20 |    1 |   31 |   25 |   13 |    1 |    2 |    1 |   12 |    1 |
# train_y: |  0 |   0 |
# train_t: |  0 |   1 |

