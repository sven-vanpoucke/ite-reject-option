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

from scipy.special import expit

def preprocessing_get_data_twin(train_rate=0.8):
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

def preprocessing_transform_data_twin(train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y):
    # Specify column names
    columns_x = [f"feature_{i}" for i in range(train_x.shape[1])]
    columns_y = ["observed_outcome"]
    #columns_potential_y = [f"potential_outcome_{i}" for i in range(test_potential_y.shape[1])]
    columns_potential_y = ["y_t0", "y_t1"]
    columns_t = ["treatment"]

    # Convert NumPy arrays to pandas DataFrames
    train_x = pd.DataFrame(train_x, columns=columns_x)
    train_y = pd.DataFrame(train_y, columns=columns_y)
    train_potential_y = pd.DataFrame(train_potential_y, columns=columns_potential_y)
    train_t = pd.DataFrame(train_t, columns=columns_t)
    
    test_x = pd.DataFrame(test_x, columns=columns_x)
    test_y = pd.DataFrame(test_y, columns=columns_y)
    test_potential_y = pd.DataFrame(test_potential_y, columns=columns_potential_y)
    test_t = pd.DataFrame(test_t, columns=columns_t)

    return train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y

"""


# Now you can call the function to load the data
train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y = preprocessing_get_data_twin()
train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y = preprocessing_transform_data_twin(train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y)


# Display the heads of DataFrames using tabulate
print("\n train_x")
print(tabulate(train_x.head(), headers='keys', tablefmt='psql'))

print("\n train_y")
print(tabulate(train_y.head(), headers='keys', tablefmt='psql'))

print("\n train__potential_y")
print(tabulate(train_potential_y.head(), headers='keys', tablefmt='psql'))

# Filtering rows where column 1 is not equal to column 2
unequal_rows = train_potential_y[train_potential_y.iloc[:, 0] != train_potential_y.iloc[:, 1]]

# Displaying the filtered DataFrame
print(unequal_rows)


print("\n train_t")
print(tabulate(train_t.head(), headers='keys', tablefmt='psql'))

print("\n test_x")
print(tabulate(test_x.head(), headers='keys', tablefmt='psql'))

print("\n test_y")
print(tabulate(test_y.head(), headers='keys', tablefmt='psql'))

print("\n test_t")
print(tabulate(test_t.head(), headers='keys', tablefmt='psql'))

"""