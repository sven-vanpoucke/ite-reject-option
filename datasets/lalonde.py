""" import of the packages """

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




# Before we can do anything we need to define the columns of our data from left to right
columns = ["training",   # Treatment assignment indicator
           "age",        # Age of participant
           "education",  # Years of education
           "black",      # Indicate whether individual is black
           "hispanic",   # Indicate whether individual is hispanic
           "married",    # Indicate whether individual is married
           "no_degree",  # Indicate if individual has no high-school diploma
           "re75",       # Real earnings in 1974, prior to study participation
           "re78"]       # Real earnings in 1978, after study end


# Change the url to where you can find the data in csv format.
url_controlled = 'https://raw.githubusercontent.com/sven-vanpoucke/Thesis-Data/main/lalonde/nsw_treated.txt'
url_treated = 'https://raw.githubusercontent.com/sven-vanpoucke/Thesis-Data/main/lalonde/nsw_control.txt'

# This is saving the data in two dataframes
controlled = pd.read_csv(url_controlled, delim_whitespace=True, header=None, names=columns)
treated = pd.read_csv(url_treated, delim_whitespace=True, header=None, names=columns)

# We merge the data into one dataframe.
all_data = pd.concat([treated, controlled], ignore_index=True)

# As a last step of verification we input print the data
print(all_data.head())
import pandas as pd

# Assuming 'treated' and 'controlled' are DataFrames
all_data = pd.concat([treated, controlled], ignore_index=True)

# Get the size (number of rows and columns)
size = all_data.shape

# Access the number of rows and columns separately
num_rows, num_columns = size[0], size[1]

print(f"The size of the concatenated DataFrame is {num_rows} rows by {num_columns} columns.")


# First of all we need to define the features and target of our dataset
"""
x are all the covariates
y is the observed outcome
t indicates if the treatment happened or not
"""
x = all_data[['age', 'education', 'black', 'hispanic', 'married', 'no_degree', 're75']]  # Covariates
y = all_data['re78']  # Outcome
t = all_data['training']  # Treatment assignment indicator

# Based on the column definitions above we can split the data into 6 different dataframes
train_x, test_x, train_y, test_y, train_t, test_t = train_test_split(x, y, t, test_size=0.2, random_state=42)

# To see what we have generated - training sets
print("\n train_x")
print(train_x)
print("\n train_y")
print(train_y)
print("\n train_t")
print(train_t)
# To see what we have generated - treatment sets
print("\n test_x")
print(test_x)
print("\n test_y")
print(test_y)
print("\n test_t")
print(test_t)