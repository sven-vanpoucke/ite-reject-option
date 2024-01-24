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
#from lime.lime_tabular import LimeTabularExplainer


# Message will be shown if executed succesfully.
print("Finished imports")