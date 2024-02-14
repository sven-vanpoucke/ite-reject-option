## This file has been created by Justine and is used to process the IHDP dataset
import pandas as pd # for data processing
import numpy as np
from sklearn.model_selection import train_test_split # to split the data into test and train

def preprocessing_get_data_ihdp():
    # X1-X21: covariates measured on the infants and mothers.
    # Treatment: Binary variable indicating whether the infant received the intervention (1) or not (0).
    # y_factual: The actual observed outcome for an individual under the treatment they received. This is the outcome that is observed in the real world.
    # y_cfactual: The potential outcome that an individual would have experienced had they received the alternative treatment (the treatment they did not actually receive).
    # mu0: True mean outcome for the control group.
    # mu1: True mean outcome for the treated group.
    data = pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv", header=None)
    
    col = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25"]
    
    data.columns = col
    data.to_csv('ihdp_data.csv', index=False)
    
    # Split the data into train and test
    train_data = data.sample(frac=0.8, random_state=42).copy()
    test_data = data.drop(train_data.index).copy()
    
    train_x = train_data.drop(columns=["y_factual", "y_cfactual", "mu0", "mu1"]).copy()
    train_t = train_data["treatment"].copy()
    train_y = train_data["y_factual"].copy()
    train_potential_y = train_data["y_cfactual"].copy()
    
    test_x = test_data.drop(columns=["y_factual", "y_cfactual", "mu0", "mu1"]).copy()
    test_t = test_data["treatment"].copy()
    test_y = test_data["y_factual"].copy()
    test_potential_y = test_data["y_cfactual"].copy()

    # Resetting index to avoid Unalignable boolean Series error
    train_x.reset_index(drop=True)
    train_y.reset_index(drop=True)
    train_potential_y.reset_index(drop=True)
    train_t.reset_index(drop=True)
    
    test_x.reset_index(drop=True)
    test_y.reset_index(drop=True)
    test_potential_y.reset_index(drop=True)
    test_t.reset_index(drop=True)
    
    return train_x, train_t, train_y, train_potential_y, test_x, test_t, test_y, test_potential_y

def preprocessing_transform_data_ihdp(train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y):
    # Specify column names
    columns_x = [f"feature_{i}" for i in range(train_x.shape[1])]
    columns_y = ["observed_outcome"]
    #columns_potential_y = [f"potential_outcome_{i}" for i in range(test_potential_y.shape[1])]
    columns_potential_y = ["y_t0", "y_t1"]
    columns_t = ["treatment"]

    # Convert NumPy arrays to pandas DataFrames
    train_x = pd.DataFrame(train_x, columns=columns_x).copy()
    train_y = pd.DataFrame(train_y, columns=columns_y).copy()
    train_potential_y = pd.DataFrame(train_potential_y, columns=columns_potential_y).copy()
    train_t = pd.DataFrame(train_t, columns=columns_t).copy()
    
    test_x = pd.DataFrame(test_x, columns=columns_x).copy()
    test_y = pd.DataFrame(test_y, columns=columns_y).copy()
    test_potential_y = pd.DataFrame(test_potential_y, columns=columns_potential_y).copy()
    test_t = pd.DataFrame(test_t, columns=columns_t).copy()

    # Resetting index to avoid Unalignable boolean Series error
    train_x.reset_index(drop=True)
    train_y.reset_index(drop=True)
    train_potential_y.reset_index(drop=True)
    train_t.reset_index(drop=True)
    
    test_x.reset_index(drop=True)
    test_y.reset_index(drop=True)
    test_potential_y.reset_index(drop=True)
    test_t.reset_index(drop=True)

    return train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y