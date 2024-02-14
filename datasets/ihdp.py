## This file has been created by Justine and is used to process the IHDP dataset
import pandas as pd # for data processing
import numpy as np
from sklearn.model_selection import train_test_split # to split the data into test and train

def processing_get_data_ihdp():
    # X1-X21: covariates measured on the infants and mothers.
    # Treatment: Binary variable indicating whether the infant received the intervention (1) or not (0).
    # y_factual: The actual observed outcome for an individual under the treatment they received. This is the outcome that is observed in the real world.
    # y_cfactual: The potential outcome that an individual would have experienced had they received the alternative treatment (the treatment they did not actually receive).
    # mu0: True mean outcome for the control group.
    # mu1: True mean outcome for the treated group.
    data= pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv", header = None)
    # col =  ["treatment", "y_factual", "y_cfactual", "mu0", "mu1" ,"x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x18","x19","x20","x21","x22", "x23", "x24","x25"]
    col =  ["treatment", "y_factual", "y_cfactual", "mu0", "mu1" ,"x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x18","x19","x20","x21","x22", "x23", "x24","x25"]

    data.columns = col
    data.to_csv('ihdp_data.csv', index = False)
    controlled= data[data['treatment'] == False]
    treated = data[data['treatment']==True]
    
    # return train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y

    return data



def processing_transform_data_ihdp(all_data):
    
    # Specify column names
    columns_x = [f"feature_{i}" for i in range(train_x.shape[1])]
    columns_y = ["observed_outcome"]
    #columns_potential_y = [f"potential_outcome_{i}" for i in range(test_potential_y.shape[1])]
    columns_potential_y = ["y_t0", "y_t1"]
    columns_t = ["treatment"]



    x = all_data[["x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x18","x19","x20","x21","x22", "x23", "x24","x25"]]  # Covariates
    y = all_data[['y_factual', "y_cfactual","mu0","mu1"]]  # Outcome
    t = all_data['treatment']  # Treatment assignment indicator
    # Based on the column definitions above we can split the data into 6 different dataframes
    train_x, test_x, train_y, test_y, train_t, test_t = train_test_split(x, y, t, test_size=0.2, random_state=42)
    return train_x, test_x, train_y, test_y, train_t, test_t

all_data = processing_get_data_ihdp()
train_x, test_x, train_y, test_y, train_t, test_t = processing_transform_data_ihdp(all_data)

test_t = pd.DataFrame(test_t)
test_t.columns = ['treatment']
train_t = pd.DataFrame(train_t)
train_t.columns = ['treatment']