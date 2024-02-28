import pandas as pd
import numpy as np
import os

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    performs preprocessing on the provided dataset

    :param DataFrame df: the dataframe on which to perform preprocessing 
    """
    column_names =  ["treatment", "y_factual", "y_cfactual", "mu0", "mu1" ,]
 
    for i in range(1,26):
        column_names.append("x"+str(i))
    df.columns = column_names
    df['ITE'] = df['y_factual'] - df['y_cfactual']
    df = df.astype({"treatment": bool})
    df_data=df.drop(columns=["y_factual", "y_cfactual", "ITE", "treatment","mu0","mu1"])
    return df, df_data

def load_dataset() -> pd.DataFrame:
    """
    loads and preprocesses teh datset for further analysis
    data = pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv", header=None)
    
    col = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25"]
    
    data.columns = col
    data.to_csv('ihdp_data.csv', index=False)
    

    """
    df = pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv", header=None)
    df, df_data = preprocess_dataset(df)
    df.to_csv('ihdp_data.csv', index=False)
    return df, df_data

if __name__ == '__main__':
    df = pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv", sep=',', header = None)
    print(df.head())



