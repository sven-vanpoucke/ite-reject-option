import pandas as pd
import numpy as np
def load_ihdp_data():
    # Load IHDP data from the URL
    data = pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv", header=None)
    col = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [f"feature_{i}" for i in range(1, 26)]
    data.columns = col

    data["observed_outcome"] = np.where(data["treatment"] == 0, data["y_factual"], data["y_factual"])
    data["y_t0"] = np.where(data["treatment"] == 0, data["y_factual"], data["y_cfactual"])
    data["y_t1"] = np.where(data["treatment"] == 1, data["y_factual"], data["y_cfactual"])
    data.to_csv('ihdp_data.csv', index=False)
    return data

def split_train_test(data):
    # Split the data into train and test
    train_data = data.sample(frac=0.8, random_state=42).copy()
    test_data = data.drop(train_data.index).copy()
    return train_data, test_data

def reset_index_and_copy(dataframe_list):
    # Reset index and create a copy of each DataFrame
    return [df.reset_index(drop=True).copy() for df in dataframe_list]

def preprocessing_get_data_ihdp():
    data = load_ihdp_data()
    train_data, test_data = split_train_test(data)

    train_x, train_t, train_y, train_potential_y = reset_index_and_copy([
        train_data.drop(columns=["treatment", "observed_outcome", "y_factual", "y_cfactual", "y_t0", "y_t1", "mu0", "mu1"]),
        train_data["treatment"],
        train_data["observed_outcome"],
        train_data[["y_t0", "y_t1"]]
    ])

    test_x, test_t, test_y, test_potential_y = reset_index_and_copy([
        test_data.drop(columns=["y_factual", "y_cfactual", "mu0", "mu1"]),
        test_data["treatment"],
        test_data["observed_outcome"],
        test_data[["y_t0", "y_t1"]]
    ])

    return train_x, train_t, train_y, train_potential_y, test_x, test_t, test_y, test_potential_y


def preprocessing_transform_data_ihdp(train_x, train_t, train_y, train_potential_y, test_x, test_y, test_t, test_potential_y):

    # Resetting index and making a copy for each DataFrame
    train_x = train_x.reset_index(drop=True).copy()
    train_t = train_t.reset_index(drop=True).copy()
    train_y = train_y.reset_index(drop=True).copy()
    train_potential_y = train_potential_y.reset_index(drop=True).copy()

    test_x = test_x.reset_index(drop=True).copy()
    test_t = test_t.reset_index(drop=True).copy()
    test_y = test_y.reset_index(drop=True).copy()
    test_potential_y = test_potential_y.reset_index(drop=True).copy()


    columns_x = [f"feature_{i}" for i in range(1, 26)]
    # columns_x = [f"feature_{i}" for i in range(train_x.shape[1])]

    columns_y = ["observed_outcome"]
    columns_potential_y = ["y_t0", "y_t1"]
    columns_t = ["treatment"]

    # Convert NumPy arrays to pandas DataFrames
    train_x = pd.DataFrame(train_x, columns=columns_x)
    train_y = pd.DataFrame(train_y, columns=columns_y)
    train_potential_y = pd.DataFrame(train_potential_y, columns=columns_potential_y)
    train_t = pd.DataFrame(train_t, columns=columns_t)
    
    # Convert NumPy arrays to pandas DataFrames
    test_x = pd.DataFrame(test_x, columns=columns_x)
    test_y = pd.DataFrame(test_y, columns=columns_y)
    test_potential_y = pd.DataFrame(test_potential_y, columns=columns_potential_y)
    test_t = pd.DataFrame(test_t, columns=columns_t)

    return train_x, train_t, train_y, train_potential_y, test_x, test_t, test_y, test_potential_y
