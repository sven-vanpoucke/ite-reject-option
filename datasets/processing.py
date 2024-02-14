
# Import of the packages.
import pandas as pd

# This is saving the data in two dataframes
def processing_get_data(url_controlled, url_treated, columns):
    controlled = pd.read_csv(url_controlled, delim_whitespace=True, header=None, names=columns)
    treated = pd.read_csv(url_treated, delim_whitespace=True, header=None, names=columns)
    all_data = pd.concat([treated, controlled], ignore_index=True)
    return all_data

# Splitting the data into treated and control groups
def preprocessing_split_t_c_data(train_x, train_y, train_t, test_x, test_y, test_t):
  
  # for training data
  train_treated_x = train_x[train_t['treatment'] == 1]
  train_control_x = train_x[train_t['treatment'] == 0] # no treatment given...
  train_treated_y = train_y[train_t['treatment'] == 1]
  train_control_y = train_y[train_t['treatment'] == 0]
  
  # for test data
  test_treated_x = test_x[test_t['treatment'] == 1]
  test_control_x = test_x[test_t['treatment'] == 0] #no treatment given...
  test_treated_y = test_y[test_t['treatment'] == 1]
  test_control_y = test_y[test_t['treatment'] == 0]

  return train_treated_x, train_control_x, train_treated_y, train_control_y, test_treated_x, test_control_x, test_treated_y, test_control_y