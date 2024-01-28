
# Import of the packages.
import pandas as pd

# This is saving the data in two dataframes
def processing_get_data(url_controlled, url_treated, columns):
    controlled = pd.read_csv(url_controlled, delim_whitespace=True, header=None, names=columns)
    treated = pd.read_csv(url_treated, delim_whitespace=True, header=None, names=columns)
    all_data = pd.concat([treated, controlled], ignore_index=True)
    return all_data

