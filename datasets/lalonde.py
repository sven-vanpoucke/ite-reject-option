
# Import of the packages.
from .processing import processing_get_data
from sklearn.model_selection import train_test_split # to split the data into test and train

def processing_get_data_lalonde():
    # constants
    url_controlled = 'https://raw.githubusercontent.com/sven-vanpoucke/thesis-data/main/lalonde/nsw_treated.txt'
    url_treated = 'https://raw.githubusercontent.com/sven-vanpoucke/thesis-data/main/lalonde/nsw_control.txt'
    columns = ["training",   # Treatment assignment indicator
            "age",        # Age of participant
            "education",  # Years of education
            "black",      # Indicate whether individual is black
            "hispanic",   # Indicate whether individual is hispanic
            "married",    # Indicate whether individual is married
            "no_degree",  # Indicate if individual has no high-school diploma
            "re75",       # Real earnings in 1974, prior to study participation
            "re78"]       # Real earnings in 1978, after study end
    all_data = processing_get_data(url_controlled, url_treated, columns)
    return all_data

"""
x are all the covariates
y is the observed outcome
t indicates if the treatment happened or not
"""
def processing_transform_data_lalonde(all_data):
    x = all_data[['age', 'education', 'black', 'hispanic', 'married', 'no_degree', 're75']]  # Covariates
    y = all_data['re78']  # Outcome
    t = all_data['training']  # Treatment assignment indicator
    # Based on the column definitions above we can split the data into 6 different dataframes
    train_x, test_x, train_y, test_y, train_t, test_t = train_test_split(x, y, t, test_size=0.2, random_state=42)
    return train_x, test_x, train_y, test_y, train_t, test_t



"""
all_data = processing_get_data_lalonde()
train_x, test_x, train_y, test_y, train_t, test_t = processing_transform_data_lalonde(all_data)


# Print Operations
## Get the size (number of rows and columns)
size = all_data.shape
## Access the number of rows and columns separately
num_rows, num_columns = size[0], size[1]
print(f"The size of the concatenated DataFrame is {num_rows} rows by {num_columns} columns.")
## To see what we have generated - training sets
print("\n train_x")
print(train_x)
print("\n train_y")
print(train_y)
print("\n train_t")
print(train_t)
## To see what we have generated - treatment sets
print("\n test_x")
print(test_x)
print("\n test_y")
print(test_y)
print("\n test_t")
print(test_t)

"""