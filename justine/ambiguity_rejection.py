import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV, cross_val_predict
from propensity_score_matching import estimate_propensity_scores
# Assuming you have a DataFrame df with your data
# Assuming X contains your covariates and y contains the treatment assignment

'''
def bootstrap_confidence_interval(df,X, y, num_bootstraps=1000, alpha=0.05):
    # Store bootstrapped results
    bootstrapped_scores = []

    for _ in range(num_bootstraps):
        # Create a bootstrap sample
        X_boot, y_boot = resample(X, y, replace=True)

        # Calculate propensity scores for the bootstrap sample
        bootstrapped_scores.append(estimate_propensity_scores(df,X,cv= 5,treatment='treatment'))

    # Calculate percentiles for confidence interval
    lower_bound = np.percentile(bootstrapped_scores, (alpha / 2) * 100)
    upper_bound = np.percentile(bootstrapped_scores, (1 - alpha / 2) * 100)

    return lower_bound, upper_bound

# Perform grid search to optimize bounds
param_grid = {
    'num_bootstraps': [100, 500, 1000, 2000],
    'C': [0.001, 0.01, 0.1, 1, 10, 100]  # Adjust the range based on your preferences
}

from data_loader import load_dataset
df, df_data = load_dataset()


estimator = LogisticRegression()  # Replace this with your desired estimator
grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5, scoring='accuracy')

grid_search.fit(X= df_data, y= df['treatment'] )

best_params = grid_search.best_params_
best_num_bootstraps = best_params['num_bootstraps']
best_alpha = best_params['alpha']

# Now, use the best parameters to get the optimized bounds
propensity_scores = estimate_propensity_scores(df,df_data,cv= 5,treatment='treatment')
lower_bound, upper_bound = bootstrap_confidence_interval(df,X= df_data, y= df['treatment'], num_bootstraps=1000, alpha=0.05)

# Identify ambiguous observations
ambiguous_indices = np.where((propensity_scores < lower_bound) | (propensity_scores > upper_bound))[0]


print(ambiguous_indices)

# Assign 'R' to ambiguous observations
df.loc[ambiguous_indices, 't_predicted'] = 'R'

print(df)





import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV

# Assuming you have a DataFrame df with your data
# Assuming X contains your covariates and y contains the treatment assignment

def estimate_propensity_scores(X, y, cv=5):
    """
    Estimate propensity scores using logistic regression.

    Parameters:
    - X: DataFrame with covariates
    - y: Series with treatment assignment (0 or 1)
    - cv: Number of cross-validation folds for logistic regression

    Returns:
    - propensity_scores: Numpy array containing propensity scores
    """
    y = y.astype(int)
    # Train a logistic regression model to estimate propensity scores using cross-validation
    propensity_model = LogisticRegression(random_state=42)
    propensity_scores = cross_val_predict(propensity_model, X, y, cv=cv, method='predict_proba')[:, 1]


    return propensity_scores

'''
def bootstrap_confidence_interval(t_true, df_data, num_bootstraps = 1000, alpha = 0.05):
    bootstrapped_scores = []
    t_true = t_true.astype(bool)

    for _ in range(num_bootstraps):
        # Create a bootstrap sample
        df_boot = pd.concat([resample(df_data, random_state=np.random.randint(1000)).reset_index(drop=True),
                            resample(t_true, random_state=np.random.randint(1000)).reset_index(drop=True)], axis=1)

        # Estimate propensity scores for the bootstrap sample
        propensity_model = LogisticRegression(random_state=42)
        propensity_scores = cross_val_predict(propensity_model, df_data, t_true, cv=5, method='predict_proba')[:, 1]

        # Store propensity scores
        bootstrapped_scores.append(np.median(propensity_scores))

    # Calculate the point estimate (median) of the bootstrapped scores
    point_estimate = np.median(bootstrapped_scores)

    # Calculate standard error
    se = np.std(bootstrapped_scores)

    # Calculate confidence interval
    lower_bound = 0.5 - 1.96 * se
    upper_bound = 0.5 + 1.96 * se

    return lower_bound, upper_bound




def ambiguity_rejection_function(df, propensity_scores, lower_bound, upper_bound):
    # Identify ambiguous observations
    ambiguous_indices = df[(propensity_scores > lower_bound) | (propensity_scores < upper_bound)].index

    # Assign 'R' to ambiguous observations
    df.loc[ambiguous_indices, 't_predicted'] = 'R'
    print(ambiguous_indices)

    return df



