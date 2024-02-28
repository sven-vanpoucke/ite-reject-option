import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def calculate_propensity_scores(x, t):
    """
    Estimates propensity scores using logistic regression.

    Args:
        x (pandas.DataFrame): Features used for model training.
        t (pandas.Series): Treatment indicator (treated: 1, control: 0).

    Returns:
        pandas.Series: Propensity scores for each individual.
    """

    # Fit logistic regression model
    logistic_model = LogisticRegression(random_state=42)
    logistic_model.fit(x, t)

    # Predict propensity scores
    propensity_scores = logistic_model.predict_proba(x)[:, 1]

    return propensity_scores

def knn_matching(treated_x, control_x, control_y, n_neighbors=1):
    """
    Performs KNN matching for propensity score matching.

    Args:
        treated_x (pandas.DataFrame): Features of treated units.
        control_x (pandas.DataFrame): Features of control units.
        n_neighbors (int, optional): Number of nearest neighbors to find. Defaults to 1.

    Returns:
        pandas.DataFrame: Matched control data (features and outcome).
    """

    # Standardize features (assuming you want to standardize)
    scaler = StandardScaler()
    treated_x_scaled = scaler.fit_transform(treated_x)
    control_x_scaled = scaler.transform(control_x)

    # Fit KNN model on control group
    knn_model = NearestNeighbors(n_neighbors=n_neighbors)
    knn_model.fit(control_x_scaled)

    # Find nearest neighbors for each treated unit
    distances, indices = knn_model.kneighbors(treated_x_scaled)

    # Get matched control units and corresponding outcome variables
    matched_control_x = control_x.iloc[indices.flatten()]

    # Assuming you have the outcome variable in a separate column named 'outcome'
    matched_control_y = control_y['observed_outcome'].iloc[indices.flatten()]

    # # Combine features and outcome into a single dataframe
    # matched_control = pd.concat([matched_control_x, matched_control_y.to_frame('outcome')], axis=1)

    return matched_control_x, matched_control_y
