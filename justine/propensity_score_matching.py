import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def estimate_propensity_scores(df,df_data,cv= 5,treatment="treatment"):
                               
    t_true = df["treatment"].astype(int)
    # Train a logistic regression model to estimate propensity scores using cross-validation
    propensity_model = LogisticRegression(random_state=42)
    propensity_scores = cross_val_predict(propensity_model, df_data,t_true, cv=5, method='predict_proba')[:, 1]

    # Add propensity scores to the DataFrame
    df['propensity_score'] = propensity_scores
    return df


def perform_matching(df, min_treatment_count, k_neighbors=10):
    # Fit Nearest Neighbors on the entire dataset's propensity scores
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(df['propensity_score'].values.reshape(-1, 1))

    # Find nearest neighbors for each unit, including both treated and control
    distances, indices = nn.kneighbors(df[['propensity_score']])
    '''
    # Extract indices for closest neighbors
    closest_indices = indices[:, 1:]

    # Count the number of treated neighbors for each unit
    #do a loop to handle missing indices (when working with accepted subset)
    treated_counts = []
    # Count the number of treated neighbors for each unit
    treated_counts = df.loc[closest_indices.flatten(), 'treatment'].values.reshape(-1, k_neighbors).sum(axis=1)
    '''
     # Extract indices for closest neighbors
    closest_indices = indices[:, 1:]

    # Count the number of treated neighbors for each unit
    treated_counts = df.loc[closest_indices.flatten(), 'treatment'].values.reshape(-1, k_neighbors).sum(axis=1)

    # Determine predicted treatment based on the minimum treated count criterion
    predicted_treatment = (treated_counts >= min_treatment_count).astype(int)
    # Determine predicted treatment based on the minimum treated count criterion
    predicted_treatment = (treated_counts >= min_treatment_count).astype(int)

    # Create a DataFrame with matched pairs and relevant information
    matched_pairs = pd.DataFrame({
        'index': df.index,
        'treatment': df['treatment'].values,
        't_predicted': predicted_treatment
    })
    df['t_predicted']= predicted_treatment
    return matched_pairs,df



def propensity_score_matching_function(df,df_data, min_treatment_count,treatment='treatment',k_neighbors=10):
    # Estimate propensity scores
    df = estimate_propensity_scores(df,df_data,cv=5,treatment = "treatment")
    matched_pairs, df = perform_matching(df,min_treatment_count,k_neighbors=k_neighbors)
    return matched_pairs,df
