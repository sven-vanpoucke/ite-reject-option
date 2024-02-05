from sklearn.neighbors import NearestNeighbors
import pandas as pd 

def nbrs_train(train_data, n_neighbors=5):
    nbrs_train = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(train_data)
    return nbrs_train

def calculate_ood_distances(nbrs_train, test_data):
    distances_test_to_train, _ = nbrs_train.kneighbors(test_data)
    return distances_test_to_train.mean(axis=1)

def distance_test_to_train(nbrs_train, test_x):
    # Assuming train_treated_x and test_x are Pandas DataFrames
    distance_test_to_train = []

    # Iterate over rows in test_x
    for index, row in test_x.iterrows():
        # Calculate distance for each row in test_x against all train_x
        distance = calculate_ood_distances(nbrs_train, row.values.reshape(1, -1))
        distance_test_to_train.append(distance[0])

    return pd.Series(distance_test_to_train)

def is_out_of_distribution(distance, threshold_distance=3):
    return distance > threshold_distance