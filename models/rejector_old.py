from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import confusion_matrix


digits = 4



def reject():
  





for k in [2,3,4]:
  print("k = ")
  print(k)

  """
  proces for the treated group
  """
  # Methode 1: Berekenen van 'out of distribution' op basis van afstand tot de trainingsgegevens
  # Initialize the Nearest Neighbors model with your training data
  nbrs_train = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(treated_x)

  # Methode 2: Een andere benadering (bijv. statistische methoden, dichtheidsmetingen, etc.)
  # Calculate distances to the nearest neighbors for treated_x_test observations compared to treated_x
  distances_test_to_train, _ = nbrs_train.kneighbors(treated_x_test)

  # Methode 3: Gebruik een ander model om 'out of distribution' te bepalen
  # Set a threshold distance based on your criteria
  std_distance = np.std(distances_test_to_train)
  k = k  # Set your desired multiplier
  threshold_distance = k * std_distance  # 'k' can be a predefined constant (e.g., 2)

  # Identify out-of-distribution observations based on distances
  is_out_of_dist = distances_test_to_train.mean(axis=1) > threshold_distance

  # Convert boolean values to binary (0 or 1)
  out_of_dist_binary = is_out_of_dist.astype(int)

  # Create a DataFrame to store the observations and their binary out-of-distribution status
  df_treated_x_test = pd.DataFrame(treated_x_test, columns=[f"feature_{i+1}" for i in range(treated_x_test.shape[1])])
  df_treated_x_test['Is_Out_Of_Distribution'] = out_of_dist_binary

  # Make predictions using treated_x_test
  treated_y_pred = treated_model.predict(treated_x_test)

  # Conditionally assign values based on out_of_dist_binary
  treated_y_pred[out_of_dist_binary == 1] = 3

  """
  Metrics for treated group
  """

  valid_treated_y_test = treated_y_test[out_of_dist_binary == 0]
  valid_treated_y_pred = treated_y_pred[out_of_dist_binary == 0]

  treated_confusion_matrix = confusion_matrix(valid_treated_y_test, valid_treated_y_pred)
  treated_TP, treated_FP, treated_FN, treated_TN = treated_confusion_matrix.ravel()
  # Count observations marked as out of distribution for treated and control groups
  treated_out_of_dist_count = df_treated_x_test['Is_Out_Of_Distribution'].sum()


  """
  proces for the control group
  """
  # Initialize the Nearest Neighbors model with your training data
  nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(control_x_test)

  # Calculate distances to the nearest neighbors for test observations
  distances, _ = nbrs.kneighbors(control_x_test)

  # Set a threshold distance based on your criteria
  std_distance = np.std(distances)
  k = k
  threshold_distance = k * std_distance  # 'k' can be a predefined constant (e.g., 2)

  # Identify out-of-distribution observations based on distances
  is_out_of_dist = distances.mean(axis=1) > threshold_distance

  # Convert boolean values to binary (0 or 1)
  out_of_dist_binary = is_out_of_dist.astype(int)

  # Create a DataFrame to store the observations and their binary out-of-distribution status
  df_control_x_test = pd.DataFrame(control_x_test, columns=[f"feature_{i+1}" for i in range(control_x_test.shape[1])])
  df_control_x_test['Is_Out_Of_Distribution'] = out_of_dist_binary

  # Make predictions using control_x_test
  control_y_pred = treated_model.predict(control_x_test)

  # Conditionally assign values based on out_of_dist_binary
  control_y_pred[out_of_dist_binary == 1] = 3

  # Filter predictions for treated group where out_of_dist_binary equals 0
  valid_control_y_test = control_y_test[out_of_dist_binary == 0]
  valid_control_y_pred = control_y_pred[out_of_dist_binary == 0]

  # Calculate confusion matrices for treated and control groups

  control_confusion_matrix = confusion_matrix(valid_control_y_test, valid_control_y_pred)

  # Extract TP, TN, FP, FN from the confusion matrices

  control_TP, control_FP, control_FN, control_TN = control_confusion_matrix.ravel()

  control_out_of_dist_count = df_control_x_test['Is_Out_Of_Distribution'].sum()

  """
  Metrics: Confusion matrix
  """

  # Calculate precision, recall, and F1 manually
  treated_precision = (treated_TP / (treated_TP + treated_FP)).round(digits)
  treated_recall = (treated_TP / (treated_TP + treated_FN)).round(digits)
  treated_f1 = (2 * (treated_precision * treated_recall) / (treated_precision + treated_recall)).round(digits)

  control_precision = (control_TP / (control_TP + control_FP)).round(digits)
  control_recall = (control_TP / (control_TP + control_FN)).round(digits)
  control_f1 = (2 * (control_precision * control_recall) / (control_precision + control_recall)).round(digits)

  # Prepare data for the table with additional metrics
  data_with_metrics = [
      ["True Positives", treated_TP, control_TP],
      ["False Positives", treated_FP, control_FP],
      ["False Negatives", treated_FN, control_FN],
      ["True Negatives", treated_TN, control_TN],
      ["Precision", treated_precision, control_precision],
      ["Recall", treated_recall, control_recall],
      ["F1 Score", treated_f1, control_f1],
      ["Out of Distribution", treated_out_of_dist_count, control_out_of_dist_count]
  ]

  # Display data in a table with additional metrics
  print(tabulate(data_with_metrics, headers=["Metric", "Treated Group", "Control Group"], tablefmt="pretty"))