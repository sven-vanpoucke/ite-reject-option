# Define a function to apply categories based on the conditions
import pandas as pd
from tabulate import tabulate
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import math
from sklearn.metrics import mean_squared_error


def calculate_rejection_rate(data):
    """
    Calculate the rejection rate based on the given data.

    Args:
        data (pandas.DataFrame): The data containing the 'ite_pred' column.

    Returns:
        float: The rejection rate, which is the proportion of cases rejected by the model.
    """
    # Calculate the rejection rate
    length_before_rejection = len(data)
    data = data[data['ite_reject'] != 'R'].copy()
    length_after_rejection = len(data)
    rr = (length_before_rejection - length_after_rejection) / length_before_rejection
    return rr

def calculate_micro_metrics(total_tp, total_fp, total_fn, total_tn):
    """
    Calculate micro metrics based on the given true positive (TP), false positive (FP),
    false negative (FN), and true negative (TN) values.

    Args:
        total_tp (int): Total number of true positives.
        total_fp (int): Total number of false positives.
        total_fn (int): Total number of false negatives.
        total_tn (int): Total number of true negatives.

    Returns:
        tuple: A tuple containing the following micro metrics:
            - accuracy (float): The accuracy of the classification model.
            - micro_tpr (float): True Positive Rate (TPR), also known as Recall or Sensitivity.
            - micro_fpr (float): False Positive Rate (FPR).
            - micro_specificity (float): Specificity, which is the complement of FPR.
            - micro_precision (float): Precision, which is the ratio of TP to the total predicted positives.
            - micro_f1score (float): F1 score, which is the harmonic mean of TPR and precision.
    """
    
    # Calculate micro metrics
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fn + total_fp)
    micro_tpr = total_tp / (total_tp + total_fn) # TPR = Recall = Sensitivity
    micro_fpr = total_fp / (total_fp + total_tn) 
    micro_specificity = 1 - (total_tn / (total_fp + total_tn))
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) != 0 else 1.0
    micro_f1score = 2 * (micro_tpr * micro_precision) / (micro_tpr + micro_precision) if (micro_tpr + micro_precision) != 0 else 0

    return accuracy, micro_tpr, micro_fpr, micro_specificity, micro_precision, micro_f1score


def calculate_macro_metrics(total_tpr, total_fpr, total_specificity, total_precision, unique_labels):
    """
    Calculates the macro metrics for a multi-class classification problem.

    Args:
        total_tpr (float): The total true positive rate (TPR) across all classes.
        total_fpr (float): The total false positive rate (FPR) across all classes.
        total_specificity (float): The total specificity across all classes.
        total_precision (float): The total precision across all classes.
        unique_labels (list): A list of unique labels/classes.

    Returns:
        tuple: A tuple containing the macro TPR, macro FPR, macro specificity, macro precision, and macro F1-score.
    """
    # Calculate macro metrics
    macro_tpr = total_tpr / len(unique_labels) # TPR = Recall = Sensitivity
    macro_fpr = total_fpr / len(unique_labels)
    macro_specificity = total_specificity / len(unique_labels)
    macro_precision = total_precision / len(unique_labels)
    macro_f1score = 2 * (macro_tpr * macro_precision) / (macro_tpr + macro_precision) if (macro_tpr + macro_precision) != 0 else 0

    return macro_tpr, macro_fpr, macro_specificity, macro_precision, macro_f1score

def calculate_threedroc(tpr, fpr, rr):
    """
    Calculates the distance in 3D space between the point (0, 1, 0) and the point (fpr, tpr, rr).

    Parameters:
    tpr (float): True Positive Rate.
    fpr (float): False Positive Rate.
    rr (float): Rejection Rate.

    Returns:
    float: The distance in 3D space between the best points and actual point.
    """
    distance_threedroc = math.sqrt((0 - fpr)**2 + (1 - tpr)**2 + (0 - rr)**2)
    return distance_threedroc

def calculate_sliced_confusion_matrix_metrics(unique_labels, confusion_matrix):
    """
    Calculate metrics for each class based on the given confusion matrix.

    Args:
        unique_labels (numpy.ndarray): Array of unique class labels.
        confusion_matrix (numpy.ndarray): Confusion matrix.

    Returns:
        dict: Dictionary containing metrics for each class.
    """
    # Step 2: Calculate the metrics (ignoring rejected instances)
    class_metrics = {}
    total_tpr, total_fpr, total_specificity, total_precision = 0, 0, 0, 0
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    for label in unique_labels:
        # Index of the class in the confusion matrix
        class_index = np.where(unique_labels == label)[0][0]
        
        # Rows and columns corresponding to the class
        true_positive = confusion_matrix[class_index, class_index]
        false_positive = confusion_matrix[:, class_index].sum() - true_positive
        false_negative = confusion_matrix[class_index, :].sum() - true_positive
        true_negative = confusion_matrix.sum() - (true_positive + false_positive + false_negative)
        
        # Create a confusion matrix for the class
        confusion_matrix_class = np.array([[true_negative, false_positive], [false_negative, true_positive]])
        
        # Calculate TPR and FPR for the class
        tpr_class = true_positive / (true_positive + false_negative)
        fpr_class = false_positive / (false_positive + true_negative)
        specificity_class = 1 - fpr_class
        precision_class = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 1.0

        # Store the metrics in the dictionary
        class_metrics[label] = {
            'confusion_matrix': confusion_matrix_class,
            'tpr': tpr_class,
            'fpr': fpr_class,
            'specificity': specificity_class
        }

        # Accumulate for micro averaging
        total_tp += true_positive
        total_fp += false_positive
        total_fn += false_negative
        total_tn += true_negative

        # Accumulate metrics for macro averaging
        total_tpr += tpr_class
        total_fpr += fpr_class
        total_specificity += specificity_class
        total_precision += precision_class

        return total_tp, total_fp, total_fn, total_tn, total_tpr, total_fpr, total_specificity, total_precision, unique_labels

def calculate_performance_metrics_penalty_rejection(data):
    # Add column: prediction correct? T/F
    data['ite_correct'] = data.apply(lambda row: True if row['ite'] == row['ite_pred'] else False, axis=1)
    data['ite_rejected'] = data.apply(lambda row: True if row['ite_reject'] == 'R' else False, axis=1)

    # Prediction correct: yes-no
    # Rejection: yes-no
    # Assuming ite_correct and ite_rejected are boolean columns
    correctly_accepted = data[(data['ite_correct'] == True) & (data['ite_rejected'] == False)] # TA Prediction is correct, and the item is not rejected = True accepted
    correctly_rejected = data[(data['ite_correct'] == False) & (data['ite_rejected'] == True)] # TR Prediction is incorrect, and the item is rejected
    incorrectly_rejected = data[(data['ite_correct'] == True) & (data['ite_rejected'] == True)] # FR Prediction is correct, but the item is rejected
    incorrectly_accepted = data[(data['ite_correct'] == False) & (data['ite_rejected'] == False)] # FA Prediction is incorrect, and the item is not rejected

    TA = len(correctly_accepted)
    FA = len(incorrectly_accepted)
    TR = len(correctly_rejected)
    FR = len(incorrectly_rejected)

    accurancy_rejection = TA / (TA + FA) if (TA + FA) != 0 else 0
    coverage_rejection = (TA+FA) / (TA+FA+FR+TR)

    #Evaluating models with a fixed rejection rate
    prediction_quality = TA / (TA + FA) if (TA + FA) != 0 else 0
    rejection_quality = (TR/FR) / ((FA+TR)/(TA+FR)) if (TA+FR) != 0 and FR != 0 else 0
    combined_quality = (TA+TR) / (TA+FA+FR+TR)

    return accurancy_rejection, coverage_rejection, prediction_quality, rejection_quality, combined_quality


def calculate_performance_metrics(value, value_pred, data, file_path, print=False):
    """
    Calculate the crosstab of two columns in a DataFrame and perform evaluation metrics.
    # Info on: https://www.v7labs.com/blog/confusion-matrix-guide

    Args:
        value (str): The name of the column representing the actual values.
        value_pred (str): The name of the column representing the predicted values.
        data (pandas.DataFrame): The DataFrame containing the data.
        file_path (str): The file path to write the evaluation results.
        print (bool, optional): Whether to print the evaluation results. Defaults to False.

    Returns:
        tuple: A tuple containing the following evaluation metrics:
            - accurancy (float): The accuracy of the predictions.
            - rr (float): The rejection rate.
            - micro_tpr (float): The micro-average true positive rate.
            - micro_fpr (float): The micro-average false positive rate.
            - macro_tpr (float): The macro-average true positive rate.
            - macro_fpr (float): The macro-average false positive rate.
            - micro_distance_threedroc (float): The micro-average distance to the 3D ROC curve.
            - macro_distance_threedroc (float): The macro-average distance to the 3D ROC curve.
    """
    # Step 1: Calculate the metrics (including rejected instances)
    if print==True:
        cross_tab = pd.crosstab(data[value], data[value_pred])
        with open(file_path, 'a') as file:
            file.write(f"This matrix is the crosstab of the column {value} and {value_pred}.\n")
            file.write(tabulate(cross_tab, headers='keys', tablefmt='simple_grid'))
    
    rr = calculate_rejection_rate(data)    
    
    accurancy_rejection, coverage_rejection, prediction_quality, rejection_quality, combined_quality = calculate_performance_metrics_penalty_rejection(data)

    # Step 2: Data Preprocessing 
    ##  Remove rejected items
    data = data[data[value_pred] != "R"].copy()
    data[value_pred] = data[value_pred].astype(int)

    # Step 3: Calculate the metrics (ignoring rejected instances)
    unique_labels = np.union1d(np.unique(data[value]), np.unique(data[value_pred]))
    confusion_matrix_overall = confusion_matrix(data[value], data[value_pred])
    total_tp, total_fp, total_fn, total_tn, total_tpr, total_fpr, total_specificity, total_precision, unique_labels = calculate_sliced_confusion_matrix_metrics(unique_labels, confusion_matrix_overall)
    ## Calculate micro metrics
    accurancy, micro_tpr, micro_fpr, micro_specificity, micro_precision, micro_f1score = calculate_micro_metrics(total_tp, total_fp, total_fn, total_tn)
    micro_distance_threedroc = calculate_threedroc(micro_tpr, micro_fpr, rr)
    ## Calculate macro metrics
    macro_tpr, macro_fpr, macro_specificity, macro_precision, macro_f1score = calculate_macro_metrics(total_tpr, total_fpr, total_specificity, total_precision, unique_labels)
    macro_distance_threedroc = calculate_threedroc(macro_tpr, macro_fpr, rr)
    rmse = np.sqrt(mean_squared_error(data['ite'], data['ite_prob'])) # Calculate Root Mean Squared Error (RMSE)
    #rmse = np.sqrt(mean_squared_error(data['ite'], data['ite_reject'])) # Calculate Root Mean Squared Error (RMSE)
    ate_accuracy = np.abs(data['ite_pred'].mean() - data['ite'].mean()) # Evaluate ATE accuracy

    # Step 4: Generate Classification Report
    classificationreport = classification_report(data[value], data[value_pred], zero_division=np.nan)
    #classificationreport_df = pd.DataFrame.from_dict(classification_report(data[value], data[value_pred], output_dict=True, zero_division=np.nan))

    # if print == True:
    #     evaluator_print(file_path, classificationreport, rr, accurancy, micro_tpr, micro_fpr, macro_tpr, macro_fpr, micro_distance_threedroc, macro_distance_threedroc)
    
    metrics_dict = {
        'Accuracy': accurancy,
        'Rejection Rate': rr,
        'Micro TPR': micro_tpr,
        'Micro FPR': micro_fpr,
        'Macro TPR': macro_tpr,
        'Macro FPR': macro_fpr,
        'Micro Distance (3D ROC)': micro_distance_threedroc,
        'Macro Distance (3D ROC)': macro_distance_threedroc,
        'Accuracy with Rejection': accurancy_rejection,
        'Coverage with Rejection': coverage_rejection,
        'Prediction Quality': prediction_quality,
        'Rejection Quality': rejection_quality,
        'Combined Quality': combined_quality
    }

    return metrics_dict



def evaluator_print(file_path, classificationreport, rr, accurancy, micro_tpr, micro_fpr, macro_tpr, macro_fpr, micro_distance_threedroc, macro_distance_threedroc):
    with open(file_path, 'a') as file:
        file.write(f"\n\nPart 1 of the evaluation: Evaluate ITE models by considering only accepted instances, excluding rejected ones \n")
        file.write(classificationreport)
        file.write("\n")
        file.write(f"Accuracy: {accurancy:.4f}\n")
        file.write(f"Rejection Rate: {rr:.4f}\n")
        file.write(f"Micro TPR: {micro_tpr:.4f}\n")
        file.write(f"Micro FPR: {micro_fpr:.4f}\n")
        file.write(f"Macro TPR: {macro_tpr:.4f}\n")
        file.write(f"Macro FPR: {macro_fpr:.4f}\n")
        file.write(f"\n\nPart 2 of the evaluation: Assess ITE models by incorporating penalties for instances that are rejected.\n")
        file.write(f"Micro Distance (3D ROC): {micro_distance_threedroc:.4f}\n")
        file.write(f"Macro Distance (3D ROC): {macro_distance_threedroc:.4f}\n")
