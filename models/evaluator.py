# Define a function to apply categories based on the conditions
import pandas as pd
from tabulate import tabulate
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, recall_score
import math

def categorize(row):
    if row['y_t0'] == 0 and row['y_t1'] == 0:
        return 'Lost Cause'
    elif row['y_t0'] == 1 and row['y_t1'] == 0:
        return 'Sleeping Dog'
    elif row['y_t0'] == 0 and row['y_t1'] == 1:
        return 'Persuadable' # (can be rescued)
    elif row['y_t0'] == 1 and row['y_t1'] == 1:
        return 'Sure Thing'

def categorize_pred(row):
    if row['y_t0_pred'] == 0 and row['y_t1_pred'] == 0:
        return 'Lost Cause'
    elif row['y_t0_pred'] == 1 and row['y_t1_pred'] == 0:
        return 'Sleeping Dog'
    elif row['y_t0_pred'] == 0 and row['y_t1_pred'] == 1:
        return 'Persuadable' # (can be rescued)
    elif row['y_t0_pred'] == 1 and row['y_t1_pred'] == 1:
        return 'Sure Thing'

def instructions_matrix(file_path):
    with open(file_path, 'a') as file:
        file.write(f"We make the matrix: Lost Cause, Sleeping Dog, Persuadable, Sure Thing \n")
        file.write(f"Comment: \n")
        file.write(f" - Upper left cell: amount of cases that have outcome 0: no matter if you would treat or not \n")
        file.write(f"   If treat, they stay alive, if no treat they also stay alive. \n")

        file.write(f" - Under right cell: amount of cases that have outcome 1: no matter if you would treat or not \n")
        file.write(f"   If treat, they die, if no treat they also die. \n")

        file.write(f" - Upper right cell: amount of cases that have outcome 1 if treated, but outcome 0 if not treated \n")
        file.write(f"   If treat, they die, if no treat they stay alive. \n")

        file.write(f" - Under left cell: amount of cases that have outcome 0 if treated, but outcome 1 if not treated \n")
        file.write(f"   If treat, they stay alive, if no treat they die. \n\n")

def calculate_crosstab_matrix_names(t0, t1, data, file_path):
    count_matrix = pd.crosstab(data[t0], data[t1], margins=False)

    # Calculate and write crosstab to the file
    with open(file_path, 'a') as file:
        file.write(f"\nCrosstab for {t0} and {t1}:\n")
        file.write(tabulate(count_matrix, headers='keys', tablefmt='simple_grid'))
        file.write(f"\n")
    



def calculate_crosstab(value, value_pred, data, file_path):
    # Info on: https://www.v7labs.com/blog/confusion-matrix-guide
    length_before_rejection = len(data)

    data = data[data[value_pred] != 'R'].copy()
    length_after_rejection = len(data)
    data[value] = data[value].astype(int)
    data[value_pred] = data[value_pred].astype(int)

    rr = (length_before_rejection-length_after_rejection) / length_before_rejection
    
    # Get unique class labels
    unique_labels = np.unique(data[value])

    # Based on sklearn
    confusion_matrix_overall = confusion_matrix(data[value], data[value_pred])
    confusionmatrix = confusion_matrix(data[value], data[value_pred])
    
    # Calculate the overall confusion matrix
    confusion_matrix_overall = confusion_matrix(data[value], data[value_pred])

    # Get unique class labels
    unique_labels = np.unique(data[value])

    # Initialize a dictionary to store confusion matrices, TPR, and FPR for each class
    class_metrics = {}
    # Initialize variables to store aggregated metrics
    total_tpr, total_fpr, total_specificity, total_precision = 0, 0, 0, 0
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    # Compute metrics for each class
    for label in unique_labels:
        # Index of the class in the confusion matrix
        class_index = np.where(unique_labels == label)[0][0]
        
        # Rows and columns corresponding to the class
        true_positive = confusion_matrix_overall[class_index, class_index]
        false_positive = confusion_matrix_overall[:, class_index].sum() - true_positive
        false_negative = confusion_matrix_overall[class_index, :].sum() - true_positive
        true_negative = confusion_matrix_overall.sum() - (true_positive + false_positive + false_negative)
        
        # Create a confusion matrix for the class
        confusion_matrix_class = np.array([[true_negative, false_positive], [false_negative, true_positive]])
        
        # Calculate TPR and FPR for the class
        tpr_class = true_positive / (true_positive + false_negative)
        fpr_class = false_positive / (false_positive + true_negative)
        specificity_class = 1 - fpr_class
        precision_class = true_positive / (true_positive + false_positive)

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
        #micro: Micro specificity is calculated by considering the global count of true negatives and false positives across all classes. It gives equal weight to each instance, regardless of its class.
        #macro: Macro specificity is calculated by computing specificity for each class individually and then taking the average. It treats each class equally, regardless of the class size.

    # Calculate micro metrics
    micro_tpr = total_tp / (total_tp + total_fn) # TPR = Recall = Sensitivity
    micro_fpr = total_fp / (total_fp + total_tn)
    micro_specificity = 1 - (total_tn / (total_fp + total_tn))
    accurancy = (total_tp + total_tn) / (total_tp + total_tn + total_fn + total_fp)
    micro_precision = total_tp / (total_tp + total_fp)
    micro_f1score = 2* (micro_tpr * micro_precision) / (micro_tpr + micro_precision)

    # Calculate macro metrics
    macro_tpr = total_tpr / len(unique_labels) # TPR = Recall = Sensitivity
    macro_fpr = total_fpr / len(unique_labels)
    macro_specificity = total_specificity / len(unique_labels)
    macro_precision = total_precision / len(unique_labels)
    macro_f1score = 2* (macro_tpr * macro_precision) / (macro_tpr + macro_precision)

    # Check unique values in t0 and t1
    unique_values_t0 = np.unique(data[value])
    unique_values_t1 = np.unique(data[value_pred])

    headers = [f"Predicted {class_name}" for class_name in unique_values_t1]
    index = [f"Actual {class_name}" for class_name in unique_values_t0]

    confusionmatrix = pd.DataFrame(confusionmatrix, columns=headers, index=index)

    classificationreport = classification_report(data[value], data[value_pred])
    classificationreport_df = classification_report(data[value], data[value_pred], output_dict=True)
    # Convert the classification report dictionary to a DataFrame
    classificationreport_df = pd.DataFrame.from_dict(classificationreport_df)

    micro_distance_threedroc = threedroc(micro_tpr, micro_fpr, rr)
    macro_distance_threedroc = threedroc(macro_tpr, macro_fpr, rr)

    evaluator_print(file_path, classificationreport, rr, accurancy, micro_tpr, micro_fpr, macro_tpr, macro_fpr, micro_distance_threedroc, macro_distance_threedroc)

    return accurancy, rr, micro_tpr, micro_fpr, macro_tpr, macro_fpr, micro_distance_threedroc, macro_distance_threedroc

def evaluator_print(file_path, classificationreport, rr, accurancy, micro_tpr, micro_fpr, macro_tpr, macro_fpr, micro_distance_threedroc, macro_distance_threedroc):
    with open(file_path, 'a') as file:
        file.write(f"\n")
        file.write(classificationreport)
        file.write("\n")
        file.write(f"Accuracy: {accurancy:.4f}\n")
        file.write(f"Rejection Rate: {rr:.4f}\n")

        file.write(f"Micro TPR: {micro_tpr:.4f}\n")
        file.write(f"Micro FPR: {micro_fpr:.4f}\n")
        file.write(f"Macro TPR: {macro_tpr:.4f}\n")
        file.write(f"Macro FPR: {macro_fpr:.4f}\n")
        file.write(f"Micro Distance (3D ROC): {micro_distance_threedroc:.4f}\n")
        file.write(f"Macro Distance (3D ROC): {macro_distance_threedroc:.4f}\n")

def threedroc(tpr, fpr, rr):
    distance_threedroc = math.sqrt((0 - fpr)**2 + (1 - tpr)**2 + (0 - rr)**2)
    return distance_threedroc

"""
┌───────┬───────┬───────┬───────┐
│ ite   │   -1  │   0   │   1   │
├───────┼───────┼───────┼───────┤
│ -1    │   TP  │   ?   │   ?   │
├───────┼───────┼───────┼───────┤
│  0    │   ?   │   TN  │   ?   │
├───────┼───────┼───────┼───────┤
│  1    │   ?   │   ?   │   ?   │
└───────┴───────┴───────┴───────┘
"""