from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from tabulate import tabulate
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def evaluation_binary(treated_y, treated_y_pred, treated_y_prob, control_y, control_y_pred, control_y_prob):
    evaluation_binary_metrics(treated_y, treated_y_pred, control_y, control_y_pred)
    evaluation_binary_confusion_table(treated_y, treated_y_pred, control_y, control_y_pred)
    evaluation_binary_roc(treated_y, treated_y_prob, control_y, control_y_prob)

def evaluation_binary_metrics(treated_y_test, treated_y_pred, control_y_test, control_y_pred):
    digits= int(2)
    # Calculate evaluation metrics
    
    treated_auc = round(roc_auc_score(treated_y_test, treated_y_pred), digits)
    control_auc = round(roc_auc_score(control_y_test, control_y_pred), digits)

    treated_f1 = round(f1_score(treated_y_test, treated_y_pred), digits)
    control_f1 = round(f1_score(control_y_test, control_y_pred), digits)

    treated_precision = round(precision_score(treated_y_test, treated_y_pred), digits)
    control_precision = round(precision_score(control_y_test, control_y_pred), digits)

    treated_recall = round(recall_score(treated_y_test, treated_y_pred), digits)
    control_recall = round(recall_score(control_y_test, control_y_pred), digits)

    treated_accuracy = round(accuracy_score(treated_y_test, treated_y_pred), digits)
    control_accuracy = round(accuracy_score(control_y_test, control_y_pred), digits)

    # Metrics in a dictionary for easier iteration
    metrics = {
    "AUC": [treated_auc, control_auc],
    "F1": [treated_f1, control_f1],
    "Precision": [treated_precision, control_precision],
    "Recall": [treated_recall, control_recall],
    "Accuracy": [treated_accuracy, control_accuracy]
    }

    # Convert dictionary to table format
    table = []
    for metric, values in metrics.items():
        table.append([metric, values[0], values[1]])
    
    # Display the table
    print(tabulate(table, headers=["Metric", "Treated Group", "Control Group"], tablefmt="pretty"))

    return treated_auc, control_auc, treated_f1, control_f1, treated_precision, control_precision, treated_recall, control_recall, treated_accuracy, control_accuracy

def evaluation_binary_confusion_table(treated_y_test, treated_y_pred, control_y_test, control_y_pred):
    # Calculate confusion matrices for treated and control groups
    treated_confusion_matrix = confusion_matrix(treated_y_test, treated_y_pred)
    control_confusion_matrix = confusion_matrix(control_y_test, control_y_pred)

    # Extract TP, TN, FP, FN from the confusion matrices
    treated_TP, treated_FP, treated_FN, treated_TN = treated_confusion_matrix.ravel()
    control_TP, control_FP, control_FN, control_TN = control_confusion_matrix.ravel()

    # Prepare data for the table
    data = [
        ["True Positives", treated_TP, control_TP],
        ["False Positives", treated_FP, control_FP],
        ["False Negatives", treated_FN, control_FN],
        ["True Negatives", treated_TN, control_TN]
    ]
    
    # Display data in a table
    print(tabulate(data, headers=["Metric", "Treated Group", "Control Group"], tablefmt="pretty"))

def evaluation_binary_roc(treated_y_test, treated_y_prob, control_y_test, control_y_prob):
    # Calculate ROC curve for treated and control groups
    treated_fpr, treated_tpr, _ = roc_curve(treated_y_test, treated_y_prob)
    control_fpr, control_tpr, _ = roc_curve(control_y_test, control_y_prob)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(treated_fpr, treated_tpr, label='Treated group')
    plt.plot(control_fpr, control_tpr, label='Control group')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()