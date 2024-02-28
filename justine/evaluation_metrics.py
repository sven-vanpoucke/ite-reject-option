from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, average_precision_score, balanced_accuracy_score, precision_score, recall_score, jaccard_score,mean_squared_error
import pandas as pd
import numpy as np

def determine_thresholds_different_functions(df: pd.DataFrame, t_true: pd.Series, propensity_scores: pd.Series):
    dict_threshold_results = {}
    functions = [f1_score, accuracy_score, roc_auc_score, average_precision_score, balanced_accuracy_score, precision_score, recall_score, jaccard_score]
    
    for score_function in functions:
        optimal_threshold = determine_optimal_threshold(df, t_true, propensity_scores, score_function)
        dict_threshold_results[score_function.__name__] = optimal_threshold
    return dict_threshold_results

def determine_optimal_threshold(df: pd.DataFrame, t_true: pd.Series, propensity_scores: pd.Series, score_function) -> tuple:
    """
    Threshold optimization for the propensity model

    returns
    :param best_threshold:
    """
    threshold_list = np.arange(0, 1, 0.025)
    max_score, best_threshold = 0, 0

    for i in threshold_list:
        # Apply the threshold to predict binary values
        current_predicted_train_treatments = np.where(propensity_scores > i, 0, 1)
        # Now 'current_predicted_train_treatments' contains binary predictions based on the threshold
        score = score_function(t_true, current_predicted_train_treatments)
        if score >= max_score:
            best_threshold = i
            max_score = score
    return best_threshold


def confusion_matrix_custom(df):
    conf_matrix = np.zeros((2, 3))
    #TN
    conf_matrix[0, 0] = np.sum((df['treatment'] == 0) & (df['t_predicted'] == 0) & (df['isoutlier'] == False))
    TN=np.sum((df['treatment'] == 0) & (df['t_predicted'] == 0) & (df['isoutlier'] == False))

    #TP
    conf_matrix[1, 2] = np.sum((df['treatment'] == 1) & (df['t_predicted'] == 1) & (df['isoutlier'] == False))
    TP= np.sum((df['treatment'] == 1) & (df['t_predicted'] == 1) & (df['isoutlier'] == False))

    #FN
    conf_matrix[1, 0] = np.sum((df['treatment'] == 1) & (df['t_predicted'] == 0) & (df['isoutlier'] == False))
    FN= np.sum((df['treatment'] == 1) & (df['t_predicted'] == 0) & (df['isoutlier'] == False))

    #FP
    conf_matrix[0, 2] = np.sum((df['treatment'] == 0) & (df['t_predicted'] == 1) & (df['isoutlier'] == False))
    FP=np.sum((df['treatment'] == 0) & (df['t_predicted'] == 1) & (df['isoutlier'] == False))

    #reject, false
    conf_matrix[0, 1] = np.sum((df['treatment'] == 0)  & (df['isoutlier'] == True))
    reject_treatm_neg=np.sum((df['treatment'] == 0)  & (df['isoutlier'] == True))

    #reject, true = positive
    conf_matrix[1, 1] = np.sum((df['treatment'] == 1)  & (df['isoutlier'] == True))
    reject_treatm_pos=np.sum((df['treatment'] == 1)  & (df['isoutlier'] == True))
    return TP,TN,FP,FN,reject_treatm_neg,reject_treatm_pos,conf_matrix


#costs calculation
def costs_calculation(TP=0,TN=0,FP=0,FN=0,reject_treatm_neg=0,reject_treatm_pos=0,cost_TP=0,cost_TN=0,cost_FN=20,cost_FP=10, cost_reject=5):
    costs= TP*cost_TP+FN*cost_FN +TN*cost_TN+ FP*cost_FP+(reject_treatm_neg+reject_treatm_pos)*cost_reject
    return costs

def metrics_rejected(TP,TN,FP,FN,reject_treatm_neg=0,reject_treatm_pos=0):
    accuracy=(TP+TN)/(TP+TN+FN+FP)
    precision= TP/(TP+FP)
    recall=TP/(TP+FN)
    f1= (2*precision*recall)/(precision+recall)
    #combined quality
    return accuracy,precision,recall,f1

def metrics_df(df):
    #you need the columns 'isoutlier', but you dont have it for baseline model:
    #RMSE
    #ATE
    #ROC-AUC
    accepted_samples = df[df['isoutlier'] == False]
    rmse= np.sqrt(mean_squared_error(accepted_samples['ITE'], accepted_samples['ITE_predicted']))
    roc_auc =roc_auc_score(accepted_samples['treatment'], accepted_samples['t_predicted'])
    ATE_true = accepted_samples['y_factual'].mean() -accepted_samples['y_cfactual'].mean()
    ATE_pred= accepted_samples['ITE_predicted'].mean()
    
    return rmse, roc_auc,ATE_true, ATE_pred