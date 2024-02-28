import data_loader as dl
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_squared_error
from propensity_score_matching import propensity_score_matching_function
from evaluation_metrics import determine_thresholds_different_functions, confusion_matrix_custom, costs_calculation, metrics_rejected

def run_models(df, df_data):
    t_true = df["treatment"].astype(int)
    return evaluate_performance(LogisticRegression(solver='newton-cg'), df_data, t_true, nr_folds=25)

def evaluate_crossvalidated_metrics(algorithm, df, y_true, nr_folds=10) -> dict:
    scoring = ['accuracy', 'f1_micro', 'f1_weighted', 'f1_macro', 'recall_macro']
    results = cross_validate(algorithm, df, y_true, cv=nr_folds, scoring=scoring, return_train_score=False)
    return results

def evaluate_performance(algorithm, df, t_true, nr_folds=10):
    results_dict = evaluate_crossvalidated_metrics(algorithm, df, t_true, nr_folds)
    t_propens = cross_val_predict(algorithm, df, t_true, cv=nr_folds, method='predict_proba')
    t_propens = t_propens[:,0]
    
    results_dict['t_propens'] = t_propens
    results_dict['t_true'] = t_true

    # optimize the thresholds
    dict_thresholds = determine_thresholds_different_functions(df, t_true, t_propens)
    
    for method in dict_thresholds:
        t_pred = np.where(t_propens > dict_thresholds[method], 0, 1)
        conf_matrix = confusion_matrix(t_true, t_pred)
        results_dict['conf_matrix_' + method] = conf_matrix
        results_dict['t_pred_' + method] = t_pred
        results_dict['threshold_' + method] = dict_thresholds[method]
    

    return results_dict


from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_isolation_forest_with_grid_search(df_data, param_grid, t_true, cv , metric):

    """
    Evaluate an Isolation Forest model with grid search on synthetic data.

    Parameters:
    - data: Input data with outliers
    - param_grid: Parameter grid for grid search
    - cv: Number of cross-validation folds (default is 5)
    Returns:
    - Best model and evaluation metrics: precision, recall, F1-score, and ROC AUC
    """

    # Create Isolation Forest model
    isolation_forest = IsolationForest(random_state=42)

    # Perform grid search
    grid_search = GridSearchCV(isolation_forest, param_grid, scoring=metric, cv=cv)
    grid_search.fit(df_data, t_true)

    # Get the best Isolation Forest model
    best_isolation_forest = grid_search.best_estimator_

    # Predict the anomaly score for each observation
    anomaly_scores = best_isolation_forest.decision_function(df_data)

    # Predict whether each observation is an outlier
    outlier_predictions = best_isolation_forest.predict(df_data)
    print(outlier_predictions)
    return outlier_predictions

def ITE_predict_column(df):
    #this function needs to be executed after the prediction
    ITE_predicted=[]
    for index in df.index:
        if df.loc[index, 'treatment'] == df.loc[index,'t_predicted']:
            ITE_predicted.append(df.loc[index,'y_factual']-df.loc[index,'y_cfactual'])
        else:
            # When treatment is given, y_factual is observed, whereas y_cfactual is an estimation of the other effect.
            # Therefore, when we don't predict the same, these values must be switched in the estimation of the predicted ITE.
            ITE_predicted.append(df.loc[index,'y_cfactual']-df.loc[index,'y_factual'])
    ITE_predicted_Series = pd.Series(ITE_predicted, name='ITE_predicted')
    df['ITE_predicted'] = ITE_predicted_Series
    return df

def ITE_prediction_column_with_rejection(df):
    ITE_pred_rej= []
    for index in df.index:
        if df.loc[index,'isoutlier']==True:
            ITE_pred_rej.append(0)
        else:
            ITE_pred_rej.append(df.loc[index,'ITE_predicted'])
    ITE_pred_rej_Series = pd.Series(ITE_pred_rej,name = 'ITE_pred_rej')
    df['ITE_pred_rej'] = ITE_pred_rej_Series
    return df


def train_model(train_x, model_class=IsolationForest, **model_options):
    # Train the model
    model = model_class(**model_options)
    model.fit(train_x)
    return model

def prediction(df, df_data):
    results = run_models(df, df_data)
    print(df.columns)

    t_propens, t_true = results['t_propens'], results['t_true']
    df['propensity_score'] =t_propens
    thresholds= determine_thresholds_different_functions(df, t_true, t_propens)
    print(thresholds)

    #method could be diff metrics, I chose F1 since it is a combination of the precision and recall
    method='f1_score'
    print(thresholds[method])
    t_pred = np.where(t_propens > thresholds[method], 0, 1)
    print(t_pred)

    #add column to df, which treatment was predicted
    df['t_predicted'] =t_pred
    df = ITE_predict_column(df)
    print(df.columns)
    return df, df_data, t_true, t_pred

from evaluation_metrics import metrics_df

def contamination_value_PSM(df, df_data):
    detail_factor = 1  # 1 (no extra detail) or 10 (extra detail)
    results_list = []  # Create an empty DataFrame to store results


    for contamination in range(int(1 * detail_factor), int(499 * detail_factor) + 1):
        contamination /= (1000 * detail_factor)  # max of 0.5
        model = train_model(df_data, IsolationForest, contamination=contamination, random_state=42)
        is_outlier = model.predict(df_data)  # Corrected from using train_x
        is_outlier_df = pd.DataFrame(is_outlier, columns=['isoutlier(IF)'])
        df['isoutlier'] = (is_outlier_df == -1)
        
        #added
        #should be: you do the prediction for the subset that is accepted
        reject_T1 = (df['treatment'] == 1) & (df['isoutlier'] == True)
        reject_treatm_pos= reject_T1.sum()  # Count the number of True values (combination)

        reject_T0 =(df['treatment'] == 0 & (df['isoutlier'] == True))
        reject_treatm_neg = reject_T0.sum()

        #then do the prediction for the accepted subset
        accepted_samples = df[df['isoutlier'] == False].reset_index(drop=True)
        accepted_samples_data = accepted_samples.drop(columns=["y_factual", "y_cfactual", "ITE", "treatment","mu0","mu1"])

        min_treatment_count = 3                                                                     
        matched_pairs, accepted_samples = propensity_score_matching_function(accepted_samples,accepted_samples_data, min_treatment_count,treatment='treatment',k_neighbors=10)
        #make the next part a function
        t_true = matched_pairs['treatment']
        t_pred = matched_pairs['t_predicted']
        from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

        # Metrics
        accuracy = accuracy_score(t_true, t_pred)
        precision = precision_score(t_true, t_pred)
        recall = recall_score(t_true, t_pred)
        f1 = f1_score(t_true, t_pred)
        conf_matrix = confusion_matrix(t_true,t_pred)
        TP = conf_matrix[1, 1]
        TN= conf_matrix[0, 0]
        FP = conf_matrix[0, 1]
        FN = conf_matrix[1, 0]
        costs = costs_calculation(TP,TN,FP,FN,reject_treatm_neg,reject_treatm_pos,cost_TP=0,cost_TN=0,cost_FN=25,cost_FP=15, cost_reject=5)
        

        #metrics using ITE_pred
        accepted_samples = ITE_predict_column(accepted_samples)

        rmse, roc_auc,ATE_true, ATE_pred = metrics_df(accepted_samples)

        result_dict = {
            'contamination value': contamination,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cost':costs,
            'confusion matrix': conf_matrix,
            'rmse_accepted': rmse,
            'roc-auc_accepted': roc_auc,
            'ATE_pred': ATE_pred
        }

        results_list.append(result_dict)

    results_df = pd.DataFrame(results_list)
    return results_df    











def contamination_value(df_data, df):
    detail_factor = 1  # 1 (no extra detail) or 10 (extra detail)
    results_list = []  # Create an empty DataFrame to store results

    for contamination in range(int(1 * detail_factor), int(499 * detail_factor) + 1):
        contamination /= (1000 * detail_factor)  # max of 0.5
        model = train_model(df_data, IsolationForest, contamination=contamination, random_state=42)
        is_outlier = model.predict(df_data)  # Corrected from using train_x
        is_outlier_df = pd.DataFrame(is_outlier, columns=['isoutlier(IF)'])
        df['isoutlier'] = (is_outlier_df == -1)

        #added
        #should be: you do the prediction for the subset that is accepted
        reject_treatm_pos = [(df['treatment'] == 1) & (df['outlier'] == True)].sum()
        reject_treatm_neg = df[(df['treatment'] == 0) & (df['outlier'] == True)].sum()
        #then do the prediction for the accepted subset
        accepted_samples = df[df['isoutlier'] == False]
        accepted_samples_data = accepted_samples.drop(columns=["y_factual", "y_cfactual", "ITE", "treatment"])
        accepted_samples,accepted_samples_data,t_true,t_pred = prediction(df = accepted_samples, df_data = accepted_samples_data)

        '''
        results = run_models(df, df_data)
        print(df.columns)

        t_propens, t_true = results['t_propens'], results['t_true']
        df['propensity_score'] =t_propens
        thresholds= determine_thresholds_different_functions(df, t_true, t_propens)
        print(thresholds)

        #method could be diff metrics, I chose F1 since it is a combination of the precision and recall
        method='f1_score'
        print(thresholds[method])
        t_pred = np.where(t_propens > thresholds[method], 0, 1)
        print(t_pred)

        #add column to df, which treatment was predicted
        df['t_predicted'] =t_pred

        df = ITE_predict_column(df)
        print(df.columns)

        '''
        conf_matrix = confusion_matrix(t_true, t_pred)
        '''
        gives a matrix of the form: 
        [[True Negative  False Positive]
        [False Negative True Positive]]
        '''
        TP = conf_matrix[1, 1]
        TN= conf_matrix[0, 0]
        FP = conf_matrix[0, 1]
        FN = conf_matrix[1, 0]

        '''
        TP, TN, FP, FN, reject_treatm_neg, reject_treatm_pos, conf_matrix = confusion_matrix_custom(df)
        '''
        accuracy,precision,recall,f1= metrics_rejected(TP,TN,FP,FN,reject_treatm_neg,reject_treatm_pos)
        
        costs = costs_calculation(TP, TN, FP, FN, reject_treatm_neg, reject_treatm_pos, cost_TP=0, cost_TN=-5, cost_FN=40, cost_FP=10, cost_reject=5)
        #get the ITE with rejection
        df =  ITE_prediction_column_with_rejection(df)
        accepted_samples = df[df['isoutlier'] == 0]  # Assuming 'isoutlier' indicates rejection
        # Create a dictionary for the results of each contamination value
        result_dict = {
            'Contamination': contamination,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1':f1,
            'Cost': costs,
            'TP' : TP,
            'TN': TN,
            'FP':FP,
            'FN':FN,
            'reject_rate': (reject_treatm_neg+reject_treatm_pos)/(TP+TN+FP+FN+reject_treatm_neg+reject_treatm_pos),
            'rmse': np.sqrt(mean_squared_error(accepted_samples['ITE'], accepted_samples['ITE_predicted'])),
            'roc-auc':roc_auc_score(accepted_samples['treatment'], accepted_samples['t_predicted']),
            #RMSE takes the magnitude of the error into account
            'Confusion_matrix': conf_matrix
        }
        

        # Append the result dictionary as a row to the DataFrame
        results_list.append(result_dict)
        
    
    results_df = pd.DataFrame(results_list)
    return results_df



    