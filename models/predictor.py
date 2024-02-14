from sklearn.linear_model import LogisticRegression
import pandas as pd
from tabulate import tabulate

# T-model
def predictor_t_model(train_treated_x, train_treated_y, train_control_x, train_control_y, model_class, model_params):
    # Training separate models for treated and control groups
    # treated_model = LogisticRegression(max_iter=10000, solver='saga', random_state=42)
    treated_model = model_class(**model_params)
    treated_model.fit(train_treated_x, train_treated_y.values.flatten())

    control_model = model_class(**model_params)
    control_model.fit(train_control_x, train_control_y.values.flatten())
    
    return treated_model, control_model

# Using RandomForestClassifier
#treated_model_rf, control_model_rf = predictor_t_model(model_class=RandomForestClassifier, n_estimators=100)


def predictor_train_predictions(treated_model, control_model, train_treated_x, train_control_x):
    # Training predictions for treated & control group
    train_treated_y_pred = pd.Series(treated_model.predict(train_treated_x), name='train_treated_y_pred')
    train_control_y_pred = pd.Series(control_model.predict(train_control_x), name='train_control_y_pred')
    
    # Getting probabilities to certain class
    train_treated_y_prob = None
    train_control_y_prob = None
    if hasattr(treated_model, 'predict_proba'):
        train_treated_y_prob = pd.Series(treated_model.predict_proba(train_treated_x)[:, 1], name='train_treated_y_prob')
        train_control_y_prob = pd.Series(control_model.predict_proba(train_control_x)[:, 1], name='train_control_y_prob')
    
    return train_treated_y_pred, train_treated_y_prob, train_control_y_pred, train_control_y_prob

def predictor_test_predictions(treated_model, control_model, test_treated_x, test_control_x):
    # Testing predictions for treated & control group
    test_treated_y_pred = pd.Series(treated_model.predict(test_treated_x), name='test_treated_y_pred')
    test_control_y_pred = pd.Series(control_model.predict(test_control_x), name='test_control_y_pred')

    # Getting probabilities to certain class
    test_treated_y_prob = None
    test_control_y_prob = None
    if hasattr(treated_model, 'predict_proba'):
        test_treated_y_prob = pd.Series(treated_model.predict_proba(test_treated_x)[:, 1], name='test_treated_y_prob')
        test_control_y_prob = pd.Series(control_model.predict_proba(test_control_x)[:, 1], name='test_control_y_prob')

    return test_treated_y_pred, test_treated_y_prob, test_control_y_pred, test_control_y_prob

def predictor_ite_predictions(treated_model, control_model, test_x):
    test_y_t1_pred = pd.Series(treated_model.predict(test_x), name='y_t1_pred')
    test_y_t0_pred = pd.Series(control_model.predict(test_x), name='y_t0_pred')
    
    #Get probabilities
    test_y_t1_prob = None
    test_y_t0_prob = None
    test_ite_prob = None
    if hasattr(treated_model, 'predict_proba'):
        test_y_t1_prob = pd.Series(treated_model.predict_proba(test_x)[:, 1], name='y_t1_prob')
        test_y_t0_prob = pd.Series(control_model.predict_proba(test_x)[:, 1], name='y_t0_prob')
        test_ite_prob = pd.Series(test_y_t1_prob-test_y_t0_prob, name='ite_prob')
        # Round the probabilities to 4 digits
        test_y_t1_prob = test_y_t1_prob.round(4)
        test_y_t0_prob = test_y_t0_prob.round(4)
        test_ite_prob = test_ite_prob.round(4)

    # Create binary ITE predictions based on a cutoff of 0.5. IS 0 if under 0.5, is 1 if above 0.5
    #test_ite_bin = (test_ite_prob > 0.5).astype(int)
    #test_ite_pred = pd.Series(test_ite_bin, name='ite_bin')

    # Create binary ITE predictions based on proximity to -1, 0, or 1
    #test_ite_pred = (test_ite_prob > 0.5).astype(int) - (test_ite_prob < -0.5).astype(int)
    test_ite_pred = test_y_t1_pred - test_y_t0_pred
    test_ite_pred = pd.Series(test_ite_pred, name='ite_pred')

    
    return test_y_t1_pred, test_y_t0_pred, test_y_t1_prob, test_y_t0_prob, test_ite_prob, test_ite_pred
