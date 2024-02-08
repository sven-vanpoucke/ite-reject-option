import pandas as pd

def learning_to_reject_confusion_matrix(test_set, file_path):
    # Add column: prediction correct? T/F
    test_set['ite_correct'] = test_set.apply(lambda row: True if row['ite'] == row['ite_pred'] else False, axis=1)

    # Prediction correct: yes-no
    # Rejection: yes-no
    cross_tab = pd.crosstab(test_set['ite_correct'], test_set['ite_reject'])
    
    # ==> True accept, false accept, true reject, false reject    
    TA = cross_tab.loc[True, False]  # Prediction is correct, and the item is not rejected = True accepted
    FA = cross_tab.loc[True, True]   # Prediction is correct, but the item is rejected
    TR = cross_tab.loc[False, True]   # Prediction is incorrect, and the item is rejected
    FR = cross_tab.loc[False, False] # Prediction is incorrect, and the item is not rejected

    accurancy = TA / (TA + FA)
    coverage = (TA+FA) / (TA+FA+FR+TR)
    
    #Evaluating models with a fixed rejection rate
    prediction_quality = TA / (TA + FA)
    rejection_quality = (TR/FR) / ((FA+TR)/(TA+FR))
    combined_quality = (TA+TR) / (TA+FA+FR+TR)
    #Evaluating the model performance/rejection trade-off
    #==> accurancy-reject curve
    #difficult to have graph

    #Evaluating models through a cost function
    # Only for classification tasks
    cost_correct = 0
    cost_wrong = 2
    # Cr < 1/K 
    cost_rejection = 2
    # Cc < Cr < Ce

    # Display confusion matrix
    with open(file_path, 'a') as file:
        file.write(f"\n ")
    
    return combined_quality