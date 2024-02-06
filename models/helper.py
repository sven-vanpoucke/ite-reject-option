from datetime import datetime
from models.evaluator import calculate_crosstab

def helper_output(folder_path='output/'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_print = datetime.now().strftime("%Y-%m-%d %H:%M")
    filename = f'results_{timestamp}.txt'
    file_path = folder_path + filename

    with open(file_path, 'a') as file:
        file.write(f"CHAPTER 1: INTRODUCTION\n\n")
        file.write(f"This file has been automatically generated on: {timestamp_print}\n\n")

        file.write(f"This file has been generated on: {timestamp_print}\n\n")

    return timestamp, filename, file_path


def improvement(old_value, new_value):
    old_value = float(old_value)
    new_value = float(new_value)

    improvement = ((new_value-old_value)/new_value*100)# .round(2)
    return improvement


def print_reject_improvements(file_path, rr_2, total_cost_ite, total_cost_ite_2, accurancy, accurancy_2, micro_distance_threedroc, micro_distance_threedroc_2, macro_distance_threedroc, macro_distance_threedroc_2):
    cost_improvementt = improvement(total_cost_ite, total_cost_ite_2)
    accurancy_improvement = improvement(accurancy, accurancy_2)
    micro_distance_improvement = improvement(micro_distance_threedroc, micro_distance_threedroc_2)
    macro_distance_improvement = improvement(macro_distance_threedroc, macro_distance_threedroc_2)

    with open(file_path, 'a') as file:
        # Write the total misclassification cost after probability rejection
        file.write(f"\nImprovements due to a rejection rate of {(rr_2*100):.2f}%:\n")

        # Write the total misclassification cost
        file.write(f'Change of the Misclassification Cost: {cost_improvementt:.2f}%\n')
        file.write(f'Change of the ITE Accurancy: {accurancy_improvement:.2f}%\n')
        file.write(f'Change of the 3D ROC (micro): {micro_distance_improvement:.2f}%\n')
        file.write(f'Change of the 3D ROC (macro): {macro_distance_improvement:.2f}%\n')

def print_rejection(file_path, test_set, total_cost_ite, accurancy, micro_distance_threedroc, macro_distance_threedroc):

    # Calculate total misclassification cost
    test_set['cost_ite_reject'] = test_set.apply(lambda row: 0 if row['ite_reject'] else row['cost_ite'], axis=1)
    total_cost_ite_2 = test_set['cost_ite_reject'].sum()

    accurancy_2, rr_2, micro_tpr_2, micro_fpr_2, macro_tpr_2, macro_fpr_2, micro_distance_threedroc_2, macro_distance_threedroc_2 = calculate_crosstab('ite', 'ite_reject', test_set, file_path, print = True)

    print_reject_improvements(file_path, rr_2, total_cost_ite, total_cost_ite_2, accurancy, accurancy_2, micro_distance_threedroc, micro_distance_threedroc_2, macro_distance_threedroc, macro_distance_threedroc_2)
