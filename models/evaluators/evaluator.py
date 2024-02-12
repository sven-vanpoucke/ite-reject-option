# Define a function to apply categories based on the conditions
import pandas as pd
from tabulate import tabulate
import numpy as np

from .performance_evaluator import calculate_performance_metrics
from .cost_evaluator import calculate_cost_metrics

def append_all_metrics(metrics_results, metrics_dict):
    for key, value in metrics_dict.items():
        if key not in metrics_results:
            metrics_results[key] = []  # Initialize the key if it doesn't exist
        metrics_results[key].append(round(value, 4))

    return metrics_results

def calculate_all_metrics(value, value_pred, data, file_path, metrics_results, append_metrics_results=True, print=False):
    #
    metrics_dict = calculate_performance_metrics(value, value_pred, data, file_path, print)
    for key, value in metrics_dict.items():
        if key not in metrics_results:
            metrics_results[key] = []  # Initialize the key if it doesn't exist
        metrics_results[key].append(round(value, 4))
    if print:
        pass

    # 
    metrics_dict = calculate_cost_metrics(value, value_pred, data, file_path, print)
    for key, value in metrics_dict.items():
        if key not in metrics_results:
            metrics_results[key] = []  # Initialize the key if it doesn't exist
        metrics_results[key].append(round(value, 4))
    if print:
        pass

    return metrics_results
