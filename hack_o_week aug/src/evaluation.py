"""
evaluation.py
Calculates performance metrics, comparison dataframes, and reports
for both Regression and Classification models.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_regression_model(model, X_test, y_test):
    """
    Computes MAE, MSE, RMSE, and R2 for a regression model on test data.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'R2': round(r2, 4),
        'y_pred': y_pred
    }


def compare_regression_models(models_dict, X_test, y_test):
    """
    Evaluates all regression models and builds a comparison DataFrame.
    """
    records = []
    predictions_dict = {}

    for name, model in models_dict.items():
        metrics = evaluate_regression_model(model, X_test, y_test)
        predictions_dict[name] = metrics['y_pred']
        records.append({
            'Model': name,
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'RMSE': metrics['RMSE'],
            'R2': metrics['R2']
        })

    comparison_df = pd.DataFrame(records)
    # Sort by R2 descending (higher is better)
    comparison_df = comparison_df.sort_values(by='R2', ascending=False).reset_index(drop=True)
    return comparison_df, predictions_dict


def evaluate_classification_model(model, X_test, y_test):
    """
    Computes Accuracy, Precision, Recall, F1, and Confusion Matrix for a classification model.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['FAIL', 'PASS'])

    # Probabilities if supported
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'Confusion_Matrix': cm,
        'Report': report,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def compare_classification_models(models_dict, X_test, y_test):
    """
    Evaluates all classification models and builds a comparison DataFrame.
    """
    records = []
    details_dict = {}

    for name, model in models_dict.items():
        metrics = evaluate_classification_model(model, X_test, y_test)
        details_dict[name] = metrics
        records.append({
            'Model': name,
            'Accuracy': metrics['Accuracy'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1-Score': metrics['F1-Score']
        })

    comparison_df = pd.DataFrame(records)
    comparison_df = comparison_df.sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
    return comparison_df, details_dict
