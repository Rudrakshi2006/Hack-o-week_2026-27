"""
train.py
Main training script:
1. Loads dataset
2. Preprocesses data and performs train/test split without data leakage
3. Trains 4 Regression Models (Linear, Polynomial, Ridge, Lasso)
4. Trains 2 Classification Models (Logistic Regression, KNN)
5. Evaluates test set performance & builds comparison tables
6. Saves all models, preprocessor, and metrics to models/ using joblib
"""

import os
import joblib
import pandas as pd
import numpy as np

from data_preprocessing import (
    load_dataset,
    prepare_regression_data,
    prepare_classification_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    PASS_THRESHOLD
)
from regression_models import train_all_regression_models
from classification_models import train_all_classification_models
from evaluation import compare_regression_models, compare_classification_models


def main():
    print("=" * 60)
    print("HACK-O-WEEK: STUDENT PERFORMANCE & SUCCESS PREDICTOR")
    print("=" * 60)

    # 1. Load Data
    data_path = 'data/student_data.csv'
    print(f"\n[1] Loading dataset from '{data_path}'...")
    df = load_dataset(data_path)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    os.makedirs('models', exist_ok=True)

    # 2. Regression Preparation & Training
    print("\n[2] Preparing Regression Data (Train: 80%, Test: 20%, random_state=42)...")
    reg_preprocessor, X_train_reg, X_test_reg, X_train_reg_trans, X_test_reg_trans, y_train_reg, y_test_reg = prepare_regression_data(df)

    print("\n[3] Training 4 Regression Models...")
    reg_models = train_all_regression_models(X_train_reg_trans, y_train_reg)

    print("\n[4] Evaluating Regression Models on Test Set...")
    reg_comparison_df, reg_predictions = compare_regression_models(reg_models, X_test_reg_trans, y_test_reg)
    print("\n--- REGRESSION MODEL COMPARISON ---")
    print(reg_comparison_df.to_string(index=False))

    # 3. Classification Preparation & Training
    print(f"\n[5] Preparing Classification Data (Threshold >= {PASS_THRESHOLD} -> PASS)...")
    cls_preprocessor, X_train_cls, X_test_cls, X_train_cls_trans, X_test_cls_trans, y_train_cls, y_test_cls = prepare_classification_data(df, threshold=PASS_THRESHOLD)

    print("\n[6] Training 2 Classification Models...")
    cls_models, knn_cv_results = train_all_classification_models(X_train_cls_trans, y_train_cls)

    print("\n[7] Evaluating Classification Models on Test Set...")
    cls_comparison_df, cls_details = compare_classification_models(cls_models, X_test_cls_trans, y_test_cls)
    print("\n--- CLASSIFICATION MODEL COMPARISON ---")
    print(cls_comparison_df.to_string(index=False))

    for name, det in cls_details.items():
        print(f"\nClassification Report for {name}:\n{det['Report']}")

    # 4. Save Models & Preprocessor
    print("\n[8] Serializing Models and Preprocessor into 'models/'...")
    joblib.dump(reg_preprocessor, 'models/preprocessor.pkl')
    joblib.dump(reg_models['Linear Regression'], 'models/linear_regression.pkl')
    joblib.dump(reg_models['Polynomial Regression'], 'models/polynomial_regression.pkl')
    joblib.dump(reg_models['Ridge Regression'], 'models/ridge_regression.pkl')
    joblib.dump(reg_models['Lasso Regression'], 'models/lasso_regression.pkl')
    joblib.dump(cls_models['Logistic Regression'], 'models/logistic_regression.pkl')
    joblib.dump(cls_models['K-Nearest Neighbors'], 'models/knn.pkl')

    # Save summary metrics and evaluation data for instant Streamlit rendering
    metrics_bundle = {
        'reg_comparison': reg_comparison_df,
        'cls_comparison': cls_comparison_df,
        'reg_predictions': reg_predictions,
        'cls_details': cls_details,
        'y_test_reg': y_test_reg.values,
        'y_test_cls': y_test_cls.values,
        'numerical_features': NUMERICAL_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'pass_threshold': PASS_THRESHOLD,
        'dataset_shape': df.shape,
        'ridge_best_alpha': getattr(reg_models['Ridge Regression'], 'alpha_', None),
        'lasso_best_alpha': getattr(reg_models['Lasso Regression'], 'alpha_', None),
        'knn_best_k': getattr(cls_models['K-Nearest Neighbors'], 'n_neighbors', None)
    }
    joblib.dump(metrics_bundle, 'models/model_metrics.pkl')

    print("\n All models and metrics saved successfully to 'models/'!")
    print("=" * 60)


if __name__ == '__main__':
    main()
