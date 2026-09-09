"""
test_pipeline.py
Unit tests for data preprocessing, regression models, classification models,
and artifact loading.
"""

import unittest
import os
import joblib
import pandas as pd
import numpy as np

from src.data_preprocessing import load_dataset, prepare_regression_data, prepare_classification_data


class TestMLPipeline(unittest.TestCase):

    def setUp(self):
        self.data_path = 'data/student_data.csv'
        self.df = load_dataset(self.data_path)

    def test_dataset_loaded(self):
        self.assertFalse(self.df.empty)
        self.assertEqual(self.df.shape[0], 6607)
        self.assertIn('Exam_Score', self.df.columns)

    def test_regression_preparation(self):
        preprocessor, X_train, X_test, X_train_t, X_test_t, y_train, y_test = prepare_regression_data(self.df)
        self.assertEqual(len(X_train), 5285)
        self.assertEqual(len(X_test), 1322)
        self.assertEqual(X_train_t.shape[0], 5285)
        self.assertEqual(X_test_t.shape[0], 1322)

    def test_classification_preparation(self):
        preprocessor, X_train, X_test, X_train_t, X_test_t, y_train, y_test = prepare_classification_data(self.df, threshold=65)
        self.assertEqual(set(np.unique(y_train)), {0, 1})
        self.assertEqual(set(np.unique(y_test)), {0, 1})

    def test_saved_artifacts_exist_and_predict(self):
        models = [
            'preprocessor.pkl',
            'linear_regression.pkl',
            'polynomial_regression.pkl',
            'ridge_regression.pkl',
            'lasso_regression.pkl',
            'logistic_regression.pkl',
            'knn.pkl',
            'model_metrics.pkl'
        ]
        for m in models:
            path = os.path.join('models', m)
            self.assertTrue(os.path.exists(path), f"Missing artifact: {path}")

        preprocessor = joblib.load('models/preprocessor.pkl')
        ridge = joblib.load('models/ridge_regression.pkl')
        log_reg = joblib.load('models/logistic_regression.pkl')

        sample = self.df.iloc[:2].drop(columns=['Exam_Score'])
        sample_trans = preprocessor.transform(sample)

        scores = ridge.predict(sample_trans)
        classes = log_reg.predict(sample_trans)

        self.assertEqual(len(scores), 2)
        self.assertEqual(len(classes), 2)
        self.assertTrue(all(0 <= s <= 100 for s in scores))


if __name__ == '__main__':
    unittest.main()
