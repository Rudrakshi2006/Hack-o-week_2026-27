"""
data_preprocessing.py
Handles data loading, cleaning, feature transformation, and train-test splitting
without data leakage.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define Numerical and Categorical feature columns
NUMERICAL_FEATURES = [
    'Hours_Studied',
    'Attendance',
    'Sleep_Hours',
    'Previous_Scores',
    'Tutoring_Sessions',
    'Physical_Activity'
]

CATEGORICAL_FEATURES = [
    'Parental_Involvement',
    'Access_to_Resources',
    'Extracurricular_Activities',
    'Motivation_Level',
    'Internet_Access',
    'Family_Income',
    'Teacher_Quality',
    'School_Type',
    'Peer_Influence',
    'Learning_Disabilities',
    'Parental_Education_Level',
    'Distance_from_Home',
    'Gender'
]

TARGET_COLUMN = 'Exam_Score'
PASS_THRESHOLD = 65


def load_dataset(file_path: str = 'data/student_data.csv') -> pd.DataFrame:
    """Load student dataset from CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at path: {file_path}")
    df = pd.read_csv(file_path)
    return df


def create_preprocessor():
    """
    Creates a scikit-learn ColumnTransformer for preprocessing.
    - Numerical: Median imputation + StandardScaler
    - Categorical: Most frequent imputation + OneHotEncoder
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_FEATURES),
            ('cat', cat_pipeline, CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor


def prepare_regression_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Prepares train and test splits for Regression.
    Fits the preprocessor strictly on training data to prevent data leakage.
    """
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = create_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    return preprocessor, X_train, X_test, X_train_transformed, X_test_transformed, y_train, y_test


def prepare_classification_data(df: pd.DataFrame, threshold: int = PASS_THRESHOLD, test_size: float = 0.2, random_state: int = 42):
    """
    Prepares train and test splits for Classification with Pass/Fail target.
    Uses stratification to maintain class balance.
    """
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    # 1 for PASS, 0 for FAIL
    y_class = (df[TARGET_COLUMN] >= threshold).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=test_size, random_state=random_state, stratify=y_class
    )

    preprocessor = create_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    return preprocessor, X_train, X_test, X_train_transformed, X_test_transformed, y_train, y_test
