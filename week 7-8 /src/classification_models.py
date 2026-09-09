"""
classification_models.py
Defines training functions for the 2 required classification models:
1. Logistic Regression
2. K-Nearest Neighbors (KNN with hyperparameter tuning over K = 3, 5, 7, 9, 11)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def train_logistic_regression(X_train, y_train, random_state: int = 42):
    """
    Trains a Logistic Regression classifier for binary Pass/Fail prediction.
    """
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_knn_classifier(X_train, y_train, k_values=(3, 5, 7, 9, 11)):
    """
    Trains a K-Nearest Neighbors classifier, tuning optimal K using 5-fold CV.
    """
    param_grid = {'n_neighbors': list(k_values)}
    grid_search = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Optimal K for KNN: {grid_search.best_params_['n_neighbors']} (Best CV F1: {grid_search.best_score_:.4f})")
    return best_model, grid_search.cv_results_


def train_all_classification_models(X_train, y_train):
    """
    Trains both classification models on preprocessed training data.
    Returns a dictionary of models and tuning details.
    """
    print("Training Logistic Regression...")
    log_reg = train_logistic_regression(X_train, y_train)

    print("Training K-Nearest Neighbors (tuning K in [3, 5, 7, 9, 11])...")
    knn, knn_results = train_knn_classifier(X_train, y_train)

    models = {
        'Logistic Regression': log_reg,
        'K-Nearest Neighbors': knn
    }
    return models, knn_results
