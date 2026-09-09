"""
regression_models.py
Defines training functions for the 4 required regression models:
1. Linear Regression
2. Polynomial Regression (Degree 2)
3. Ridge Regression (L2 Regularization with CV)
4. Lasso Regression (L1 Regularization with CV)
"""

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def train_linear_regression(X_train, y_train):
    """
    Trains an Ordinary Least Squares Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_polynomial_regression(X_train, y_train, degree: int = 2):
    """
    Trains a Polynomial Regression model (PolynomialFeatures + LinearRegression).
    """
    model = Pipeline([
        ('poly_features', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear_reg', LinearRegression())
    ])
    model.fit(X_train, y_train)
    return model


def train_ridge_regression(X_train, y_train, alphas=(0.01, 0.1, 1.0, 10.0, 100.0)):
    """
    Trains a Ridge Regression model (L2 regularization) selecting best alpha via 5-fold CV.
    """
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_train, y_train)
    return model


def train_lasso_regression(X_train, y_train, alphas=(0.001, 0.01, 0.1, 1.0, 10.0)):
    """
    Trains a Lasso Regression model (L1 regularization) selecting best alpha via 5-fold CV.
    """
    model = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=2000)
    model.fit(X_train, y_train)
    return model


def train_all_regression_models(X_train, y_train):
    """
    Trains all 4 regression models on preprocessed training data.
    Returns a dictionary of models and their metadata.
    """
    print("Training Linear Regression...")
    lin_reg = train_linear_regression(X_train, y_train)

    print("Training Polynomial Regression (degree=2)...")
    poly_reg = train_polynomial_regression(X_train, y_train, degree=2)

    print("Training Ridge Regression (tuning alpha)...")
    ridge_reg = train_ridge_regression(X_train, y_train)

    print("Training Lasso Regression (tuning alpha)...")
    lasso_reg = train_lasso_regression(X_train, y_train)

    models = {
        'Linear Regression': lin_reg,
        'Polynomial Regression': poly_reg,
        'Ridge Regression': ridge_reg,
        'Lasso Regression': lasso_reg
    }
    return models
