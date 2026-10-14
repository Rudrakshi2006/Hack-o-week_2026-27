# Ensemble Learning and Bias–Variance Analysis

## Objective
This mini-project demonstrates key concepts in machine learning, focusing on Ensemble Learning (Bagging and Boosting), Bias-Variance Trade-off, Overfitting/Underfitting, and Regularization techniques (L1/L2).

## Topics Covered
* **Decision Trees:** Baseline model to understand tree depth impact.
* **Bagging:** Random Forest classifier.
* **Boosting:** XGBoost and LightGBM classifiers.
* **Bias-Variance Trade-off:** Exploring model complexity vs. training/testing accuracy.
* **Underfitting & Overfitting:** Practical demonstration using tree depth.
* **Regularization:** Comparing L1 (Lasso) and L2 (Ridge) penalties in Logistic Regression.

## Dataset Description
The project uses the **Breast Cancer Wisconsin dataset**, available in `sklearn.datasets`. It contains 569 instances of tumor observations with 30 numeric, predictive features. The target variable is binary, classifying the tumor as malignant or benign.

## Models Used
1. Decision Tree (Base model)
2. Random Forest (Bagging)
3. XGBoost (Boosting)
4. LightGBM (Boosting)
5. Logistic Regression (for Regularization)

## Bias–Variance Explanation
The Bias-Variance Trade-off describes the balance between a model's ability to learn the training data and its ability to generalize to unseen data.
* **High Bias (Underfitting):** The model is too simple and fails to capture the underlying pattern (e.g., very shallow decision tree). Both training and testing performance are poor.
* **High Variance (Overfitting):** The model is too complex and memorizes the training data, including noise (e.g., an unconstrained decision tree). Training performance is excellent, but testing performance drops significantly.

## Overfitting/Underfitting Explanation
* **Underfitting:** The model lacks the capacity to represent the data's complexity.
* **Overfitting:** The model fits the training data too closely, failing to generalize to new data.

## L1/L2 Regularization Explanation
Regularization helps prevent overfitting by penalizing large coefficients in the model.
* **L1 Regularization (Lasso):** Can force some feature coefficients to exactly zero, performing feature selection.
* **L2 Regularization (Ridge):** Shrinks coefficients toward zero, but rarely exactly zero, distributing the importance across features.

## Installation Instructions

1. **Clone or Download** the project folder.
2. **Navigate** to the project directory:
   ```bash
   cd ensemble_learning_project
   ```
3. **Install dependencies** using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Project
Simply execute the `main.py` script:
```bash
python main.py
```

## Expected Output
* Textual output in the console describing data shapes and model performances (Accuracy, Precision, Recall, F1).
* A Pandas DataFrame comparing the four main classifiers.
* A text block explicitly showing underfitting and overfitting with Decision Trees.
* Results of Logistic Regression models with no penalty, L1, and L2 regularization.
* Several Matplotlib plots, which will appear one by one:
  1. A bar chart comparing model accuracies.
  2. A line plot demonstrating the Bias-Variance trade-off via Tree Depth vs. Accuracy.
  3. A confusion matrix for the best-performing ensemble model.

## Conclusion
This project demonstrates that ensemble models (Random Forest, XGBoost, LightGBM) typically outperform single base estimators (Decision Trees) by reducing variance or bias. It also illustrates how tuning hyperparameters (like tree depth) or using regularization (L1/L2) helps manage the bias-variance trade-off to achieve better generalization on unseen data.
