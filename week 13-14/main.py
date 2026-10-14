import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-Learn tools and models
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

# Boosting libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def main():
    # -------------------------------------------------------------------------
    # 1. Dataset Loading and Preparation
    # -------------------------------------------------------------------------
    print("=== 1. Dataset Loading ===")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    print(f"Dataset shape: {df.shape}")
    print("Class distribution:")
    print(df['target'].value_counts(normalize=True))
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature scaling (important for linear models and distance-based algorithms, 
    # though tree-based models don't strictly need it, it's good practice)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\nDataset split and scaled successfully.\n")


    # -------------------------------------------------------------------------
    # 2. Models to Implement & 3. Model Evaluation
    # -------------------------------------------------------------------------
    print("=== 2 & 3. Training Models and Evaluation ===")
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(random_state=42)
    }
    
    results = []
    best_model_name = ""
    best_f1 = -1
    best_model_obj = None
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})
        
        # Keep track of the best ensemble model for the confusion matrix later
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model
            
    # Create and display comparison DataFrame
    results_df = pd.DataFrame(results)
    print("\nModel Comparison DataFrame:")
    print(results_df.to_string(index=False))
    print("\n")
    
    # Plot model accuracy comparison
    plt.figure(figsize=(8, 5))
    plt.bar(results_df['Model'], results_df['Accuracy'], color=['skyblue', 'lightgreen', 'salmon', 'orchid'])
    plt.ylim(0.8, 1.0)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.show()


    # -------------------------------------------------------------------------
    # 5. Overfitting and Underfitting Demonstration
    # -------------------------------------------------------------------------
    print("=== 5. Overfitting and Underfitting Demonstration ===")
    
    # Underfitting model (Too simple)
    dt_under = DecisionTreeClassifier(max_depth=1, random_state=42)
    dt_under.fit(X_train_scaled, y_train)
    acc_train_under = accuracy_score(y_train, dt_under.predict(X_train_scaled))
    acc_test_under = accuracy_score(y_test, dt_under.predict(X_test_scaled))
    
    # Overfitting model (Too complex)
    dt_over = DecisionTreeClassifier(max_depth=None, random_state=42)
    dt_over.fit(X_train_scaled, y_train)
    acc_train_over = accuracy_score(y_train, dt_over.predict(X_train_scaled))
    acc_test_over = accuracy_score(y_test, dt_over.predict(X_test_scaled))
    
    print(f"Shallow Tree (max_depth=1): Train Acc = {acc_train_under:.4f}, Test Acc = {acc_test_under:.4f}")
    print(f"Deep Tree (max_depth=None): Train Acc = {acc_train_over:.4f}, Test Acc = {acc_test_over:.4f}")
    
    print("\nInterpretation:")
    if acc_train_under < 0.95 and acc_test_under < 0.95:
         print("- The shallow tree has relatively lower training and testing performance -> possible underfitting.")
    if acc_train_over == 1.0 and acc_test_over < acc_train_over:
         print("- The deep tree has perfect training performance but lower testing performance -> possible overfitting.")
    print("\n")


    # -------------------------------------------------------------------------
    # 4. Bias–Variance Trade-off Analysis
    # -------------------------------------------------------------------------
    print("=== 4. Bias-Variance Trade-off Analysis ===")
    depths = [1, 2, 3, 5, 8, 12, None]
    train_accs = []
    test_accs = []
    
    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X_train_scaled, y_train)
        
        train_accs.append(accuracy_score(y_train, dt.predict(X_train_scaled)))
        test_accs.append(accuracy_score(y_test, dt.predict(X_test_scaled)))
        
    # For plotting purposes, replace 'None' with a numeric value slightly larger than max numeric depth
    plot_depths = [d if d is not None else 15 for d in depths]
    
    plt.figure(figsize=(8, 5))
    plt.plot(plot_depths, train_accs, marker='o', label='Training Accuracy', color='blue')
    plt.plot(plot_depths, test_accs, marker='o', label='Testing Accuracy', color='red')
    plt.xticks(plot_depths, [str(d) for d in depths])
    plt.title("Tree Depth vs Training and Testing Accuracy")
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    print("Graph plotted. Explanation of Bias-Variance Trade-off:")
    print("- Left side (low depth): High bias, underfitting. Model is too simple to learn the data well.")
    print("- Middle: Good balance. Testing accuracy peaks, suggesting good generalization.")
    print("- Right side (high depth): High variance, overfitting. Training accuracy approaches 1.0, but testing accuracy drops or plateaus.")
    plt.show()


    # -------------------------------------------------------------------------
    # 6. Regularization (L1 vs L2)
    # -------------------------------------------------------------------------
    print("\n=== 6. Regularization (L1 vs L2) ===")
    # Using Logistic Regression
    
    # 1. No penalty (requires a solver that supports no penalty, 'lbfgs' is default, set penalty=None)
    lr_none = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=42)
    lr_none.fit(X_train_scaled, y_train)
    acc_none = accuracy_score(y_test, lr_none.predict(X_test_scaled))
    
    # 2. L1 penalty
    lr_l1 = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
    lr_l1.fit(X_train_scaled, y_train)
    acc_l1 = accuracy_score(y_test, lr_l1.predict(X_test_scaled))
    
    # 3. L2 penalty
    lr_l2 = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)
    lr_l2.fit(X_train_scaled, y_train)
    acc_l2 = accuracy_score(y_test, lr_l2.predict(X_test_scaled))
    
    print(f"Logistic Regression (No Penalty) Test Acc: {acc_none:.4f}")
    print(f"Logistic Regression (L1 Penalty) Test Acc: {acc_l1:.4f}")
    print(f"Logistic Regression (L2 Penalty) Test Acc: {acc_l2:.4f}")
    
    print("\nExplanation:")
    print("- L1 regularization (Lasso) can force some coefficients to become zero, acting as feature selection.")
    print("- L2 regularization (Ridge) discourages very large coefficients but rarely makes them exactly zero.")
    print("- Both techniques help reduce overfitting and can improve generalization compared to an unregularized model.")


    # -------------------------------------------------------------------------
    # 7. Confusion Matrix
    # -------------------------------------------------------------------------
    print(f"\n=== 7. Confusion Matrix (Best Model: {best_model_name}) ===")
    
    ConfusionMatrixDisplay.from_estimator(
        best_model_obj, 
        X_test_scaled, 
        y_test,
        display_labels=data.target_names,
        cmap=plt.cm.Blues
    )
    plt.title(f"Confusion Matrix: {best_model_name}")
    plt.show()

    
    # -------------------------------------------------------------------------
    # 10. Final Interpretation
    # -------------------------------------------------------------------------
    print("\n=== Final Interpretation ===")
    print(f"The best performing model on the test set is {best_model_name} with an F1 score of {best_f1:.4f}.")
    print("Ensemble models (Random Forest, XGBoost, LightGBM) generally outperformed the single Decision Tree,")
    print("demonstrating the power of bagging and boosting to reduce variance and bias.")

if __name__ == "__main__":
    main()
