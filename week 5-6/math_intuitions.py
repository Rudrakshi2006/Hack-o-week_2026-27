import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("==========================================================")
    print(" MATH INTUITIONS ON IRIS DATASET (WITH PLOTS) ")
    print("==========================================================\n")

    os.makedirs('dashboard/assets', exist_ok=True)
    dashboard_data = {}

    # Setup plotting style
    plt.rcParams['figure.figsize'] = (8, 5)
    sns.set_theme(style='whitegrid')

    # ---------------------------------------------------------
    # 0. Data Preparation
    # ---------------------------------------------------------
    df = pd.read_csv('Iris.csv')
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    
    # Create binary target: 0 if Setosa, 1 otherwise
    df['target'] = (df['Species'] != 'Iris-setosa').astype(int)
    target = 'target'
    
    dashboard_data['data_prep'] = {
        'records_loaded': len(df),
        'features': features
    }

    X_all = df[features].values
    X_std = (X_all - np.mean(X_all, axis=0)) / np.std(X_all, axis=0)

    # ---------------------------------------------------------
    # 1. Vectors
    # ---------------------------------------------------------
    plant_1_features = df[features].iloc[0].values
    
    # Plot Vector Magnitudes
    norms = np.linalg.norm(X_std, axis=1)
    plt.figure()
    plt.hist(norms, bins=25, color='royalblue', edgecolor='white')
    plt.axvline(norms.mean(), color='orange', linestyle='--', label=f'Mean = {norms.mean():.2f}')
    plt.title('Distribution of Flower Vector Magnitudes')
    plt.xlabel('L2 Norm (Magnitude from the "Average" Flower)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig('dashboard/assets/vectors.png', dpi=150)
    plt.close()

    dashboard_data['vectors'] = {
        'description': 'A vector describes one specific flower in 4D space.',
        'purpose': 'In data science, vectors represent a single data point. We can measure their length (magnitude) to see how mathematically "unusual" or extreme a data point is compared to the average.',
        'value': plant_1_features.tolist(),
        'image': 'assets/vectors.png'
    }

    # ---------------------------------------------------------
    # 2. Matrices
    # ---------------------------------------------------------
    X_sample = df[features].iloc[0:5].values
    
    cov_matrix = np.cov(X_std, rowvar=False)
    
    # Plot Covariance Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cov_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, 
                xticklabels=['Sepal L', 'Sepal W', 'Petal L', 'Petal W'], 
                yticklabels=['Sepal L', 'Sepal W', 'Petal L', 'Petal W'])
    plt.title('Covariance Matrix (Features vs Features)')
    plt.tight_layout()
    plt.savefig('dashboard/assets/matrices.png', dpi=150)
    plt.close()

    dashboard_data['matrices'] = {
        'description': 'A matrix groups multiple vectors together. The covariance matrix shows how features relate.',
        'purpose': 'Matrices hold our entire dataset. The covariance matrix is a special square matrix showing if two features increase together (like Petal Length and Width) or move oppositely.',
        'value': X_sample.tolist(),
        'shape': list(X_sample.shape),
        'image': 'assets/matrices.png'
    }

    # ---------------------------------------------------------
    # 3. Dot Product
    # ---------------------------------------------------------
    weights = np.array([0.1, 0.5, 1.5, -0.2]) 
    weighted_sum = np.dot(plant_1_features, weights)
    
    # Plot Pairwise Dot Products (Gram Matrix) for first 50
    subset = X_std[:50]
    gram = subset @ subset.T
    plt.figure()
    sns.heatmap(gram, cmap='plasma', xticklabels=False, yticklabels=False)
    plt.title('Gram Matrix - Pairwise Dot Products (50 Flowers)')
    plt.tight_layout()
    plt.savefig('dashboard/assets/dot_product.png', dpi=150)
    plt.close()

    dashboard_data['dot_product'] = {
        'description': 'The dot product measures similarity or computes a weighted sum.',
        'purpose': 'The dot product calculates a "weighted sum" for predictions, or measures the geometric similarity between two data points (as seen in the Gram Matrix heatmap).',
        'weights': weights.tolist(),
        'result': float(weighted_sum),
        'image': 'assets/dot_product.png'
    }

    # ---------------------------------------------------------
    # 4. Eigenvalues
    # ---------------------------------------------------------
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Plot Eigenvalues (Scree Plot)
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(eigenvalues)+1), sorted(eigenvalues, reverse=True), 'o-', color='crimson')
    plt.title('Eigenvalues (Variance Explained)')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.xticks([1, 2, 3, 4])
    plt.tight_layout()
    plt.savefig('dashboard/assets/eigen.png', dpi=150)
    plt.close()

    dashboard_data['eigen'] = {
        'description': 'Eigenvalues correspond to the variance captured by each eigenvector.',
        'purpose': 'Eigenvectors find the "main directions" hidden inside our data (e.g. general size). Eigenvalues tell us how much variance each direction captures. This is how PCA compresses data.',
        'covariance_matrix': np.round(cov_matrix, 2).tolist(),
        'eigenvalues': np.round(eigenvalues, 2).tolist(),
        'eigenvectors': np.round(eigenvectors, 2).tolist(),
        'image': 'assets/eigen.png'
    }

    # ---------------------------------------------------------
    # 5. Calculus (Gradient Descent)
    # ---------------------------------------------------------
    X = X_std
    y = df[target].values
    w = np.zeros(X.shape[1])
    b = 0.0
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    learning_rate = 0.1
    epsilon = 1e-9
    losses = []
    
    # Run 50 epochs of gradient descent
    for epoch in range(50):
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)
        loss = -np.mean(y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon))
        losses.append(loss)
        
        dz = y_hat - y
        m = len(y)
        dw = np.dot(X.T, dz) / m 
        db = np.sum(dz) / m
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        if epoch == 0:
            fw_loss = loss
            fw_preds = y_hat[:5]
            grad_dw = dw
            grad_db = db

    plt.figure(figsize=(7, 4))
    plt.plot(losses, color='green', linewidth=2)
    plt.title('Loss Curve over 50 Gradient Descent Steps')
    plt.xlabel('Step')
    plt.ylabel('Binary Cross Entropy Loss')
    plt.tight_layout()
    plt.savefig('dashboard/assets/gradient_descent.png', dpi=150)
    plt.close()

    dashboard_data['forward_pass'] = {
        'description': 'Forward pass pushes data through equations to get a prediction.',
        'purpose': 'The forward pass is "using the model". It pushes data through our equations to get a prediction, and the Loss function calculates exactly how wrong our prediction was.',
        'initial_predictions': np.round(fw_preds, 3).tolist(),
        'initial_loss': float(np.round(fw_loss, 4))
    }
    
    dashboard_data['gradients'] = {
        'description': 'The gradient (vector of derivatives) points towards higher loss.',
        'purpose': 'Derivatives measure the "rate of change". The gradient uses the chain rule to tell us precisely how tweaking every single weight will affect our final loss.',
        'dw': grad_dw.tolist(),
        'db': float(grad_db)
    }

    dashboard_data['gradient_descent'] = {
        'description': 'By repeating gradient descent, the loss decreases dramatically.',
        'purpose': 'This is how machines "learn". We take our weights and step opposite to the gradient to reduce our loss. Over many steps, the model becomes accurate, as seen in the loss curve.',
        'updated_weights': w.tolist(),
        'new_loss': float(np.round(losses[-1], 4)),
        'loss_decrease': float(np.round(fw_loss - losses[-1], 4)),
        'image': 'assets/gradient_descent.png'
    }

    # Export to JSON
    with open('dashboard/data.json', 'w') as f:
        json.dump(dashboard_data, f, indent=4)
    print("Data and plots exported to dashboard/ successfully!")

if __name__ == "__main__":
    main()
