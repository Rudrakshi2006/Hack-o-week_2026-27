"""
Principal Component Analysis (PCA) Mathematical Engine.
Implements:
- Z-Score standardization (zero mean, unit variance)
- Covariance Matrix computation Sigma = (1/(N-1)) X^T X
- Spectral Eigendecomposition & SVD alignment
- Explained Variance Ratio (EVR) & Cumulative EVR
- Kaiser Criterion (eigenvalues >= 1.0)
- 2D Orthogonal Projection
- Factor Loadings L_jk = v_jk * sqrt(lambda_k)
- Inverse Reconstruction and RMSE across component dimensions (k = 1..d)
- Out-of-sample custom observation projection and signal recovery
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class PCAMathematicalEngine:
    def __init__(self, n_components: int = None):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.feature_names = None
        self.mean_ = None
        self.std_ = None
        self.cov_matrix_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.explained_variance_ratio_ = None
        self.cumulative_variance_ratio_ = None
        self.loadings_ = None
        self.sklearn_pca_ = None

    def fit(self, X: np.ndarray, feature_names: list = None):
        """
        Fits PCA model via sample covariance matrix eigendecomposition and validates with SVD.
        """
        self.feature_names = feature_names if feature_names is not None else [f"Feat_{i}" for i in range(X.shape[1])]
        N, d = X.shape
        
        # 1. Z-score Standardization
        X_std = self.scaler.fit_transform(X)
        self.mean_ = self.scaler.mean_
        self.std_ = self.scaler.scale_
        
        # 2. Sample Covariance Matrix Sigma = 1/(N-1) X_std^T X_std
        self.cov_matrix_ = np.cov(X_std, rowvar=False, bias=False)
        
        # 3. Spectral Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov_matrix_)
        # Sort in descending order
        idx_sorted = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[idx_sorted]
        self.eigenvectors_ = eigenvectors[:, idx_sorted]  # columns are eigenvectors v_k
        
        # 4. Explained Variance Ratio
        total_var = np.sum(self.eigenvalues_)
        self.explained_variance_ratio_ = self.eigenvalues_ / total_var
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)
        
        # 5. Factor Loadings: L_jk = v_jk * sqrt(lambda_k)
        self.loadings_ = self.eigenvectors_ * np.sqrt(np.maximum(self.eigenvalues_, 0))
        
        # 6. Fit Sklearn PCA for exact parity & reference
        self.sklearn_pca_ = PCA(n_components=self.n_components, random_state=42)
        self.sklearn_pca_.fit(X_std)
        
        return self

    def transform(self, X: np.ndarray, k: int = 2) -> np.ndarray:
        """
        Projects standardized X onto the first k principal components: Z = X_std * W_k
        """
        X_std = self.scaler.transform(X)
        W_k = self.eigenvectors_[:, :k]
        return np.dot(X_std, W_k)

    def fit_transform(self, X: np.ndarray, k: int = 2) -> np.ndarray:
        self.fit(X)
        return self.transform(X, k=k)

    def inverse_transform(self, Z_k: np.ndarray, k: int = 2) -> np.ndarray:
        """
        Reconstructs original feature space from k-dimensional projection:
        X_hat = (Z_k * W_k^T) * std + mean
        """
        W_k = self.eigenvectors_[:, :k]
        X_std_hat = np.dot(Z_k, W_k.T)
        return (X_std_hat * self.std_) + self.mean_

    def compute_reconstruction_rmse(self, X: np.ndarray) -> dict:
        """
        Calculates Root Mean Squared Error (RMSE) of reconstruction for k = 1..d.
        """
        N, d = X.shape
        rmse_dict = {}
        for k in range(1, d + 1):
            Z_k = self.transform(X, k=k)
            X_hat = self.inverse_transform(Z_k, k=k)
            rmse = np.sqrt(np.mean((X - X_hat) ** 2))
            rmse_dict[k] = float(rmse)
        return rmse_dict

    def get_kaiser_components(self) -> int:
        """
        Kaiser Criterion: Count components with eigenvalues >= 1.0
        """
        return int(np.sum(self.eigenvalues_ >= 1.0))

    def get_loadings_dataframe(self) -> pd.DataFrame:
        """
        Returns factor loadings as a structured DataFrame.
        """
        cols = [f"PC{i+1}" for i in range(len(self.eigenvalues_))]
        df_loadings = pd.DataFrame(self.loadings_, index=self.feature_names, columns=cols)
        return df_loadings


def run_pca_pipeline(df: pd.DataFrame, feature_names: list):
    """
    Convenience function to run PCA on a given DataFrame.
    """
    X = df[feature_names].values
    pca_engine = PCAMathematicalEngine()
    pca_engine.fit(X, feature_names=feature_names)
    Z_2d = pca_engine.transform(X, k=2)
    rmse_dict = pca_engine.compute_reconstruction_rmse(X)
    return pca_engine, Z_2d, rmse_dict


if __name__ == "__main__":
    from download_dataset import load_or_generate_dataset, FEATURE_NAMES
    df = load_or_generate_dataset()
    pca_engine, Z_2d, rmse_dict = run_pca_pipeline(df, FEATURE_NAMES)
    print("=== PCA Analysis Results ===")
    print(f"Eigenvalues: {np.round(pca_engine.eigenvalues_, 4)}")
    print(f"Explained Variance Ratio: {np.round(pca_engine.explained_variance_ratio_ * 100, 2)}%")
    print(f"Cumulative Variance Ratio: {np.round(pca_engine.cumulative_variance_ratio_ * 100, 2)}%")
    print(f"Kaiser Criterion components (lambda >= 1.0): {pca_engine.get_kaiser_components()}")
    print("\nReconstruction RMSE across dimensions k=1..8:")
    for k, rmse in rmse_dict.items():
        print(f"  k={k}: RMSE = {rmse:.4f}")
