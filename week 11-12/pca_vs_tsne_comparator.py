"""
Quantitative Tournament Comparator: PCA vs t-SNE.
Evaluates both dimensionality reduction methods across standard metrics:
1. 2D Silhouette Score (cluster separation & compactness)
2. 2D 5-NN Classification Accuracy (neighborhood preservation for downstream prediction)
3. Trustworthiness (sklearn.manifold.trustworthiness - penalty for false neighbors)
4. Global Distance Spearman Rank Correlation (macro-geometric structure preservation)
5. Wall-Clock Computation Time (speed & scalability)
6. Out-of-sample inference & Invertibility capabilities
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import StandardScaler


class DimensionalityReductionComparator:
    def __init__(self, X: np.ndarray, y: np.ndarray, labels_str: list = None):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=str)
        self.labels_str = labels_str
        self.scaler = StandardScaler()
        self.X_std = self.scaler.fit_transform(self.X)

    def evaluate_embedding(self, embedding_2d: np.ndarray, name: str, compute_time: float) -> dict:
        """
        Evaluates a 2D projection or manifold embedding across all benchmark metrics.
        """
        emb = np.asarray(embedding_2d, dtype=np.float64)
        
        # 1. Silhouette Score
        sil = float(silhouette_score(emb, self.y))
        
        # 2. 5-NN Cross-Validated Classification Accuracy
        knn = KNeighborsClassifier(n_neighbors=5)
        scores = cross_val_score(knn, emb, self.y, cv=5, scoring="accuracy")
        knn_acc = float(np.mean(scores) * 100.0)
        
        # 3. Trustworthiness (n_neighbors=15)
        trust = float(trustworthiness(self.X_std, emb, n_neighbors=15))
        
        # 4. Global Distance Spearman Rank Correlation (sample points for speed)
        n_sample = min(len(self.X_std), 600)
        idx = np.random.RandomState(42).choice(len(self.X_std), n_sample, replace=False)
        d_high = pdist(self.X_std[idx])
        d_low = pdist(emb[idx])
        spearman_rho, _ = spearmanr(d_high, d_low)
        
        return {
            "Method": name,
            "Silhouette Score": sil,
            "5-NN Accuracy (%)": knn_acc,
            "Trustworthiness": trust,
            "Global Spearman rho": float(spearman_rho),
            "Compute Latency (s)": float(compute_time),
            "Out-of-Sample Projection": "Instant (W^T x)" if "PCA" in name else "Not Supported",
            "Invertible / Lossless": "Yes (Lossy/Exact)" if "PCA" in name else "No (Irreversible)"
        }

    def run_full_tournament(self, pca_2d: np.ndarray, pca_time: float, tsne_2d: np.ndarray, tsne_time: float) -> pd.DataFrame:
        """
        Runs tournament between PCA and t-SNE (at optimal perplexity).
        """
        pca_metrics = self.evaluate_embedding(pca_2d, "PCA (k=2)", pca_time)
        tsne_metrics = self.evaluate_embedding(tsne_2d, "t-SNE (Perp=30)", tsne_time)
        
        df_results = pd.DataFrame([pca_metrics, tsne_metrics])
        return df_results


def run_benchmark(df: pd.DataFrame, feature_names: list, pca_2d: np.ndarray, pca_time: float, tsne_2d: np.ndarray, tsne_time: float):
    X = df[feature_names].values
    y = df["academic_cohort"].values
    comparator = DimensionalityReductionComparator(X, y)
    leaderboard = comparator.run_full_tournament(pca_2d, pca_time, tsne_2d, tsne_time)
    return comparator, leaderboard
