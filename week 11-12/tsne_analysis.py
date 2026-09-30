"""
t-Distributed Stochastic Neighbor Embedding (t-SNE) Analysis Engine.
Implements:
- Standardized high-dimensional feature preprocessing
- Perplexity exploration sweep (5, 15, 30, 50)
- Low-dimensional Student-t probability manifold mapping
- KL Divergence tracking & Barnes-Hut optimization
- Extraction of 2D embeddings for comparative benchmarking
"""

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


class TSNEAnalysisEngine:
    def __init__(self, perplexities: list = None, random_state: int = 42, max_iter: int = 500):
        self.perplexities = perplexities if perplexities is not None else [5, 15, 30, 50]
        self.random_state = random_state
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.embeddings_ = {}
        self.kl_divergences_ = {}
        self.execution_times_ = {}

    def fit_transform_sweep(self, X: np.ndarray) -> dict:
        """
        Runs t-SNE for each perplexity in the sweep list and records coordinates, KL divergence, and latency.
        """
        X_std = self.scaler.fit_transform(np.asarray(X, dtype=np.float64))
        
        for perp in self.perplexities:
            t0 = time.time()
            tsne = TSNE(
                n_components=2,
                perplexity=perp,
                random_state=self.random_state,
                max_iter=self.max_iter,
                init="pca",
                learning_rate="auto"
            )
            embedding_2d = tsne.fit_transform(X_std)
            elapsed = time.time() - t0
            
            self.embeddings_[perp] = embedding_2d
            self.kl_divergences_[perp] = float(tsne.kl_divergence_) if hasattr(tsne, "kl_divergence_") else 0.0
            self.execution_times_[perp] = float(elapsed)
            
        return self.embeddings_

    def get_embedding(self, perplexity: int = 30) -> np.ndarray:
        return self.embeddings_.get(perplexity)


def run_tsne_pipeline(df: pd.DataFrame, feature_names: list, perplexities: list = None):
    """
    Convenience function to run t-SNE sweep on a DataFrame.
    """
    X = df[feature_names].values
    tsne_engine = TSNEAnalysisEngine(perplexities=perplexities)
    embeddings = tsne_engine.fit_transform_sweep(X)
    return tsne_engine, embeddings
