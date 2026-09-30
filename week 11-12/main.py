"""
Main execution pipeline for Week 11-12: Dimensionality Reduction (PCA & t-SNE).
Executes the full end-to-end analytical workflow:
1. Dataset generation & loading
2. PCA Mathematical Decomposition & Inverse Reconstruction
3. t-SNE Manifold Optimization & Perplexity Exploration Sweep
4. Quantitative Tournament Benchmarking
5. Rendering of all 5 Diagnostic Figures in reports/
"""

import os
import time
import numpy as np
import pandas as pd
from download_dataset import load_or_generate_dataset, FEATURE_NAMES
from pca_analysis import PCAMathematicalEngine
from tsne_analysis import TSNEAnalysisEngine
from pca_vs_tsne_comparator import DimensionalityReductionComparator
from visualizer import render_all_figures


def main():
    print("=" * 80)
    print("WEEK 11-12: DIMENSIONALITY REDUCTION — PCA vs t-SNE BENCHMARK PIPELINE")
    print("=" * 80)

    # 1. Dataset Loading
    print("\n[Step 1/5] Loading Student Lifestyle & Academic Performance Dataset...")
    df = load_or_generate_dataset()
    X = df[FEATURE_NAMES].values
    y = df["academic_cohort"].values
    print(f"  -> Total records: N = {len(df)}")
    print(f"  -> Feature dimensions: d = {X.shape[1]}")
    print(f"  -> Cohort distribution: {dict(df['academic_cohort'].value_counts())}")

    # 2. PCA Engine
    print("\n[Step 2/5] Fitting PCA Mathematical Engine...")
    t0_pca = time.time()
    pca_engine = PCAMathematicalEngine()
    pca_engine.fit(X, feature_names=FEATURE_NAMES)
    pca_time = time.time() - t0_pca
    Z_2d_pca = pca_engine.transform(X, k=2)

    evr = pca_engine.explained_variance_ratio_ * 100
    cum_evr = pca_engine.cumulative_variance_ratio_ * 100
    kaiser_k = pca_engine.get_kaiser_components()

    print(f"  -> Eigenvalues (lambda): {np.round(pca_engine.eigenvalues_, 3)}")
    print(f"  -> PC1 Variance: {evr[0]:.2f}% | PC2 Variance: {evr[1]:.2f}% | 2D Total: {cum_evr[1]:.2f}%")
    print(f"  -> Kaiser Criterion (lambda >= 1.0): {kaiser_k} components exceed baseline variance")
    print(f"  -> Compute Latency: {pca_time:.4f} seconds")

    # 3. t-SNE Engine Sweep
    print("\n[Step 3/5] Fitting t-SNE Manifold Engine (Perplexity Sweep [5, 15, 30, 50])...")
    tsne_engine = TSNEAnalysisEngine(perplexities=[5, 15, 30, 50], random_state=42)
    tsne_engine.fit_transform_sweep(X)

    for perp in [5, 15, 30, 50]:
        kl = tsne_engine.kl_divergences_[perp]
        lat = tsne_engine.execution_times_[perp]
        print(f"  -> Perp = {perp:2d} | KL Divergence = {kl:.4f} | Latency = {lat:.2f}s")

    tsne_2d_optimal = tsne_engine.get_embedding(30)
    tsne_time_optimal = tsne_engine.execution_times_[30]

    # 4. Quantitative Tournament Benchmark
    print("\n[Step 4/5] Executing Quantitative Tournament Benchmark (PCA vs t-SNE)...")
    comparator = DimensionalityReductionComparator(X, y)
    leaderboard_df = comparator.run_full_tournament(Z_2d_pca, pca_time, tsne_2d_optimal, tsne_time_optimal)

    print("\n" + "-" * 80)
    print("QUANTITATIVE BENCHMARK LEADERBOARD")
    print("-" * 80)
    print(leaderboard_df.to_string(index=False))
    print("-" * 80)

    # 5. Visualizer
    print("\n[Step 5/5] Generating Publication-Quality Figures (300 DPI)...")
    render_all_figures(pca_engine, tsne_engine, df, FEATURE_NAMES, leaderboard_df)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("Artifacts generated in './reports/':")
    print("  - 01_pca_scree_and_cumulative_variance.png")
    print("  - 02_pca_biplot_and_loadings.png")
    print("  - 03_tsne_perplexity_exploration.png")
    print("  - 04_pca_vs_tsne_2d_projection_showdown.png")
    print("  - 05_reconstruction_error_and_benchmark.png")
    print("\nTo test custom student projections, run:")
    print("  python interactive_projection.py --preset high_achiever")
    print("=" * 80)


if __name__ == "__main__":
    main()
