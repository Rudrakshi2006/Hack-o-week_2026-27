"""
Publication-Quality Visualizer for Dimensionality Reduction (PCA & t-SNE).
Generates 5 comprehensive figures saved in the 'reports/' directory:
1. 01_pca_scree_and_cumulative_variance.png
2. 02_pca_biplot_and_loadings.png
3. 03_tsne_perplexity_exploration.png
4. 04_pca_vs_tsne_2d_projection_showdown.png
5. 05_reconstruction_error_and_benchmark.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Custom curated palette for academic cohorts
COHORT_COLORS = {
    "High Achievers": "#2563eb",       # Royal Blue
    "Balanced Mainstream": "#059669",  # Emerald Green
    "At-Risk / Distracted": "#dc2626"  # Ruby Red
}


def apply_style():
    """Sets a clean, modern aesthetic style for matplotlib."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
        "figure.dpi": 300
    })


def plot_01_pca_scree_and_cumulative_variance(pca_engine, save_path=None):
    """
    Figure 1: PCA Scree Plot (Eigenvalue Decay) and Cumulative Explained Variance.
    """
    apply_style()
    save_path = save_path or os.path.join(REPORTS_DIR, "01_pca_scree_and_cumulative_variance.png")
    
    eigenvalues = pca_engine.eigenvalues_
    exp_var = pca_engine.explained_variance_ratio_ * 100
    cum_var = pca_engine.cumulative_variance_ratio_ * 100
    k_components = np.arange(1, len(eigenvalues) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Left: Scree Plot
    bars = ax1.bar(k_components, eigenvalues, color="#3b82f6", alpha=0.8, edgecolor="#1d4ed8", width=0.55, label=r"Eigenvalue ($\lambda_k$)")
    ax1.plot(k_components, eigenvalues, color="#1e40af", marker="o", linewidth=2.2, markersize=7)
    ax1.axhline(1.0, color="#ef4444", linestyle="--", linewidth=1.8, label=r"Kaiser Criterion ($\lambda = 1.0$)")
    
    # Annotate elbow
    ax1.annotate(r"Elbow Point ($k=2$)" + f"\n" + r"$\lambda_2 = " + f"{eigenvalues[1]:.2f}$",
                 xy=(2, eigenvalues[1]), xytext=(3.2, eigenvalues[1] + 1.0),
                 arrowprops=dict(facecolor="#1e293b", shrink=0.08, width=1.5, headwidth=7),
                 fontsize=10, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#f1f5f9", ec="#cbd5e1"))
    
    for bar, val in zip(bars, eigenvalues):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.12, f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_title("PCA Scree Plot & Kaiser Criterion", pad=12)
    ax1.set_xlabel(r"Principal Component Index ($k$)")
    ax1.set_ylabel(r"Eigenvalue ($\lambda$)")
    ax1.set_xticks(k_components)
    ax1.set_ylim(0, max(eigenvalues) * 1.25)
    ax1.legend(loc="upper right", frameon=True)
    
    # Right: Cumulative Explained Variance
    ax2.plot(k_components, cum_var, color="#059669", marker="s", linewidth=2.5, markersize=7.5, label="Cumulative Variance")
    ax2.bar(k_components, exp_var, color="#10b981", alpha=0.45, edgecolor="#047857", width=0.5, label="Individual EVR (%)")
    
    # Thresholds
    ax2.axhline(70.0, color="#f59e0b", linestyle=":", linewidth=1.5, label="70% Variance Threshold")
    ax2.axhline(90.0, color="#8b5cf6", linestyle=":", linewidth=1.5, label="90% Variance Threshold")
    
    for k, c_val, e_val in zip(k_components, cum_var, exp_var):
        ax2.text(k, c_val + 2.2, f"{c_val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#065f46")
        
    ax2.set_title("Individual & Cumulative Explained Variance", pad=12)
    ax2.set_xlabel(r"Number of Principal Components ($k$)")
    ax2.set_ylabel("Explained Variance Ratio (%)")
    ax2.set_xticks(k_components)
    ax2.set_ylim(0, 110)
    ax2.legend(loc="lower right", frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path


def plot_02_pca_biplot_and_loadings(pca_engine, df, feature_names, save_path=None):
    """
    Figure 2: 2D Biplot (Scores + Loadings vectors) & Factor Loadings Heatmap.
    """
    apply_style()
    save_path = save_path or os.path.join(REPORTS_DIR, "02_pca_biplot_and_loadings.png")
    
    X = df[feature_names].values
    Z_2d = pca_engine.transform(X, k=2)
    loadings = pca_engine.loadings_[:, :2]  # Loadings for PC1 and PC2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left: 2D Biplot
    cohorts = df["academic_cohort"].unique()
    for cohort in cohorts:
        mask = df["academic_cohort"] == cohort
        ax1.scatter(Z_2d[mask, 0], Z_2d[mask, 1], label=cohort, color=COHORT_COLORS.get(cohort, "#64748b"),
                    alpha=0.6, edgecolors="none", s=35)
        
    # Plot factor loading vectors (scaled for visibility)
    scale_factor = 3.5
    for i, (feat, (l1, l2)) in enumerate(zip(feature_names, loadings)):
        ax1.arrow(0, 0, l1 * scale_factor, l2 * scale_factor, color="#1e293b", width=0.035, head_width=0.18,
                  head_length=0.25, length_includes_head=True, alpha=0.85, zorder=5)
        # Position label slightly beyond arrow tip
        ax1.text(l1 * scale_factor * 1.15, l2 * scale_factor * 1.15, feat.replace("_", "\n"),
                 color="#0f172a", fontsize=8.5, fontweight="bold", ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#cbd5e1", alpha=0.85), zorder=6)
        
    ax1.axhline(0, color="#94a3b8", linestyle="--", linewidth=0.9)
    ax1.axvline(0, color="#94a3b8", linestyle="--", linewidth=0.9)
    ax1.set_title("PCA 2D Biplot (Observations & Feature Loadings)", pad=12)
    ax1.set_xlabel(f"PC1 ({pca_engine.explained_variance_ratio_[0]*100:.1f}% Variance)")
    ax1.set_ylabel(f"PC2 ({pca_engine.explained_variance_ratio_[1]*100:.1f}% Variance)")
    ax1.legend(loc="upper left", frameon=True)
    
    # Right: Factor Loadings Bar Chart / Matrix
    df_loadings = pd.DataFrame(loadings, index=[f.replace("_", " ").title() for f in feature_names], columns=["PC1", "PC2"])
    
    y_pos = np.arange(len(feature_names))
    width = 0.38
    
    ax2.barh(y_pos - width/2, df_loadings["PC1"], height=width, color="#3b82f6", label="PC1 Loadings", edgecolor="#1d4ed8")
    ax2.barh(y_pos + width/2, df_loadings["PC2"], height=width, color="#10b981", label="PC2 Loadings", edgecolor="#047857")
    
    ax2.axvline(0, color="#475569", linestyle="-", linewidth=1.0)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_loadings.index, fontweight="semibold")
    ax2.invert_yaxis()
    ax2.set_title("Factor Loadings (Feature-Component Correlations)", pad=12)
    ax2.set_xlabel(r"Loading Coefficient ($L_{jk} = v_{jk} \sqrt{\lambda_k}$)")
    ax2.set_xlim(-1.1, 1.1)
    ax2.legend(loc="lower right", frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path


def plot_03_tsne_perplexity_exploration(embeddings_dict, df, save_path=None):
    """
    Figure 3: 2x2 grid of t-SNE embeddings across Perplexity = [5, 15, 30, 50].
    """
    apply_style()
    save_path = save_path or os.path.join(REPORTS_DIR, "03_tsne_perplexity_exploration.png")
    
    perplexities = list(embeddings_dict.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    cohorts = df["academic_cohort"].unique()
    y_labels = np.asarray(df["academic_cohort"], dtype=str)
    
    for idx, perp in enumerate(perplexities):
        ax = axes[idx]
        emb = embeddings_dict[perp]
        sil = silhouette_score(emb, y_labels)
        
        for cohort in cohorts:
            mask = df["academic_cohort"] == cohort
            ax.scatter(emb[mask, 0], emb[mask, 1], label=cohort, color=COHORT_COLORS.get(cohort, "#64748b"),
                       alpha=0.65, s=35, edgecolors="none")
            
        ax.set_title(f"Perplexity = {perp} (Silhouette = {sil:.3f})", pad=10)
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        if idx == 0:
            ax.legend(loc="upper right", frameon=True, fontsize=9)
            
    plt.suptitle("t-SNE Perplexity Exploration Sweep (Local vs Global Topology)", fontsize=15, y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path


def plot_04_pca_vs_tsne_2d_projection_showdown(pca_2d, tsne_2d, df, save_path=None):
    """
    Figure 4: Direct 2D Projection Showdown (PCA vs t-SNE Perp=30).
    """
    apply_style()
    save_path = save_path or os.path.join(REPORTS_DIR, "04_pca_vs_tsne_2d_projection_showdown.png")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    cohorts = df["academic_cohort"].unique()
    y_labels = np.asarray(df["academic_cohort"], dtype=str)
    
    sil_pca = silhouette_score(pca_2d, y_labels)
    sil_tsne = silhouette_score(tsne_2d, y_labels)
    
    # Left: PCA 2D
    for cohort in cohorts:
        mask = df["academic_cohort"] == cohort
        ax1.scatter(pca_2d[mask, 0], pca_2d[mask, 1], label=cohort, color=COHORT_COLORS.get(cohort, "#64748b"),
                    alpha=0.65, s=40, edgecolors="none")
    ax1.set_title(f"PCA: Linear Orthogonal Projection (Silhouette = {sil_pca:.3f})", pad=12)
    ax1.set_xlabel("Principal Component 1 (Global Academic Gradient)")
    ax1.set_ylabel("Principal Component 2 (Lifestyle & Balance)")
    ax1.legend(loc="upper right", frameon=True)
    
    # Right: t-SNE 2D
    for cohort in cohorts:
        mask = df["academic_cohort"] == cohort
        ax2.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1], label=cohort, color=COHORT_COLORS.get(cohort, "#64748b"),
                    alpha=0.65, s=40, edgecolors="none")
    ax2.set_title(f"t-SNE: Non-Linear Manifold Embedding [Perp=30] (Silhouette = {sil_tsne:.3f})", pad=12)
    ax2.set_xlabel("t-SNE Dimension 1")
    ax2.set_ylabel("t-SNE Dimension 2")
    ax2.legend(loc="upper right", frameon=True)
    
    plt.suptitle("PCA vs t-SNE: Global Linear Variance vs Local Manifold Clustering", fontsize=15, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path


def plot_05_reconstruction_error_and_benchmark(rmse_dict, leaderboard_df, save_path=None):
    """
    Figure 5: PCA Reconstruction RMSE decay curve and Tournament Benchmark comparison.
    """
    apply_style()
    save_path = save_path or os.path.join(REPORTS_DIR, "05_reconstruction_error_and_benchmark.png")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    
    # Left: PCA Reconstruction RMSE
    ks = list(rmse_dict.keys())
    rmses = list(rmse_dict.values())
    
    ax1.plot(ks, rmses, color="#dc2626", marker="o", linewidth=2.5, markersize=8, label="Reconstruction RMSE")
    ax1.fill_between(ks, 0, rmses, color="#fee2e2", alpha=0.5)
    
    for k, rmse in zip(ks, rmses):
        ax1.text(k, rmse + 0.02, f"{rmse:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        
    ax1.set_title(r"PCA Inverse Signal Reconstruction Error ($k = 1 \dots 8$)", pad=12)
    ax1.set_xlabel(r"Number of Principal Components Retained ($k$)")
    ax1.set_ylabel("Reconstruction RMSE")
    ax1.set_xticks(ks)
    ax1.set_ylim(0, max(rmses) * 1.25)
    ax1.legend(loc="upper right", frameon=True)
    
    # Right: Grouped Benchmark Bar Chart
    metrics = ["Silhouette Score", "Trustworthiness", "Global Spearman rho"]
    pca_vals = [leaderboard_df.loc[leaderboard_df["Method"].str.contains("PCA"), m].values[0] for m in metrics]
    tsne_vals = [leaderboard_df.loc[leaderboard_df["Method"].str.contains("t-SNE"), m].values[0] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    rects1 = ax2.bar(x - width/2, pca_vals, width, label="PCA (k=2)", color="#3b82f6", edgecolor="#1d4ed8")
    rects2 = ax2.bar(x + width/2, tsne_vals, width, label="t-SNE (Perp=30)", color="#8b5cf6", edgecolor="#6d28d9")
    
    for rect in rects1:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    ax2.set_title("Quantitative Metric Showdown (PCA vs t-SNE)", pad=12)
    ax2.set_ylabel("Metric Score (0.0 to 1.0)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontweight="semibold")
    ax2.set_ylim(0, 1.18)
    ax2.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path


def render_all_figures(pca_engine, tsne_engine, df, feature_names, leaderboard_df):
    """Renders all 5 diagnostic figures."""
    print("[Visualizer] Rendering all 5 publication-quality figures in reports/...")
    p1 = plot_01_pca_scree_and_cumulative_variance(pca_engine)
    p2 = plot_02_pca_biplot_and_loadings(pca_engine, df, feature_names)
    p3 = plot_03_tsne_perplexity_exploration(tsne_engine.embeddings_, df)
    p4 = plot_04_pca_vs_tsne_2d_projection_showdown(pca_engine.transform(df[feature_names].values, k=2), tsne_engine.get_embedding(30), df)
    p5 = plot_05_reconstruction_error_and_benchmark(pca_engine.compute_reconstruction_rmse(df[feature_names].values), leaderboard_df)
    print(f"[Visualizer] All figures saved successfully:\n  1. {p1}\n  2. {p2}\n  3. {p3}\n  4. {p4}\n  5. {p5}")
