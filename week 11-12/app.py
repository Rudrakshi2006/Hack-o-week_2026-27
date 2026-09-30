"""
Flask Web Application Server for Dimensionality Reduction (PCA & t-SNE) Dashboard.
Serves interactive APIs and premium frontend web interface.
"""

import os
import time
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template, send_from_directory
from sklearn.neighbors import KNeighborsClassifier

from download_dataset import load_or_generate_dataset, FEATURE_NAMES, TARGET_NAMES
from pca_analysis import PCAMathematicalEngine
from tsne_analysis import TSNEAnalysisEngine
from pca_vs_tsne_comparator import DimensionalityReductionComparator

app = Flask(__name__, template_folder="templates", static_folder="static")

# Global Cache
DATA = {}


def init_app_data():
    global DATA
    print("[App] Initializing dataset and analytical engines...")
    df = load_or_generate_dataset()
    X = np.asarray(df[FEATURE_NAMES].values, dtype=np.float64)
    y = np.asarray(df["academic_cohort"].values, dtype=str)

    # Fit PCA
    t0 = time.time()
    pca_engine = PCAMathematicalEngine()
    pca_engine.fit(X, feature_names=FEATURE_NAMES)
    pca_time = time.time() - t0
    pca_2d = pca_engine.transform(X, k=2)

    # Train 5-NN classifier on PCA 2D coordinates for instant real-time predictions
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(pca_2d, y)

    # Fit t-SNE across perplexities [5, 15, 30, 50]
    tsne_engine = TSNEAnalysisEngine(perplexities=[5, 15, 30, 50], random_state=42, max_iter=500)
    tsne_engine.fit_transform_sweep(X)
    tsne_2d_30 = tsne_engine.get_embedding(30)
    tsne_time_30 = tsne_engine.execution_times_[30]

    # Benchmark Tournament
    comparator = DimensionalityReductionComparator(X, y)
    leaderboard = comparator.run_full_tournament(pca_2d, pca_time, tsne_2d_30, tsne_time_30)

    rmse_dict = pca_engine.compute_reconstruction_rmse(X)

    DATA = {
        "df": df,
        "X": X,
        "y": y,
        "pca_engine": pca_engine,
        "pca_2d": pca_2d,
        "pca_time": pca_time,
        "knn_classifier": knn_classifier,
        "tsne_engine": tsne_engine,
        "leaderboard": leaderboard,
        "rmse_dict": rmse_dict
    }
    print("[App] Initialization complete!")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/reports/<path:filename>")
def serve_report(filename):
    return send_from_directory("reports", filename)


@app.route("/api/summary")
def get_summary():
    df = DATA["df"]
    cohort_counts = df["academic_cohort"].value_counts().to_dict()
    feature_stats = {}
    for col in FEATURE_NAMES:
        feature_stats[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "std": float(df[col].std())
        }
    return jsonify({
        "total_records": len(df),
        "total_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "cohort_counts": cohort_counts,
        "feature_stats": feature_stats
    })


@app.route("/api/pca")
def get_pca():
    pca = DATA["pca_engine"]
    df = DATA["df"]
    pca_2d = DATA["pca_2d"]

    records = []
    for i in range(len(df)):
        records.append({
            "id": i + 1,
            "pc1": float(pca_2d[i, 0]),
            "pc2": float(pca_2d[i, 1]),
            "cohort": str(df.iloc[i]["academic_cohort"]),
            "study": float(df.iloc[i]["study_hours_weekly"]),
            "attendance": float(df.iloc[i]["attendance_rate"]),
            "sleep": float(df.iloc[i]["sleep_hours_daily"]),
            "screen": float(df.iloc[i]["screen_time_daily"]),
            "gpa": float(df.iloc[i]["prior_gpa"])
        })

    loadings_list = []
    for idx, feat in enumerate(FEATURE_NAMES):
        loadings_list.append({
            "feature": feat,
            "pc1": float(pca.loadings_[idx, 0]),
            "pc2": float(pca.loadings_[idx, 1])
        })

    return jsonify({
        "eigenvalues": [float(x) for x in pca.eigenvalues_],
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "cumulative_variance_ratio": [float(x) for x in pca.cumulative_variance_ratio_],
        "kaiser_components": pca.get_kaiser_components(),
        "points": records,
        "loadings": loadings_list,
        "rmse": DATA["rmse_dict"],
        "compute_time": DATA["pca_time"]
    })


@app.route("/api/tsne")
def get_tsne():
    tsne = DATA["tsne_engine"]
    df = DATA["df"]

    results = {}
    for perp in tsne.perplexities:
        emb = tsne.get_embedding(perp)
        points = []
        for i in range(len(df)):
            points.append({
                "id": i + 1,
                "x": float(emb[i, 0]),
                "y": float(emb[i, 1]),
                "cohort": str(df.iloc[i]["academic_cohort"]),
                "gpa": float(df.iloc[i]["prior_gpa"]),
                "study": float(df.iloc[i]["study_hours_weekly"])
            })
        results[str(perp)] = {
            "points": points,
            "kl_divergence": tsne.kl_divergences_[perp],
            "execution_time": tsne.execution_times_[perp]
        }

    return jsonify(results)


@app.route("/api/leaderboard")
def get_leaderboard():
    leaderboard = DATA["leaderboard"]
    return jsonify(leaderboard.to_dict(orient="records"))


@app.route("/api/project", methods=["POST"])
def project_student():
    data = request.json or {}
    pca = DATA["pca_engine"]
    knn = DATA["knn_classifier"]

    # Gather 8 features
    input_vals = [
        float(data.get(feat, 0.0)) for feat in FEATURE_NAMES
    ]
    input_vector = np.array([input_vals], dtype=np.float64)

    # Project to 2D
    z_2d = pca.transform(input_vector, k=2)[0]

    # Predict cohort and confidence
    pred_cohort = knn.predict([z_2d])[0]
    probs = knn.predict_proba([z_2d])[0]
    classes = knn.classes_
    confidence = float(np.max(probs) * 100.0)

    # Inverse reconstruction
    recon_vector = pca.inverse_transform(np.array([z_2d]), k=2)[0]

    reconstruction_table = []
    for feat, orig_v, rec_v in zip(FEATURE_NAMES, input_vals, recon_vector):
        reconstruction_table.append({
            "feature": feat,
            "original": float(orig_v),
            "reconstructed": round(float(rec_v), 2),
            "absolute_error": round(float(abs(orig_v - rec_v)), 2)
        })

    return jsonify({
        "pc1": float(z_2d[0]),
        "pc2": float(z_2d[1]),
        "predicted_cohort": str(pred_cohort),
        "confidence": confidence,
        "probabilities": {str(c): round(float(p) * 100, 1) for c, p in zip(classes, probs)},
        "reconstruction": reconstruction_table
    })


if __name__ == "__main__":
    init_app_data()
    print("\n" + "=" * 60)
    print("DimensionLab Web App running at: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
