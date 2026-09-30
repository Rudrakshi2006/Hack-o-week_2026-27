"""
Interactive CLI Tool: Student Profile Projection & Inverse Reconstruction.
Allows testing custom student profiles, projecting out-of-sample data points onto PCA coordinates,
predicting academic performance cohorts, and calculating lossy inverse reconstruction errors.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from download_dataset import load_or_generate_dataset, FEATURE_NAMES
from pca_analysis import PCAMathematicalEngine


PRESET_PROFILES = {
    "high_achiever": {
        "study_hours_weekly": 30.0,
        "attendance_rate": 95.0,
        "sleep_hours_daily": 8.0,
        "screen_time_daily": 2.0,
        "extracurricular_hours": 12.0,
        "stress_level": 2.0,
        "prior_gpa": 3.90,
        "assignment_completion_rate": 98.0
    },
    "balanced": {
        "study_hours_weekly": 18.0,
        "attendance_rate": 82.0,
        "sleep_hours_daily": 7.0,
        "screen_time_daily": 4.5,
        "extracurricular_hours": 8.0,
        "stress_level": 5.0,
        "prior_gpa": 3.20,
        "assignment_completion_rate": 80.0
    },
    "distracted": {
        "study_hours_weekly": 8.0,
        "attendance_rate": 55.0,
        "sleep_hours_daily": 5.0,
        "screen_time_daily": 7.5,
        "extracurricular_hours": 3.0,
        "stress_level": 8.5,
        "prior_gpa": 2.10,
        "assignment_completion_rate": 50.0
    }
}


def main():
    parser = argparse.ArgumentParser(
        description="Week 11-12: Interactive Student Lifestyle Projection & Reconstruction Engine"
    )
    parser.add_argument("--preset", choices=["high_achiever", "balanced", "distracted"],
                        help="Load a preset student lifestyle profile")
    parser.add_argument("--study", type=float, help="Weekly study hours (e.g. 5 - 35)")
    parser.add_argument("--attendance", type=float, help="Attendance rate % (e.g. 40 - 100)")
    parser.add_argument("--sleep", type=float, help="Daily sleep hours (e.g. 4 - 9.5)")
    parser.add_argument("--screen", type=float, help="Daily screen time hours (e.g. 1 - 9)")
    parser.add_argument("--extra", type=float, help="Extracurricular hours (e.g. 0 - 20)")
    parser.add_argument("--stress", type=float, help="Stress level (1 - 10)")
    parser.add_argument("--gpa", type=float, help="Prior GPA (1.5 - 4.0)")
    parser.add_argument("--assignment", type=float, help="Assignment completion rate % (35 - 100)")

    args = parser.parse_args()

    # Determine profile
    profile = {}
    if args.preset:
        profile = PRESET_PROFILES[args.preset].copy()
    else:
        profile = PRESET_PROFILES["high_achiever"].copy()

    # Override with explicit CLI args if provided
    if args.study is not None: profile["study_hours_weekly"] = args.study
    if args.attendance is not None: profile["attendance_rate"] = args.attendance
    if args.sleep is not None: profile["sleep_hours_daily"] = args.sleep
    if args.screen is not None: profile["screen_time_daily"] = args.screen
    if args.extra is not None: profile["extracurricular_hours"] = args.extra
    if args.stress is not None: profile["stress_level"] = args.stress
    if args.gpa is not None: profile["prior_gpa"] = args.gpa
    if args.assignment is not None: profile["assignment_completion_rate"] = args.assignment

    # Load baseline dataset and fit PCA engine
    df = load_or_generate_dataset()
    X = np.asarray(df[FEATURE_NAMES].values, dtype=np.float64)
    y = np.asarray(df["academic_cohort"].values, dtype=str)

    pca_engine = PCAMathematicalEngine()
    pca_engine.fit(X, feature_names=FEATURE_NAMES)
    Z_train = pca_engine.transform(X, k=2)

    # Train a 5-NN classifier on 2D PCA space
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(Z_train, y)

    # Convert test profile to numpy vector
    input_vector = np.array([[profile[feat] for feat in FEATURE_NAMES]], dtype=np.float64)

    # 1. 2D Projection
    z_2d = pca_engine.transform(input_vector, k=2)[0]
    pred_cohort = knn.predict([z_2d])[0]
    pred_probs = knn.predict_proba([z_2d])[0]
    confidence = np.max(pred_probs) * 100.0

    # 2. Inverse Reconstruction from 2D coordinates
    reconstructed_vector = pca_engine.inverse_transform(np.array([z_2d]), k=2)[0]

    # Display report
    print("=" * 80)
    print("WEEK 11-12: INTERACTIVE STUDENT PROFILE PROJECTION & RECONSTRUCTION")
    print("=" * 80)
    print("Input Student Profile:")
    for feat in FEATURE_NAMES:
        print(f"  {feat:<28}: {profile[feat]:>6.2f}")

    print("\n--- PCA 2D Out-of-Sample Projection ---")
    sign_pc1 = "+" if z_2d[0] >= 0 else "-"
    sign_pc2 = "+" if z_2d[1] >= 0 else "-"
    print(f"2D Coordinates: PC1 = {sign_pc1}{abs(z_2d[0]):.3f}, PC2 = {sign_pc2}{abs(z_2d[1]):.3f}")
    print(f"Predicted Academic Tier: {pred_cohort} (Confidence: {confidence:.1f}%)")

    print("\n--- PCA Inverse Reconstruction from 2D Coordinates ---")
    print(f"  {'Feature':<30} {'Original':>10} {'Reconstructed':>15} {'Absolute Error':>16}")
    print("  " + "-" * 75)
    for feat, orig_val, rec_val in zip(FEATURE_NAMES, input_vector[0], reconstructed_vector):
        err = abs(orig_val - rec_val)
        print(f"  {feat:<30} {orig_val:>10.2f} {rec_val:>15.2f} {err:>16.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
