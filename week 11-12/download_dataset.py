"""
Dataset generation and management module for Dimensionality Reduction (PCA & t-SNE).
Generates a realistic synthetic dataset: 'Student Lifestyle & Academic Performance'
(N=1,000, 8 lifestyle & academic features across 3 distinct student cohorts).
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_FILE = os.path.join(DATA_DIR, "student_lifestyle_academic.csv")

FEATURE_NAMES = [
    "study_hours_weekly",
    "attendance_rate",
    "sleep_hours_daily",
    "screen_time_daily",
    "extracurricular_hours",
    "stress_level",
    "prior_gpa",
    "assignment_completion_rate"
]

TARGET_NAMES = ["High Achievers", "Balanced Mainstream", "At-Risk / Distracted"]


def generate_student_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic multi-modal dataset representing undergraduate students.
    
    Cohorts:
    - High Achievers (~30%): High study hours, high attendance, high GPA, low screen time.
    - Balanced Mainstream (~45%): Moderate study hours, balanced sleep/extracurriculars, solid GPA.
    - At-Risk / Distracted (~25%): Low attendance, high screen time, high stress, lower GPA.
    """
    np.random.seed(random_state)
    
    n_high = int(0.30 * n_samples)
    n_balanced = int(0.45 * n_samples)
    n_at_risk = n_samples - n_high - n_balanced

    # 1. High Achievers
    high_study = np.clip(np.random.normal(28.0, 3.5, n_high), 18.0, 38.0)
    high_attend = np.clip(np.random.normal(92.0, 4.0, n_high), 80.0, 100.0)
    high_sleep = np.clip(np.random.normal(7.5, 0.8, n_high), 6.0, 9.5)
    high_screen = np.clip(np.random.normal(2.5, 0.8, n_high), 1.0, 4.5)
    high_extra = np.clip(np.random.normal(12.0, 3.0, n_high), 4.0, 20.0)
    high_stress = np.clip(np.random.normal(3.5, 1.2, n_high), 1.0, 6.5)
    high_gpa = np.clip(np.random.normal(3.80, 0.15, n_high), 3.40, 4.00)
    high_assign = np.clip(np.random.normal(95.0, 3.5, n_high), 85.0, 100.0)
    high_labels = ["High Achievers"] * n_high

    df_high = pd.DataFrame({
        "study_hours_weekly": high_study,
        "attendance_rate": high_attend,
        "sleep_hours_daily": high_sleep,
        "screen_time_daily": high_screen,
        "extracurricular_hours": high_extra,
        "stress_level": high_stress,
        "prior_gpa": high_gpa,
        "assignment_completion_rate": high_assign,
        "academic_cohort": high_labels
    })

    # 2. Balanced Mainstream
    bal_study = np.clip(np.random.normal(18.0, 3.0, n_balanced), 12.0, 25.0)
    bal_attend = np.clip(np.random.normal(80.0, 6.0, n_balanced), 65.0, 92.0)
    bal_sleep = np.clip(np.random.normal(7.0, 1.0, n_balanced), 5.5, 9.0)
    bal_screen = np.clip(np.random.normal(4.5, 1.0, n_balanced), 2.5, 6.5)
    bal_extra = np.clip(np.random.normal(8.0, 3.0, n_balanced), 2.0, 16.0)
    bal_stress = np.clip(np.random.normal(5.5, 1.2, n_balanced), 3.0, 8.0)
    bal_gpa = np.clip(np.random.normal(3.15, 0.25, n_balanced), 2.60, 3.65)
    bal_assign = np.clip(np.random.normal(80.0, 7.0, n_balanced), 65.0, 92.0)
    bal_labels = ["Balanced Mainstream"] * n_balanced

    df_balanced = pd.DataFrame({
        "study_hours_weekly": bal_study,
        "attendance_rate": bal_attend,
        "sleep_hours_daily": bal_sleep,
        "screen_time_daily": bal_screen,
        "extracurricular_hours": bal_extra,
        "stress_level": bal_stress,
        "prior_gpa": bal_gpa,
        "assignment_completion_rate": bal_assign,
        "academic_cohort": bal_labels
    })

    # 3. At-Risk / Distracted
    risk_study = np.clip(np.random.normal(9.0, 2.5, n_at_risk), 4.0, 15.0)
    risk_attend = np.clip(np.random.normal(60.0, 8.0, n_at_risk), 40.0, 75.0)
    risk_sleep = np.clip(np.random.normal(5.2, 1.0, n_at_risk), 3.5, 7.0)
    risk_screen = np.clip(np.random.normal(7.2, 1.2, n_at_risk), 5.0, 10.0)
    risk_extra = np.clip(np.random.normal(3.5, 2.0, n_at_risk), 0.0, 8.0)
    risk_stress = np.clip(np.random.normal(8.2, 1.0, n_at_risk), 5.5, 10.0)
    risk_gpa = np.clip(np.random.normal(2.25, 0.35, n_at_risk), 1.50, 2.90)
    risk_assign = np.clip(np.random.normal(55.0, 9.0, n_at_risk), 30.0, 72.0)
    risk_labels = ["At-Risk / Distracted"] * n_at_risk

    df_risk = pd.DataFrame({
        "study_hours_weekly": risk_study,
        "attendance_rate": risk_attend,
        "sleep_hours_daily": risk_sleep,
        "screen_time_daily": risk_screen,
        "extracurricular_hours": risk_extra,
        "stress_level": risk_stress,
        "prior_gpa": risk_gpa,
        "assignment_completion_rate": risk_assign,
        "academic_cohort": risk_labels
    })

    # Combine and shuffle
    df = pd.concat([df_high, df_balanced, df_risk], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    # Round columns realistically
    df["study_hours_weekly"] = df["study_hours_weekly"].round(1)
    df["attendance_rate"] = df["attendance_rate"].round(1)
    df["sleep_hours_daily"] = df["sleep_hours_daily"].round(1)
    df["screen_time_daily"] = df["screen_time_daily"].round(1)
    df["extracurricular_hours"] = df["extracurricular_hours"].round(1)
    df["stress_level"] = df["stress_level"].round(1)
    df["prior_gpa"] = df["prior_gpa"].round(2)
    df["assignment_completion_rate"] = df["assignment_completion_rate"].round(1)

    return df


def load_or_generate_dataset(force_recreate: bool = False) -> pd.DataFrame:
    """
    Loads cached dataset from CSV or generates and saves it if not found.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if not force_recreate and os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    
    df = generate_student_dataset(n_samples=1000, random_state=42)
    df.to_csv(DATA_FILE, index=False)
    return df


if __name__ == "__main__":
    df = load_or_generate_dataset(force_recreate=True)
    print(f"[Dataset] Generated {len(df)} rows across {len(FEATURE_NAMES)} features.")
    print(f"[Dataset] Saved to: {DATA_FILE}")
    print(df.head())
    print("\nCohort distribution:")
    print(df["academic_cohort"].value_counts())
