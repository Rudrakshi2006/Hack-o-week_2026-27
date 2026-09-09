"""
app.py
Clean, Simple & Interactive Streamlit Web Application for
Student Performance & Success Prediction.
"""

import os
import joblib
import pandas as pd
import streamlit as st

# Page Configuration - Clean & Centered Layout
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)


@st.cache_resource
def load_models():
    """Loads all trained models, preprocessor, and metrics."""
    models_dir = 'models'
    if not os.path.exists(os.path.join(models_dir, 'preprocessor.pkl')):
        st.error("Model files not found. Please run 'python src/train.py' first.")
        st.stop()

    preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))
    reg_models = {
        'Ridge Regression (Best)': joblib.load(os.path.join(models_dir, 'ridge_regression.pkl')),
        'Linear Regression': joblib.load(os.path.join(models_dir, 'linear_regression.pkl')),
        'Lasso Regression': joblib.load(os.path.join(models_dir, 'lasso_regression.pkl')),
        'Polynomial Regression': joblib.load(os.path.join(models_dir, 'polynomial_regression.pkl'))
    }
    cls_models = {
        'Logistic Regression (Best)': joblib.load(os.path.join(models_dir, 'logistic_regression.pkl')),
        'K-Nearest Neighbors': joblib.load(os.path.join(models_dir, 'knn.pkl'))
    }
    metrics_bundle = joblib.load(os.path.join(models_dir, 'model_metrics.pkl'))

    return preprocessor, reg_models, cls_models, metrics_bundle


# Load artifacts
preprocessor, reg_models, cls_models, metrics_bundle = load_models()

# App Header
st.title("🎓 Student Performance & Success Predictor")
st.write("Enter student study habits and details below to predict their **expected exam score** and **pass/fail outcome**.")
st.markdown("---")

# Main Input Section
st.subheader("📋 Student Profile")

col1, col2, col3 = st.columns(3)

with col1:
    hours_studied = st.slider("Study Hours (per week)", min_value=1, max_value=44, value=20)
    attendance = st.slider("Attendance Percentage (%)", min_value=50, max_value=100, value=85)

with col2:
    previous_scores = st.slider("Previous Exam Score", min_value=40, max_value=100, value=75)
    tutoring_sessions = st.slider("Tutoring Sessions (per month)", min_value=0, max_value=8, value=1)

with col3:
    sleep_hours = st.slider("Sleep Hours (per day)", min_value=4, max_value=10, value=7)
    physical_activity = st.slider("Physical Activity (hours/week)", min_value=0, max_value=6, value=3)

# Additional Details in an Expander (Keeps UI clean and uncluttered)
with st.expander("⚙️ Additional Background Factors (Optional)", expanded=False):
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"], index=1)
        access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"], index=1)
        motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"], index=1)
        family_income = st.selectbox("Family Income Level", ["Low", "Medium", "High"], index=1)
        teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"], index=1)

    with ec2:
        extracurricular = st.selectbox("Extracurricular Activities", ["No", "Yes"], index=1)
        internet_access = st.selectbox("Internet Access", ["Yes", "No"], index=0)
        school_type = st.selectbox("School Type", ["Public", "Private"], index=0)
        peer_influence = st.selectbox("Peer Influence", ["Positive", "Neutral", "Negative"], index=0)

    with ec3:
        learning_disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"], index=0)
        parental_edu = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"], index=1)
        distance_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"], index=0)
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)

# Build DataFrame for prediction
input_df = pd.DataFrame([{
    'Hours_Studied': hours_studied,
    'Attendance': attendance,
    'Sleep_Hours': sleep_hours,
    'Previous_Scores': previous_scores,
    'Tutoring_Sessions': tutoring_sessions,
    'Physical_Activity': physical_activity,
    'Parental_Involvement': parental_involvement,
    'Access_to_Resources': access_to_resources,
    'Extracurricular_Activities': extracurricular,
    'Motivation_Level': motivation_level,
    'Internet_Access': internet_access,
    'Family_Income': family_income,
    'Teacher_Quality': teacher_quality,
    'School_Type': school_type,
    'Peer_Influence': peer_influence,
    'Learning_Disabilities': learning_disabilities,
    'Parental_Education_Level': parental_edu,
    'Distance_from_Home': distance_home,
    'Gender': gender
}])

st.markdown("---")

# Predict Button
if st.button("🚀 Predict Student Performance", type="primary", use_container_width=True):
    # Preprocess inputs
    X_trans = preprocessor.transform(input_df)

    # 1. Regression (Score) Prediction
    best_reg_model = reg_models['Ridge Regression (Best)']
    pred_score = float(best_reg_model.predict(X_trans)[0])
    pred_score = max(0.0, min(100.0, round(pred_score, 1)))

    # 2. Classification (Pass/Fail) Prediction
    best_cls_model = cls_models['Logistic Regression (Best)']
    pass_pred = int(best_cls_model.predict(X_trans)[0])
    pass_prob = float(best_cls_model.predict_proba(X_trans)[0][1]) * 100.0

    st.subheader("🎯 Prediction Results")
    
    res1, res2 = st.columns(2)
    with res1:
        st.metric(
            label="📈 Predicted Final Score",
            value=f"{pred_score} / 100"
        )
        st.progress(pred_score / 100.0)

    with res2:
        status_label = "PASS ✅" if pass_pred == 1 else "FAIL ❌"
        st.metric(
            label="🎯 Result Status",
            value=status_label,
            delta=f"{pass_prob:.1f}% Pass Probability"
        )
        if pass_pred == 1:
            st.success("🎉 **Student is likely to PASS!** Consistent study habits and good attendance are paying off.")
        else:
            st.error("⚠️ **Student is AT RISK of failing.** Recommend increasing study hours and attending tutoring sessions.")

    # Model Breakdown Expander
    with st.expander("🔍 View All Model Predictions"):
        all_preds = []
        for name, model in reg_models.items():
            s = max(0.0, min(100.0, round(float(model.predict(X_trans)[0]), 1)))
            all_preds.append({"Task": "Regression (Score)", "Algorithm": name, "Prediction": f"{s} / 100"})
        for name, model in cls_models.items():
            c = int(model.predict(X_trans)[0])
            all_preds.append({"Task": "Classification (Pass/Fail)", "Algorithm": name, "Prediction": "PASS" if c == 1 else "FAIL"})
        
        st.dataframe(pd.DataFrame(all_preds), hide_index=True, use_container_width=True)

st.markdown("---")

# Quick Benchmarks section
with st.expander("📊 View Model Accuracy & Benchmark Metrics"):
    st.write("Performance of all models evaluated on test dataset:")
    tab_reg, tab_cls = st.tabs(["Regression Models", "Classification Models"])
    with tab_reg:
        st.dataframe(metrics_bundle['reg_comparison'], hide_index=True, use_container_width=True)
    with tab_cls:
        st.dataframe(metrics_bundle['cls_comparison'], hide_index=True, use_container_width=True)
