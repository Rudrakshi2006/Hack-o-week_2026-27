# 🎓 Student Performance & Success Predictor
### *Hack-O-Week 5 & 6 — Machine Learning Project*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Project Title
**Student Performance & Success Predictor**

---

## 2. Project Overview
The **Student Performance & Success Predictor** is a production-ready Machine Learning system that models and forecasts academic performance based on multi-dimensional student data (study habits, attendance, socio-economic factors, and school environment). The application solves two core tasks:
- **Task A (Regression):** Continuous prediction of final exam score ($0 - 100$).
- **Task B (Classification):** Binary prediction of academic outcome (**PASS** vs **FAIL**).

---

## 3. Problem Statement
Academic failure and underperformance frequently occur due to late detection of learning difficulties, poor study habits, or low attendance. Educational institutions require transparent, interpretable machine learning models to anticipate student needs early and allocate tutoring or advisory interventions proactively.

---

## 4. Objectives
1. Implement and evaluate **4 Regression Algorithms**: Linear Regression, Polynomial Regression (Degree 2), Ridge Regression (L2), and Lasso Regression (L1).
2. Implement and evaluate **2 Classification Algorithms**: Logistic Regression and K-Nearest Neighbors (KNN with tuned $K$).
3. Eliminate **data leakage** through strict training-time preprocessing pipelines (`ColumnTransformer`).
4. Build an interactive **5-Page Streamlit Web App** for live predictions, model benchmarking, and educational insights.

---

## 5. Features
- **Continuous Score Predictor:** Estimates final exam score with real-time multi-model comparisons.
- **Pass/Fail Classifier:** Predicts pass probability ($\ge 65 \implies \text{PASS}$) with confidence metrics.
- **Benchmark Dashboard:** Visualizes MAE, RMSE, $R^2$, Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
- **Educational Guide:** Explains the math, strengths, and limitations of each algorithm for college-level learning.
- **Zero Data Leakage:** Preprocessing transformations (imputation, scaling, one-hot encoding) are strictly fitted on the training split.

---

## 6. Dataset
- **Source:** Public Student Performance Factors Dataset
- **Total Records:** 6,607 student entries
- **Total Features:** 19 predictor features + 1 target (`Exam_Score`)
- **Missing Value Handling:** Median imputation for numerical features; Mode imputation for categorical features.

### Feature Dictionary
| Feature | Type | Description |
| :--- | :--- | :--- |
| `Hours_Studied` | Numerical | Weekly hours spent studying ($1 - 44$) |
| `Attendance` | Numerical | Class attendance percentage ($50 - 100\%$) |
| `Sleep_Hours` | Numerical | Average daily sleep hours ($4 - 10$) |
| `Previous_Scores` | Numerical | Previous exam score ($40 - 100$) |
| `Tutoring_Sessions`| Numerical | Monthly tutoring sessions ($0 - 8$) |
| `Physical_Activity`| Numerical | Weekly physical activity hours ($0 - 6$) |
| `Parental_Involvement` | Categorical | Low, Medium, High |
| `Access_to_Resources` | Categorical | Low, Medium, High |
| `Extracurricular_Activities` | Categorical | Yes, No |
| `Motivation_Level` | Categorical | Low, Medium, High |
| `Internet_Access` | Categorical | Yes, No |
| `Family_Income` | Categorical | Low, Medium, High |
| `Teacher_Quality` | Categorical | Low, Medium, High |
| `School_Type` | Categorical | Public, Private |
| `Peer_Influence` | Categorical | Positive, Neutral, Negative |
| `Learning_Disabilities` | Categorical | Yes, No |
| `Parental_Education_Level` | Categorical | High School, College, Postgraduate |
| `Distance_from_Home` | Categorical | Near, Moderate, Far |
| `Gender` | Categorical | Male, Female |
| **`Exam_Score`** | **Target (Reg)** | Final academic score ($0 - 100$) |
| **`Pass_Fail`** | **Target (Cls)** | 1 (Pass: $\ge 65$), 0 (Fail: $< 65$) |

---

## 7. Technologies Used
- **Language:** Python 3.10+
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`
- **Visualization:** `matplotlib`, `seaborn`
- **Model Serialization:** `joblib`
- **Web Application:** `streamlit`

---

## 8. ML Algorithms
### Regression
1. **Linear Regression (OLS):** Models direct linear relationships ($y = \mathbf{w}^T \mathbf{x} + b$).
2. **Polynomial Regression (Degree 2):** Captures quadratic feature combinations ($x_i^2, x_i x_j$).
3. **Ridge Regression:** Penalizes large weights using L2 norm ($\alpha \sum w_i^2$).
4. **Lasso Regression:** Encourages sparsity and feature selection via L1 norm ($\alpha \sum |w_i|$).

### Classification
1. **Logistic Regression:** Models class probability using the sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$.
2. **K-Nearest Neighbors (KNN):** Distance-based majority voting over optimal $K=11$ neighbors.

---

## 9. Project Architecture
```
hack_o_week aug/
├── data/
│   └── student_data.csv            # Clean raw dataset
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Data pipeline & train-test split
│   ├── regression_models.py        # 4 Regression training functions
│   ├── classification_models.py    # 2 Classification training functions
│   ├── evaluation.py               # Metrics and comparison tables
│   └── train.py                    # Training & joblib export script
├── notebooks/
│   └── model_experiments.ipynb     # Jupyter EDA & experiment notebook
├── app/
│   └── app.py                      # 5-Page Streamlit Web App
├── models/
│   ├── preprocessor.pkl            # Fitted ColumnTransformer
│   ├── linear_regression.pkl
│   ├── polynomial_regression.pkl
│   ├── ridge_regression.pkl
│   ├── lasso_regression.pkl
│   ├── logistic_regression.pkl
│   ├── knn.pkl
│   └── model_metrics.pkl           # Precomputed evaluation metrics
├── requirements.txt
└── README.md
```

---

## 10. Data Preprocessing
- **Numerical Pipeline:** `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()`.
- **Categorical Pipeline:** `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore')`.
- **Leakage Prevention:** Transformers are fitted strictly on the $80\%$ training split and applied to the $20\%$ testing split.

---

## 11. Exploratory Data Analysis (EDA)
Key findings from EDA:
- **Attendance & Study Hours** have the strongest positive linear correlation with `Exam_Score` ($r > 0.60$).
- Target `Exam_Score` follows an approximately normal distribution centered around $\mu = 67.24$, $\sigma = 3.89$.
- No duplicate records detected in the dataset.

---

## 12. Regression Methodology
1. **Split:** 80% Train (5,285), 20% Test (1,322), `random_state=42`.
2. **Hyperparameter Tuning:** 5-fold cross validation for Ridge ($\alpha \in [0.01, 0.1, 1.0, 10, 100]$) and Lasso ($\alpha \in [0.001, 0.01, 0.1, 1.0, 10]$).
3. **Evaluation Metrics:** Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and $R^2$ Score.

---

## 13. Classification Methodology
1. **Target Generation:**
   $$\text{Pass\_Fail} = \begin{cases} 1 (\text{PASS}) & \text{if } \text{Exam\_Score} \ge 65 \\ 0 (\text{FAIL}) & \text{if } \text{Exam\_Score} < 65 \end{cases}$$
2. **Stratification:** Maintained 78% Pass / 22% Fail distribution across train and test sets.
3. **KNN Tuning:** 5-fold Cross-Validation tested $K \in [3, 5, 7, 9, 11]$ (Optimal $K=11$).

---

## 14. Model Evaluation
### Regression Test Performance
| Model | MAE | MSE | RMSE | $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression** | **0.4524** | **3.2549** | **1.8041** | **0.7697** |
| **Lasso Regression** | 0.4524 | 3.2555 | 1.8043 | 0.7697 |
| **Linear Regression** | 0.4524 | 3.2560 | 1.8044 | 0.7696 |
| **Polynomial Regression** | 0.6417 | 3.5745 | 1.8906 | 0.7471 |

### Classification Test Performance
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **98.26%** | **98.18%** | **99.61%** | **0.9889** |
| **K-Nearest Neighbors ($K=11$)** | 87.44% | 86.81% | 98.93% | 0.9248 |

---

## 15. Model Comparison
- **Best Regression Model:** **Ridge Regression** achieves highest $R^2$ ($0.7697$) and lowest RMSE ($1.8041$).
- **Best Classification Model:** **Logistic Regression** delivers superior discrimination ($98.26\%$ accuracy, $0.9889$ F1) with probabilistic outputs.

---

## 16. How to Run the Project

### Step 1: Clone or Navigate to Directory
```bash
cd "c:\hack_o_week aug"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train Models & Generate Artifacts
```bash
python src/train.py
```

---

## 17. Streamlit Application Instructions
To launch the interactive 5-page web app:
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 18. Example Predictions
- **Input:** Study Hours = 20, Attendance = 85%, Previous Score = 75, Sleep = 7, Tutoring = 1.
- **Predicted Score:** **68.9 / 100**
- **Predicted Class:** **PASS ✅ (99.9% Probability)**

---

## 19. Results & Insights
- Study hours and class attendance are the primary drivers of student academic success.
- Regularization (Ridge/Lasso) prevents overfitting on high-dimensional one-hot encoded categories.
- Logistic Regression provides reliable calibrated probabilities for early student intervention.

---

## 20. Limitations
- Dataset contains self-reported survey items (e.g., motivation, study hours).
- Does not account for longitudinal mid-semester trend data.

---

## 21. Future Improvements
- Add feature importance SHAP / LIME explanations.
- Incorporate time-series tracking across semesters.
- Support batch CSV student upload and automated email alerts.

---

## 22. Team / Hack-O-Week Information
- **Event:** Hack-O-Week 5 & 6 — Machine Learning Track
- **Project:** Student Performance & Success Predictor
- **Status:** Complete, Verified, and Production-Ready
