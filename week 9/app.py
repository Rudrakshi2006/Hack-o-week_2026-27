from flask import Flask, render_template, request, jsonify
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import pandas as pd
import numpy as np

app = Flask(__name__)

# Global state to store the initial dataset with missing values
INITIAL_DF = None
TARGET = None

def get_initial_data():
    global INITIAL_DF, TARGET
    if INITIAL_DF is None:
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        TARGET = data.target
        
        # Inject some random missing values (approx 5% overall) into the first 5 columns to demonstrate imputation
        np.random.seed(42)
        for col in df.columns[:5]:
            mask = np.random.rand(len(df)) < 0.05
            df.loc[mask, col] = np.nan
            
        # Keep only the first 10 columns for simplicity in visualization + target
        df = df.iloc[:, :10]
        df['target'] = TARGET
        INITIAL_DF = df
    return INITIAL_DF.copy()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dataset', methods=['GET'])
def dataset():
    df = get_initial_data()
    
    # Calculate dataset stats
    total_rows = len(df)
    total_features = len(df.columns) - 1 # excluding target
    missing_vals = int(df.isnull().sum().sum())
    
    # Prepare data for frontend table (first 100 rows for performance)
    records = df.head(100).replace({np.nan: None}).to_dict(orient='records')
    
    # Missing values per column
    missing_by_col = df.isnull().sum().to_dict()
    
    return jsonify({
        'total_rows': total_rows,
        'total_features': total_features,
        'missing_values': missing_vals,
        'missing_by_col': missing_by_col,
        'records': records,
        'columns': df.columns.tolist()
    })

@app.route('/api/impute', methods=['POST'])
def impute():
    req = request.json
    strategy = req.get('strategy', 'mean')
    df = get_initial_data()
    
    if strategy == 'mean':
        df = df.fillna(df.mean())
    elif strategy == 'median':
        df = df.fillna(df.median())
    elif strategy == 'mode':
        df = df.fillna(df.mode().iloc[0])
    elif strategy == 'drop':
        df = df.dropna()
        
    missing_vals = int(df.isnull().sum().sum())
    missing_by_col = df.isnull().sum().to_dict()
    records = df.head(100).replace({np.nan: None}).to_dict(orient='records')
    
    return jsonify({
        'missing_values': missing_vals,
        'missing_by_col': missing_by_col,
        'records': records,
        'total_rows': len(df)
    })

@app.route('/api/engineer', methods=['POST'])
def engineer():
    req = request.json
    # Assume we receive imputed data from frontend, but for simplicity, we'll re-impute here using mean
    df = get_initial_data()
    df = df.fillna(df.mean())
    
    feature_name = req.get('feature_type', 'ratio')
    
    original_cols = len(df.columns) - 1
    
    if feature_name == 'ratio':
        df['radius_texture_ratio'] = df['mean radius'] / df['mean texture']
    elif feature_name == 'area_approx':
        df['area_approx'] = 3.14159 * (df['mean radius'] ** 2)
        
    new_cols = len(df.columns) - 1
    
    # Send back distribution of new feature
    new_feature = df.columns[-2] # Before target
    distribution = df[new_feature].tolist()
    
    return jsonify({
        'original_count': original_cols,
        'new_count': new_cols,
        'new_feature_name': new_feature,
        'distribution': distribution,
        'records': df.head(100).replace({np.nan: None}).to_dict(orient='records'),
        'columns': df.columns.tolist()
    })

@app.route('/api/scale', methods=['POST'])
def scale():
    req = request.json
    strategy = req.get('strategy', 'standard')
    df = get_initial_data()
    df = df.fillna(df.mean())
    
    # Just take 'mean radius' for visualization
    feature = 'mean radius'
    original_dist = df[feature].tolist()
    
    X = df[[feature]]
    if strategy == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
        
    scaled_dist = scaler.fit_transform(X).flatten().tolist()
    
    return jsonify({
        'feature': feature,
        'original': original_dist,
        'scaled': scaled_dist
    })

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    req = request.json
    train_size = req.get('train_size', 0.8)
    threshold = req.get('threshold', 0.5)
    k_folds = req.get('k_folds', 5)
    
    df = get_initial_data()
    df = df.fillna(df.mean())
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Scale all
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=train_size, random_state=42)
    
    # 2. Cross Validation (on whole dataset for demo purposes)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    cv_scores = []
    for train_index, val_index in kf.split(X_scaled):
        X_tr, X_val = X_scaled[train_index], X_scaled[val_index]
        y_tr, y_val = y.iloc[train_index], y.iloc[val_index]
        model_cv = LogisticRegression(max_iter=1000)
        model_cv.fit(X_tr, y_tr)
        cv_scores.append(model_cv.score(X_val, y_val))
        
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    
    # 3. Model Training on Train/Test split
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Get probabilities
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Apply threshold
    y_pred = (y_probs >= threshold).astype(int)
    
    # Metrics
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    # The cm is [[TN, FP], [FN, TP]] for labels [0,1]. Let's format it.
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # ROC
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    return jsonify({
        'split': {
            'train_size': len(X_train),
            'test_size': len(X_test)
        },
        'cv': {
            'scores': cv_scores,
            'mean': cv_mean,
            'std': cv_std
        },
        'metrics': {
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'roc_auc': float(roc_auc)
        },
        'roc': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
