<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Model Evaluation & Feature Engineering</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
</head>
<body>
    <div class="dashboard-container">
        <!-- Sidebar -->
        <aside class="sidebar">
            <div class="sidebar-header">
                <h2>ML Pipeline</h2>
            </div>
            <nav class="sidebar-nav">
                <a href="#section-1" class="nav-link active" data-target="section-1">1. Dataset Overview</a>
                <a href="#section-2" class="nav-link" data-target="section-2">2. Missing Data</a>
                <a href="#section-3" class="nav-link" data-target="section-3">3. Feature Engineering</a>
                <a href="#section-4" class="nav-link" data-target="section-4">4. Feature Scaling</a>
                <a href="#section-5" class="nav-link" data-target="section-5">5. Train/Test Split</a>
                <a href="#section-6" class="nav-link" data-target="section-6">6. Cross-Validation</a>
                <a href="#section-7" class="nav-link" data-target="section-7">7. Confusion Matrix</a>
                <a href="#section-8" class="nav-link" data-target="section-8">8. Metrics (P/R/F1)</a>
                <a href="#section-9" class="nav-link" data-target="section-9">9. ROC-AUC</a>
                <a href="#section-10" class="nav-link" data-target="section-10">10. Final Dashboard</a>
            </nav>
            <div class="sidebar-footer">
                <button id="reset-btn" class="btn btn-outline">Reset Pipeline</button>
            </div>
        </aside>

        <!-- Main Content -->
        <main class="main-content">
            <header class="top-header">
                <h1>Machine Learning Model Evaluation & Feature Engineering Visualizer</h1>
                <p>Interactive Academic Dashboard - Breast Cancer Wisconsin Dataset</p>
            </header>

            <div class="sections-wrapper">
                
                <!-- SECTION 1 -->
                <section id="section-1" class="content-section active">
                    <h2>Dataset Overview</h2>
                    <div class="info-panel">
                        <strong>What is this?</strong> A real-world dataset representing breast cancer tumors, categorized as malignant or benign.<br>
                        <strong>Why is it important?</strong> Real-world datasets often contain incomplete or inconsistent data. Before training a model, the data must be cleaned and transformed.
                    </div>
                    <div class="stats-cards" id="overview-stats">
                        <div class="card"><h3>Total Rows</h3><span id="stat-rows">-</span></div>
                        <div class="card"><h3>Total Features</h3><span id="stat-cols">-</span></div>
                        <div class="card"><h3>Missing Values</h3><span id="stat-missing">-</span></div>
                    </div>
                    <div class="card mt-20">
                        <h3>Dataset Preview (First 100 rows)</h3>
                        <div class="table-container">
                            <table id="dataset-table">
                                <thead><tr id="dataset-head"></tr></thead>
                                <tbody id="dataset-body"></tbody>
                            </table>
                        </div>
                    </div>
                </section>

                <!-- SECTION 2 -->
                <section id="section-2" class="content-section">
                    <h2>Handling Missing Data</h2>
                    <div class="info-panel">
                        <strong>What is this?</strong> Techniques to handle missing values (NaNs) in the dataset, such as imputation (filling with mean/median/mode) or dropping rows.<br>
                        <strong>Why is it important?</strong> Most machine learning algorithms cannot handle missing values and will throw an error.
                    </div>
                    <div class="grid-2">
                        <div class="card">
                            <h3>Missing Values per Feature</h3>
                            <canvas id="missing-chart"></canvas>
                        </div>
                        <div class="card">
                            <h3>Imputation Strategy</h3>
                            <p>Select a method to handle the missing values:</p>
                            <div class="controls">
                                <button class="btn impute-btn" data-strategy="mean">Mean Imputation</button>
                                <button class="btn impute-btn" data-strategy="median">Median Imputation</button>
                                <button class="btn impute-btn" data-strategy="mode">Mode Imputation</button>
                                <button class="btn impute-btn" data-strategy="drop">Drop Rows</button>
                            </div>
                            <div id="impute-result" class="mt-20"></div>
                        </div>
                    </div>
                </section>

                <!-- SECTION 3 -->
                <section id="section-3" class="content-section">
                    <h2>Feature Engineering</h2>
                    <div class="info-panel">
                        <strong>What is this?</strong> Creating new features from existing ones to help the model learn better patterns.<br>
                        <strong>Why is it important?</strong> Feature engineering transforms existing variables into more useful representations that can improve predictive performance.
                    </div>
                    <div class="grid-2">
                        <div class="card">
                            <h3>Create New Feature</h3>
                            <p>Let's create a derived feature:</p>
                            <div class="controls">
                                <button class="btn engineer-btn" data-type="ratio">Create Ratio (Radius/Texture)</button>
                                <button class="btn engineer-btn" data-type="area_approx">Create Area Approx (pi * R²)</button>
                            </div>
                            <div id="engineer-result" class="mt-20">
                                <p>Original Features: <span id="eng-orig-count">-</span></p>
                                <p>New Features: <span id="eng-new-count">-</span></p>
                            </div>
                        </div>
                        <div class="card">
                            <h3>New Feature Distribution</h3>
                            <canvas id="engineer-chart"></canvas>
                        </div>
                    </div>
                </section>

                <!-- SECTION 4 -->
                <section id="section-4" class="content-section">
                    <h2>Feature Scaling</h2>
                    <div class="info-panel">
                        <strong>What is this?</strong> Transforming numeric features to be on a similar scale.<br>
                        <strong>Why is it important?</strong> Algorithms like Logistic Regression or SVM perform poorly if features have vastly different scales (e.g., Age 0-100 vs Income 0-100,000).
                    </div>
                    <div class="grid-2">
                        <div class="card">
                            <h3>Scaling Method</h3>
                            <div class="controls">
                                <button class="btn scale-btn active" data-strategy="standard">Standardization (Z-Score)</button>
                                <button class="btn scale-btn" data-strategy="minmax">Min-Max Normalization</button>
                            </div>
                            <p class="mt-20" id="scale-explanation">Standardization scales data to have a mean of 0 and std deviation of 1.</p>
                        </div>
                        <div class="card">
                            <h3>Feature Distribution (Mean Radius)</h3>
                            <canvas id="scale-chart"></canvas>
                        </div>
                    </div>
                </section>

                <!-- SECTION 5 -->
                <section id="section-5" class="content-section">
                    <h2>Train/Test Split</h2>
                    <div class="info-panel">
                        <strong>What is this?</strong> Dividing the dataset into a training set and a testing set.<br>
                        <strong>Why is it important?</strong> The training set is used to learn patterns, while the test set is used to evaluate how well the trained model performs on unseen data.
                    </div>
                    <div class="grid-2">
                        <div class="card">
                            <h3>Split Control</h3>
                            <label for="split-slider">Training Data Percentage: <span id="split-val">80%</span></label>
                            <input type="range" id="split-slider" min="50" max="90" step="5" value="80" class="slider">
                            <div class="split-stats mt-20">
                                <p>Training Samples: <span id="train-samples">-</span></p>
                                <p>Testing Samples: <span id="test-samples">-</span></p>
                            </div>
                        </div>
                        <div class="card">
                            <h3>Dataset Division</h3>
                            <canvas id="split-chart"></canvas>
                        </div>
                    </div>
                </section>

                <!-- SECTION 6 -->
                <section id="section-6" class="content-section">
                    <h2>Cross-Validation</h2>
                    <div class="info-panel">
                        <strong>What is this?</strong> K-Fold Cross-Validation splits data into K equal folds, training on K-1 and validating on the remaining fold, repeating K times.<br>
                        <strong>Why is it important?</strong> Gives a more reliable estimate of model performance than relying on a single train/test split.
                    </div>
                    <div class="grid-2">
                        <div class="card">
                            <h3>Select K Folds</h3>
                            <div class="controls">
                                <button class="btn cv-btn" data-k="3">3-Fold</button>
                                <button class="btn cv-btn active" data-k="5">5-Fold</button>
                                <button class="btn cv-btn" data-k="10">10-Fold</button>
                            </div>
                            <div class="mt-20">
                                <h3>Results</h3>
                                <p>Mean CV Accuracy: <strong id="cv-mean">-</strong></p>
                                <p>Standard Deviation: <strong id="cv-std">-</strong></p>
                            </div>
                        </div>
                        <div class="card">
                            <h3>Fold Accuracies</h3>
                            <canvas id="cv-chart"></canvas>
                        </div>
                    </div>
                </section>

                <!-- SECTION 7 -->
                <section id="section-7" class="content-section">
                    <h2>Confusion Matrix</h2>
                    <div class="info-panel">
                        <strong>What is this?</strong> A 2x2 table showing the count of correct and incorrect predictions made by the model.<br>
                        <strong>Why is it important?</strong> Accuracy alone can be misleading. The confusion matrix shows exactly where the model is making errors (False Positives vs False Negatives).
                    </div>
                    <div class="card matrix-card">
                        <table class="confusion-matrix">
                            <tr>
                                <td></td>
                                <th>Predicted Benign (0)</th>
                                <th>Predicted Malignant (1)</th>
                            </tr>
                            <tr>
                                <th>Actual Benign (0)</th>
                                <td id="cell-tn" class="matrix-cell tn" title="True Negative: Predicted 0, Actual 0">-</td>
                                <td id="cell-fp" class="matrix-cell fp" title="False Positive: Predicted 1, Actual 0">-</td>
                            </tr>
                            <tr>
                                <th>Actual Malignant (1)</th>
                                <td id="cell-fn" class="matrix-cell fn" title="False Negative: Predicted 0, Actual 1">-</td>
                                <td id="cell-tp" class="matrix-cell tp" title="True Positive: Predicted 1, Actual 1">-</td>
                            </tr>
                        </table>
                        <div id="matrix-explanation" class="mt-20 text-center">Click a cell for explanation.</div>
                    </div>
                </section>

                <!-- SECTION 8 -->
                <section id="section-8" class="content-section">
                    <h2>Precision, Recall & F1</h2>
                    <div class="info-panel">
                        <strong>What is this?</strong> Advanced metrics derived from the Confusion Matrix.<br>
                        <strong>Why is it important?</strong> In medical datasets (like Breast Cancer), a False Negative (missing a tumor) is much worse than a False Positive. Recall helps us optimize for this.
                    </div>
                    <div class="stats-cards grid-3">
                        <div class="card metric-card">
                            <h3>Precision</h3>
                            <div class="metric-val" id="metric-precision">-</div>
                            <p class="formula">TP / (TP + FP)</p>
                            <p class="desc">Out of all predicted positive, how many were actually positive?</p>
                        </div>
                        <div class="card metric-card">
                            <h3>Recall</h3>
                            <div class="metric-val" id="metric-recall">-</div>
                            <p class="formula">TP / (TP + FN)</p>
                            <p class="desc">Out of all actual positives, how many did we predict correctly?</p>
                        </div>
                        <div class="card metric-card">
                            <h3>F1 Score</h3>
                            <div class="metric-val" id="metric-f1">-</div>
                            <p class="formula">Harmonic Mean</p>
                            <p class="desc">Balance between Precision and Recall.</p>
                        </div>
                    </div>
                    <div class="card mt-20">
                        <canvas id="metrics-chart" style="max-height: 250px;"></canvas>
                    </div>
                </section>

                <!-- SECTION 9 -->
                <section id="section-9" class="content-section">
                    <h2>ROC Curve & ROC-AUC</h2>
                    <div class="info-panel">
                        <strong>What is this?</strong> Receiver Operating Characteristic curve. It plots True Positive Rate vs False Positive Rate at different classification thresholds.<br>
                        <strong>Why is it important?</strong> AUC (Area Under Curve) summarizes model performance across all thresholds. AUC = 1 is perfect, AUC = 0.5 is random guessing.
                    </div>
                    <div class="grid-2">
                        <div class="card text-center">
                            <h3>Classification Threshold</h3>
                            <label for="thresh-slider">Threshold: <span id="thresh-val">0.50</span></label>
                            <input type="range" id="thresh-slider" min="0" max="1" step="0.05" value="0.5" class="slider w-100 mt-20">
                            <h2 class="mt-20">ROC-AUC</h2>
                            <div class="huge-metric" id="metric-auc">-</div>
                            <p class="mt-20 text-muted">Changing the threshold impacts the Confusion Matrix and metrics, but the ROC-AUC score remains constant as it evaluates all thresholds.</p>
                        </div>
                        <div class="card">
                            <canvas id="roc-chart"></canvas>
                        </div>
                    </div>
                </section>

                <!-- SECTION 10 -->
                <section id="section-10" class="content-section">
                    <h2>Final Evaluation Dashboard</h2>
                    <div class="card final-pipeline-card mb-20 text-center">
                        <strong>Visual Pipeline:</strong> Data → Clean → Engineer → Scale → Split → Validate → Evaluate
                    </div>
                    <div class="grid-2">
                        <div class="card">
                            <h3>Final Performance Summary</h3>
                            <ul class="summary-list">
                                <li><strong>Accuracy:</strong> <span id="fin-acc">-</span></li>
                                <li><strong>Precision:</strong> <span id="fin-prec">-</span></li>
                                <li><strong>Recall:</strong> <span id="fin-rec">-</span></li>
                                <li><strong>F1 Score:</strong> <span id="fin-f1">-</span></li>
                                <li><strong>ROC-AUC:</strong> <span id="fin-auc">-</span></li>
                                <li><strong>CV Mean:</strong> <span id="fin-cv">-</span></li>
                            </ul>
                        </div>
                        <div class="card">
                            <h3>About this Project</h3>
                            <p>This project demonstrates fundamental machine-learning preprocessing and evaluation techniques interactively.</p>
                            <p>Powered by Vanilla JS, Chart.js, Flask, and Scikit-Learn using the Breast Cancer Wisconsin dataset.</p>
                        </div>
                    </div>
                </section>

            </div>
        </main>
    </div>
    <script src="{{ url_for('static', filename='script.js') }}"></script>
</body>
</html>
