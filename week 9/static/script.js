document.addEventListener('DOMContentLoaded', () => {
    // === Navigation ===
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.content-section');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            navLinks.forEach(l => l.classList.remove('active'));
            sections.forEach(s => s.classList.remove('active'));
            
            link.classList.add('active');
            const targetId = link.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
            
            // Re-render charts when their section becomes active to fix resize issues
            if (targetId === 'section-2' && charts.missing) charts.missing.update();
            if (targetId === 'section-3' && charts.engineer) charts.engineer.update();
            if (targetId === 'section-4' && charts.scale) charts.scale.update();
            if (targetId === 'section-5' && charts.split) charts.split.update();
            if (targetId === 'section-6' && charts.cv) charts.cv.update();
            if (targetId === 'section-8' && charts.metrics) charts.metrics.update();
            if (targetId === 'section-9' && charts.roc) charts.roc.update();
        });
    });

    // === Global Chart Instances ===
    const charts = {
        missing: null,
        engineer: null,
        scale: null,
        split: null,
        cv: null,
        metrics: null,
        roc: null
    };

    // Chart default colors
    const colors = {
        primary: '#2E7D32',
        secondary: '#4CAF50',
        accent: '#81C784',
        red: '#e53935',
        blue: '#1e88e5',
        gray: '#e0e0e0'
    };

    // === Utility: Fetch API ===
    async function apiCall(endpoint, method = 'GET', body = null) {
        const options = {
            method,
            headers: { 'Content-Type': 'application/json' }
        };
        if (body) options.body = JSON.stringify(body);
        
        try {
            const response = await fetch(`/api/${endpoint}`, options);
            return await response.json();
        } catch (error) {
            console.error(`Error fetching ${endpoint}:`, error);
            return null;
        }
    }

    // === Initialize Data ===
    async function init() {
        const data = await apiCall('dataset');
        if (data) {
            updateOverview(data);
            updateMissingSection(data);
            
            // Trigger subsequent initializations
            await handleEngineer('ratio');
            await handleScale('standard');
            await handleEvaluate();
        }
    }

    // === SECTION 1: Overview ===
    function updateOverview(data) {
        document.getElementById('stat-rows').innerText = data.total_rows;
        document.getElementById('stat-cols').innerText = data.total_features;
        document.getElementById('stat-missing').innerText = data.missing_values;

        // Render Table Header
        const head = document.getElementById('dataset-head');
        head.innerHTML = data.columns.map(c => `<th>${c}</th>`).join('');

        // Render Table Body
        const body = document.getElementById('dataset-body');
        body.innerHTML = data.records.map(row => {
            return `<tr>` + data.columns.map(col => {
                const val = row[col];
                const isMissing = val === null || val === undefined || Number.isNaN(val);
                return `<td class="${isMissing ? 'missing' : ''}">${isMissing ? 'NaN' : (typeof val === 'number' ? val.toFixed(4) : val)}</td>`;
            }).join('') + `</tr>`;
        }).join('');
    }

    // === SECTION 2: Missing Data ===
    function updateMissingSection(data) {
        const labels = Object.keys(data.missing_by_col);
        const values = Object.values(data.missing_by_col);

        const ctx = document.getElementById('missing-chart').getContext('2d');
        if (charts.missing) charts.missing.destroy();

        charts.missing = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Missing Values Count',
                    data: values,
                    backgroundColor: values.map(v => v > 0 ? colors.red : colors.primary)
                }]
            },
            options: { responsive: true, scales: { y: { beginAtZero: true } } }
        });
    }

    document.querySelectorAll('.impute-btn').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            document.querySelectorAll('.impute-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            
            const strategy = e.target.getAttribute('data-strategy');
            const data = await apiCall('impute', 'POST', { strategy });
            if (data) {
                updateMissingSection(data);
                document.getElementById('stat-missing').innerText = data.missing_values;
                document.getElementById('stat-rows').innerText = data.total_rows;
                document.getElementById('impute-result').innerHTML = `<strong>Result:</strong> Applied ${strategy} imputation. Remaining missing values: ${data.missing_values}. Total rows: ${data.total_rows}.`;
                
                // Re-evaluate model with imputed data implicitly handled by backend for simplicity
                handleEvaluate();
            }
        });
    });

    // === SECTION 3: Feature Engineering ===
    async function handleEngineer(type) {
        const data = await apiCall('engineer', 'POST', { feature_type: type });
        if (data) {
            document.getElementById('eng-orig-count').innerText = data.original_count;
            document.getElementById('eng-new-count').innerText = data.new_count;

            const ctx = document.getElementById('engineer-chart').getContext('2d');
            if (charts.engineer) charts.engineer.destroy();

            // Create a histogram-like chart for the new feature
            const sorted = data.distribution.sort((a,b)=>a-b);
            
            charts.engineer = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: sorted.map((_, i) => i),
                    datasets: [{
                        label: data.new_feature_name,
                        data: sorted,
                        borderColor: colors.accent,
                        backgroundColor: colors.light_green,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: { 
                    responsive: true,
                    plugins: { legend: { position: 'top' } },
                    scales: { x: { display: false } }
                }
            });
        }
    }

    document.querySelectorAll('.engineer-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.engineer-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            handleEngineer(e.target.getAttribute('data-type'));
        });
    });

    // === SECTION 4: Scaling ===
    async function handleScale(strategy) {
        const data = await apiCall('scale', 'POST', { strategy });
        if (data) {
            const ctx = document.getElementById('scale-chart').getContext('2d');
            if (charts.scale) charts.scale.destroy();

            charts.scale = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.original.map((_, i) => i),
                    datasets: [
                        {
                            label: 'Original (unscaled)',
                            data: data.original,
                            borderColor: colors.gray,
                            tension: 0.4,
                            yAxisID: 'y'
                        },
                        {
                            label: `Scaled (${strategy})`,
                            data: data.scaled,
                            borderColor: colors.primary,
                            tension: 0.4,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: { display: false },
                        y: { type: 'linear', display: true, position: 'left' },
                        y1: { type: 'linear', display: true, position: 'right', grid: { drawOnChartArea: false } }
                    }
                }
            });

            document.getElementById('scale-explanation').innerText = strategy === 'standard' 
                ? "Standardization scales data to have a mean of 0 and std deviation of 1." 
                : "Min-Max scales data into a fixed range, usually 0 to 1.";
        }
    }

    document.querySelectorAll('.scale-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.scale-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            handleScale(e.target.getAttribute('data-strategy'));
        });
    });

    // === EVALUATION PIPELINE (Sections 5, 6, 7, 8, 9, 10) ===
    
    // State
    let currentState = {
        train_size: 0.8,
        threshold: 0.5,
        k_folds: 5
    };

    async function handleEvaluate() {
        const data = await apiCall('evaluate', 'POST', currentState);
        if (data) {
            updateSplitSection(data.split);
            updateCVSection(data.cv);
            updateMetricsSection(data.metrics);
            updateROCSection(data.roc, data.metrics.roc_auc);
            updateFinalDashboard(data.metrics, data.cv);
        }
    }

    // Handlers for sliders and buttons
    const splitSlider = document.getElementById('split-slider');
    splitSlider.addEventListener('input', (e) => {
        document.getElementById('split-val').innerText = `${e.target.value}%`;
    });
    splitSlider.addEventListener('change', (e) => {
        currentState.train_size = e.target.value / 100;
        handleEvaluate();
    });

    const threshSlider = document.getElementById('thresh-slider');
    threshSlider.addEventListener('input', (e) => {
        document.getElementById('thresh-val').innerText = parseFloat(e.target.value).toFixed(2);
    });
    threshSlider.addEventListener('change', (e) => {
        currentState.threshold = parseFloat(e.target.value);
        handleEvaluate();
    });

    document.querySelectorAll('.cv-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.cv-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            currentState.k_folds = parseInt(e.target.getAttribute('data-k'));
            handleEvaluate();
        });
    });

    // Updaters
    function updateSplitSection(splitData) {
        document.getElementById('train-samples').innerText = splitData.train_size;
        document.getElementById('test-samples').innerText = splitData.test_size;

        const ctx = document.getElementById('split-chart').getContext('2d');
        if (charts.split) charts.split.destroy();

        charts.split = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Train', 'Test'],
                datasets: [{
                    data: [splitData.train_size, splitData.test_size],
                    backgroundColor: [colors.primary, colors.accent]
                }]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });
    }

    function updateCVSection(cvData) {
        document.getElementById('cv-mean').innerText = (cvData.mean * 100).toFixed(2) + '%';
        document.getElementById('cv-std').innerText = (cvData.std * 100).toFixed(2) + '%';

        const ctx = document.getElementById('cv-chart').getContext('2d');
        if (charts.cv) charts.cv.destroy();

        charts.cv = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: cvData.scores.map((_, i) => `Fold ${i+1}`),
                datasets: [{
                    label: 'Accuracy',
                    data: cvData.scores.map(s => s * 100),
                    backgroundColor: colors.secondary
                }]
            },
            options: { 
                responsive: true, 
                scales: { y: { min: 0, max: 100 } }
            }
        });
    }

    function updateMetricsSection(metrics) {
        // Confusion Matrix
        document.getElementById('cell-tn').innerText = metrics.tn;
        document.getElementById('cell-fp').innerText = metrics.fp;
        document.getElementById('cell-fn').innerText = metrics.fn;
        document.getElementById('cell-tp').innerText = metrics.tp;

        // Metric Cards
        document.getElementById('metric-precision').innerText = (metrics.precision * 100).toFixed(1) + '%';
        document.getElementById('metric-recall').innerText = (metrics.recall * 100).toFixed(1) + '%';
        document.getElementById('metric-f1').innerText = (metrics.f1 * 100).toFixed(1) + '%';

        // Metrics Chart
        const ctx = document.getElementById('metrics-chart').getContext('2d');
        if (charts.metrics) charts.metrics.destroy();

        charts.metrics = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                datasets: [{
                    label: 'Score %',
                    data: [metrics.accuracy*100, metrics.precision*100, metrics.recall*100, metrics.f1*100],
                    backgroundColor: [colors.primary, colors.secondary, colors.accent, colors.blue]
                }]
            },
            options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0, max: 100 } } }
        });
    }

    function updateROCSection(roc, aucVal) {
        document.getElementById('metric-auc').innerText = aucVal.toFixed(4);

        const ctx = document.getElementById('roc-chart').getContext('2d');
        if (charts.roc) charts.roc.destroy();

        charts.roc = new Chart(ctx, {
            type: 'line',
            data: {
                labels: roc.fpr,
                datasets: [
                    {
                        label: 'ROC Curve',
                        data: roc.tpr,
                        borderColor: colors.primary,
                        backgroundColor: 'rgba(46, 125, 50, 0.1)',
                        fill: true,
                        tension: 0.1
                    },
                    {
                        label: 'Random Guess',
                        data: roc.fpr, // y = x
                        borderColor: colors.gray,
                        borderDash: [5, 5],
                        fill: false,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: { type: 'linear', title: { display: true, text: 'False Positive Rate' } },
                    y: { type: 'linear', title: { display: true, text: 'True Positive Rate' } }
                }
            }
        });
    }

    function updateFinalDashboard(metrics, cv) {
        document.getElementById('fin-acc').innerText = (metrics.accuracy * 100).toFixed(2) + '%';
        document.getElementById('fin-prec').innerText = (metrics.precision * 100).toFixed(2) + '%';
        document.getElementById('fin-rec').innerText = (metrics.recall * 100).toFixed(2) + '%';
        document.getElementById('fin-f1').innerText = (metrics.f1 * 100).toFixed(2) + '%';
        document.getElementById('fin-auc').innerText = metrics.roc_auc.toFixed(4);
        document.getElementById('fin-cv').innerText = (cv.mean * 100).toFixed(2) + '% ± ' + (cv.std * 100).toFixed(2) + '%';
    }

    // Confusion matrix click explanations
    const explanations = {
        'cell-tp': 'True Positive: The model correctly predicted a malignant tumor. This is a successful detection.',
        'cell-tn': 'True Negative: The model correctly predicted a benign tumor. This is a successful normal result.',
        'cell-fp': 'False Positive (Type I Error): The model predicted malignant, but it was actually benign. This causes unnecessary stress and further testing for the patient.',
        'cell-fn': 'False Negative (Type II Error): The model predicted benign, but it was actually malignant. This is highly dangerous as a cancer goes undetected.'
    };

    document.querySelectorAll('.matrix-cell').forEach(cell => {
        cell.addEventListener('click', (e) => {
            const id = e.target.id;
            document.getElementById('matrix-explanation').innerText = explanations[id];
        });
    });

    // Reset button
    document.getElementById('reset-btn').addEventListener('click', () => {
        location.reload();
    });

    // Start
    init();
});
