// This script relies on the `datasets` variable being loaded from `data.js`

document.addEventListener('DOMContentLoaded', () => {
    
    const tableHead = document.getElementById('table-head');
    const tableBody = document.getElementById('table-body');
    const refreshBtn = document.getElementById('refresh-data-btn');
    
    const opsList = document.getElementById('operations-list');
    const opsUl = document.getElementById('ops-ul');
    const viewOpsBtn = document.getElementById('view-ops-btn');
    
    const chartsGrid = document.getElementById('charts-grid');
    const viewChartsBtn = document.getElementById('view-charts-btn');
    
    const ds1Btn = document.getElementById('ds1-btn');
    const ds2Btn = document.getElementById('ds2-btn');
    const mainTitle = document.getElementById('main-title');

    let currentDatasetId = 'dataset1';

    // Function to render the top 10 dataset entries dynamically
    function renderTable() {
        tableHead.innerHTML = '';
        tableBody.innerHTML = ''; 
        
        if (typeof datasets !== 'undefined' && datasets[currentDatasetId] && datasets[currentDatasetId].data.length > 0) {
            const data = datasets[currentDatasetId].data;
            
            // Generate headers based on keys of the first row
            const keys = Object.keys(data[0]);
            const headerRow = document.createElement('tr');
            keys.forEach(key => {
                const th = document.createElement('th');
                th.textContent = key.replace(/_/g, ' ');
                headerRow.appendChild(th);
            });
            tableHead.appendChild(headerRow);
            
            // Generate body
            data.forEach(row => {
                const tr = document.createElement('tr');
                keys.forEach(key => {
                    const td = document.createElement('td');
                    let val = row[key];
                    if (val === null || val === undefined) {
                        val = 'N/A';
                    } else if (typeof val === 'number' && key.toLowerCase().includes('price')) {
                        val = '$' + val.toLocaleString();
                    } else if (typeof val === 'number') {
                        // Rounding for cleanliness on floats
                        val = val % 1 !== 0 ? val.toFixed(2) : val;
                    }
                    td.textContent = val;
                    tr.appendChild(td);
                });
                tableBody.appendChild(tr);
            });
        } else {
            tableBody.innerHTML = '<tr><td style="text-align:center;">No data available.</td></tr>';
        }
    }

    // Function to render the operations performed
    function renderOperations() {
        opsUl.innerHTML = '';
        if (typeof datasets !== 'undefined' && datasets[currentDatasetId]) {
            const operations = datasets[currentDatasetId].operations;
            operations.forEach(op => {
                const li = document.createElement('li');
                li.textContent = op;
                opsUl.appendChild(li);
            });
        } else {
            opsUl.innerHTML = '<li>No operations recorded.</li>';
        }
    }

    // Function to update images and their titles
    function updateImages() {
        if (typeof datasets !== 'undefined' && datasets[currentDatasetId]) {
            const titles = datasets[currentDatasetId].chartTitles || ["Chart 1", "Chart 2", "Chart 3", "Chart 4"];
            
            for (let i = 1; i <= 4; i++) {
                const img = document.getElementById(`chart${i}-img`);
                const title = document.getElementById(`chart${i}-title`);
                
                img.src = `images/${currentDatasetId}_chart${i}.png`;
                title.textContent = titles[i - 1];
            }
        }
    }

    // Full UI update
    function updateUI() {
        if (typeof datasets !== 'undefined' && datasets[currentDatasetId]) {
            mainTitle.textContent = datasets[currentDatasetId].name;
        }

        renderTable();
        renderOperations();
        updateImages();
        
        // Update theme
        if (currentDatasetId === 'dataset2') {
            document.body.classList.add('blue-theme');
            ds2Btn.classList.add('active-ds');
            ds1Btn.classList.remove('active-ds');
        } else {
            document.body.classList.remove('blue-theme');
            ds1Btn.classList.add('active-ds');
            ds2Btn.classList.remove('active-ds');
        }
    }

    // Auto-render on start
    updateUI();

    // Dataset Selectors
    ds1Btn.addEventListener('click', () => {
        currentDatasetId = 'dataset1';
        updateUI();
    });

    ds2Btn.addEventListener('click', () => {
        currentDatasetId = 'dataset2';
        updateUI();
    });

    // Action Buttons
    refreshBtn.addEventListener('click', () => {
        tableBody.innerHTML = '<tr><td colspan="10" style="text-align:center; padding: 20px;">Refreshing data...</td></tr>';
        setTimeout(() => {
            renderTable();
        }, 500); // simulate a slight delay
    });

    viewOpsBtn.addEventListener('click', () => {
        if (opsList.classList.contains('hidden')) {
            renderOperations();
            opsList.classList.remove('hidden');
            viewOpsBtn.textContent = 'Hide Operations';
        } else {
            opsList.classList.add('hidden');
            viewOpsBtn.textContent = 'View Operations';
        }
    });

    viewChartsBtn.addEventListener('click', () => {
        if (chartsGrid.classList.contains('hidden')) {
            chartsGrid.classList.remove('hidden');
            viewChartsBtn.textContent = 'Hide Charts';
        } else {
            chartsGrid.classList.add('hidden');
            viewChartsBtn.textContent = 'Load Charts';
        }
    });
});
