document.addEventListener('DOMContentLoaded', () => {
    // Add cache busting query parameter
    fetch('data.json?v=' + new Date().getTime())
        .then(response => response.json())
        .then(data => {
            populateDashboard(data);
        })
        .catch(error => {
            console.error('Error fetching data:', error);
            document.getElementById('data-prep-pill').textContent = 'Error loading data. Make sure to run math_intuitions.py first.';
        });
});

function formatArray(arr, is2D = false) {
    if (!arr) return '[]';
    if (is2D) {
        return '[\n  ' + arr.map(row => '[' + row.map(v => typeof v === 'number' ? v.toFixed(2) : v).join(', ') + ']').join(',\n  ') + '\n]';
    }
    return '[' + arr.map(v => typeof v === 'number' ? v.toFixed(3) : v).join(', ') + ']';
}

function populateDashboard(data) {
    // Header
    const prep = data.data_prep;
    document.getElementById('data-prep-pill').innerHTML = `<b>${prep.records_loaded}</b> records · Features: ${prep.features.join(', ')}`;

    // Helpers
    const setElem = (id, text) => { if(document.getElementById(id)) document.getElementById(id).textContent = text; };
    const setImg = (id, src) => { 
        if(document.getElementById(id) && src) {
            // cache bust images
            document.getElementById(id).src = src + '?v=' + new Date().getTime();
            document.getElementById(id).style.display = 'block';
        } else if (document.getElementById(id)) {
            document.getElementById(id).style.display = 'none';
        }
    };

    // Vectors
    const vec = data.vectors;
    setElem('vectors-desc', vec.description);
    setElem('vectors-purpose', vec.purpose);
    setElem('vectors-val', formatArray(vec.value));
    setImg('vectors-image', vec.image);

    // Matrices
    const mat = data.matrices;
    setElem('matrices-desc', mat.description);
    setElem('matrices-purpose', mat.purpose);
    setElem('matrices-shape', `(${mat.shape.join(', ')})`);
    setElem('matrices-val', formatArray(mat.value, true));
    setImg('matrices-image', mat.image);

    // Dot Product
    const dot = data.dot_product;
    setElem('dot-desc', dot.description);
    setElem('dot-purpose', dot.purpose);
    setElem('dot-weights', formatArray(dot.weights));
    setElem('dot-result', dot.result.toFixed(2));
    setImg('dot-image', dot.image);

    // Eigen
    const eigen = data.eigen;
    setElem('eigen-desc', eigen.description);
    setElem('eigen-purpose', eigen.purpose);
    setElem('eigen-cov', formatArray(eigen.covariance_matrix, true));
    setElem('eigen-vals', formatArray(eigen.eigenvalues));
    setElem('eigen-vecs', formatArray(eigen.eigenvectors, true));
    setImg('eigen-image', eigen.image);

    // Forward Pass
    const fw = data.forward_pass;
    setElem('fw-desc', fw.description);
    setElem('fw-purpose', fw.purpose);
    setElem('fw-preds', formatArray(fw.initial_predictions));
    setElem('fw-loss', fw.initial_loss.toFixed(4));

    // Gradients
    const grad = data.gradients;
    setElem('grad-desc', grad.description);
    setElem('grad-purpose', grad.purpose);
    setElem('grad-dw', formatArray(grad.dw));
    setElem('grad-db', grad.db.toFixed(4));

    // Gradient Descent
    const gd = data.gradient_descent;
    setElem('gd-desc', gd.description);
    setElem('gd-purpose', gd.purpose);
    setElem('gd-weights', formatArray(gd.updated_weights));
    setElem('gd-loss', gd.new_loss.toFixed(4));
    setImg('gd-image', gd.image);
    
    const diffElem = document.getElementById('gd-diff');
    if(diffElem) diffElem.textContent = `Total Decrease: ${gd.loss_decrease.toFixed(4)}`;
}
