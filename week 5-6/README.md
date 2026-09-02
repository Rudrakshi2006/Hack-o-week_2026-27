### Person 1: Data Preparation, Vectors, and Matrices
- **Data Preparation**: Handled the initial loading and standardization of the Iris dataset's 4D feature space (Sepal/Petal lengths and widths) to ensure all variables are on the same scale. Created a binary classification target to distinguish Setosa from other species.
- **Vectors**: Explained the intuition of data points as vectors in a 4D space. Computed and plotted the L2 norms (magnitudes) to demonstrate how mathematically "extreme" or unusual each flower is compared to the dataset's average.
- **Matrices**: Generated the Covariance Matrix to show multi-dimensional relationships. Visualized how different features correlate with one another (e.g., whether Petal Length and Width increase together or move oppositely).

### Person 2: Dot Product and Eigenvalues (Linear Algebra)
- **Dot Product**: Demonstrated the dot product's role in measuring similarity and computing weighted sums. Calculated the Gram Matrix to visualize the pairwise geometric similarity between different flowers.
- **Eigenvalues & Eigenvectors**: Explained the foundations of Principal Component Analysis (PCA). Extracted eigenvectors from the covariance matrix to find the "main directions" hidden in the data, and used eigenvalues (visualized via a Scree Plot) to show the proportion of variance captured by each principal component.

### Person 3: Calculus (Gradient Descent) and Dashboard Integration
- **Calculus & Optimization**: Implemented the mathematical foundation of machine learning using Gradient Descent. Built a forward pass with a Sigmoid activation function, calculated the Binary Cross-Entropy Loss, and used derivatives (the chain rule) to compute gradients. Ran 50 epochs of gradient descent to demonstrate loss minimization.
- **Dashboard Development**: Exported the calculated metrics and Matplotlib/Seaborn visual plots into JSON format. Developed an interactive local web dashboard (HTML/CSS/JS) to dynamically present these mathematical intuitions in a visual format.
