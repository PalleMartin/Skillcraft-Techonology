# Task 2: Mall Customer Clustering

This task applies unsupervised learning techniques to segment mall customers based on their spending patterns and demographics.

## 🎯 Objective

Implement and compare multiple clustering algorithms (K-Means, Hierarchical, DBSCAN) to identify customer segments in a mall customer dataset.

## 📁 Project Structure

```
Task-2/
├── app.py              # Streamlit web application
├── kmeans_backend.py   # Clustering algorithms implementation
├── Mall_Customers.csv  # Sample dataset
└── README.md           # This file
```

## 🚀 Features

- **Multiple Clustering Algorithms**:
  - K-Means Clustering
  - Hierarchical Clustering
  - DBSCAN Clustering
- **Optimal Cluster Detection**: Elbow method and Silhouette analysis
- **Dimensionality Reduction**: PCA (Principal Component Analysis)
- **Comprehensive Evaluation**: Multiple clustering metrics
- **Interactive Visualization**: 2D and 3D cluster plots
- **Data Export**: Download clustered data as CSV

## 📦 Installation

1. Navigate to the Task-2 directory:
   ```bash
   cd Task-2
   ```

2. Install the required dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```

## ▶️ Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. In the web interface:
   - Upload a mall customer dataset (or use the default)
   - Select a clustering algorithm
   - Adjust algorithm parameters
   - View optimal cluster analysis
   - Analyze clustering results and metrics
   - Export results

## 🧠 Implementation Details

### Clustering Algorithms

1. **K-Means**:
   - Partitions data into K clusters
   - Minimizes within-cluster sum of squares
   - Best for spherical clusters of similar size

2. **Hierarchical Clustering**:
   - Builds a hierarchy of clusters
   - Agglomerative approach (bottom-up)
   - Good for discovering nested clusters

3. **DBSCAN**:
   - Density-based clustering
   - Identifies outliers and noise points
   - Excellent for clusters of varying shapes

### Evaluation Metrics

- **Silhouette Score**: Measures how similar objects are to their own cluster compared to other clusters
- **Calinski-Harabasz Index**: Ratio of between-cluster dispersion to within-cluster dispersion
- **Davies-Bouldin Index**: Average similarity ratio of each cluster with its most similar cluster

### Optimal Cluster Detection

- **Elbow Method**: Plots WCSS (Within-Cluster Sum of Squares) against the number of clusters
- **Silhouette Analysis**: Plots average silhouette scores for different numbers of clusters

## 📊 Visualization

- **2D Cluster Plots**: Scatter plots showing cluster assignments
- **3D Cluster Plots**: Three-dimensional visualization when applicable
- **Evaluation Charts**: Elbow method and silhouette analysis plots
- **Cluster Statistics**: Summary statistics for each cluster

## 🛠️ Technical Requirements

- Python 3.7+
- Scikit-learn for machine learning algorithms
- Streamlit for web interface
- NumPy for numerical operations
- Matplotlib and Seaborn for visualization

## 📝 Notes

- The application uses Annual Income and Spending Score for clustering by default
- Scaling is recommended for algorithms sensitive to feature magnitudes
- PCA can be applied for dimensionality reduction and better visualization
- The default dataset contains 200 customer records

## 📧 Contact

For issues or questions, please contact the development team at SkillCraft Technology.