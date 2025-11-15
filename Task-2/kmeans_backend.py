# kmeans_backend.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_data(file=None):
    """Load data from uploaded file or default dataset"""
    if file is not None:
        # If file is uploaded
        data = pd.read_csv(file)
    else:
        # Load default dataset
        data = pd.read_csv('Mall_Customers.csv')
    return data

def preprocess_data(data, scale=True):
    """Preprocess the data for clustering"""
    # Select features for clustering (Annual Income and Spending Score)
    X = data.iloc[:, [3, 4]].values  # Assuming Annual Income is column 3 and Spending Score is column 4
    
    # Scale the data if requested
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X

def find_optimal_clusters(X, max_clusters=10):
    """Find optimal number of clusters using elbow method"""
    wcss = []
    silhouette_scores = []
    
    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        
        # Calculate silhouette score
        labels = kmeans.labels_
        sil_score = silhouette_score(X, labels)
        silhouette_scores.append(sil_score)
    
    return wcss, silhouette_scores

def apply_kmeans(X, n_clusters=5):
    """Apply K-Means clustering algorithm"""
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    return y_kmeans, centers, kmeans

def apply_hierarchical(X, n_clusters=5):
    """Apply Hierarchical clustering algorithm"""
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    y_hierarchical = hierarchical.fit_predict(X)
    return y_hierarchical

def apply_dbscan(X, eps=0.5, min_samples=5):
    """Apply DBSCAN clustering algorithm"""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    y_dbscan = dbscan.fit_predict(X)
    return y_dbscan

def evaluate_clustering(X, labels):
    """Evaluate clustering performance using multiple metrics"""
    if len(set(labels)) > 1 and -1 not in set(labels):
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        davies = davies_bouldin_score(X, labels)
        return silhouette, calinski, davies
    else:
        return None, None, None

def perform_pca(X, n_components=2):
    """Perform PCA for dimensionality reduction"""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

def get_cluster_statistics(data, labels):
    """Calculate statistics for each cluster"""
    # Add cluster labels to data
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = labels
    
    # Calculate statistics
    stats = data_with_clusters.groupby('Cluster').agg({
        data.columns[3]: ['mean', 'std'],
        data.columns[4]: ['mean', 'std']
    }).round(2)
    
    return stats
