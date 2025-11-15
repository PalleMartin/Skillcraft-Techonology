# app.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from kmeans_backend import (
    load_data, 
    preprocess_data, 
    find_optimal_clusters, 
    apply_kmeans, 
    apply_hierarchical, 
    apply_dbscan,
    evaluate_clustering,
    perform_pca,
    get_cluster_statistics
)

# Configure page settings
st.set_page_config(
    page_title="Advanced Mall Customer Clustering", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("🛍️ Advanced Mall Customer Segmentation")
st.markdown("### SkillCraft Technology - Enhanced Clustering Analysis")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# --- Data Upload Section ---
st.sidebar.subheader("📁 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Mall Customer Dataset (.csv)", type=["csv"])

# --- Clustering Algorithm Selection ---
st.sidebar.subheader("🧮 Clustering Algorithm")
algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["K-Means", "Hierarchical", "DBSCAN"]
)

# --- Preprocessing Options ---
st.sidebar.subheader("🔧 Preprocessing")
scale_data = st.sidebar.checkbox("Scale Data", value=True)
pca_analysis = st.sidebar.checkbox("Apply PCA", value=False)

# --- Algorithm Parameters ---
st.sidebar.subheader("🎛️ Parameters")
if algorithm == "K-Means":
    k = st.sidebar.slider("Number of Clusters (K)", 2, 15, 5)
elif algorithm == "Hierarchical":
    k = st.sidebar.slider("Number of Clusters", 2, 15, 5)
else:  # DBSCAN
    eps = st.sidebar.slider("Epsilon", 0.1, 2.0, 0.5, 0.1)
    min_samples = st.sidebar.slider("Min Samples", 1, 20, 5)

# --- Load and Display Data ---
try:
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.success("✅ Dataset uploaded successfully!")
    else:
        data = load_data()
        st.info("ℹ️ Using default Mall Customers dataset")
    
    # Display dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(data))
    with col2:
        st.metric("Features", data.shape[1]-1)  # Excluding ID
    with col3:
        st.metric("Data Size", f"{data.memory_usage(deep=True).sum()/1024:.1f} KB")
    
    # Dataset preview
    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True)
    
    # Dataset statistics
    st.subheader("📈 Dataset Statistics")
    st.dataframe(data.describe(), use_container_width=True)
    
    # --- Preprocess Data ---
    X = preprocess_data(data, scale=scale_data)
    
    # Apply PCA if selected
    if pca_analysis:
        X, pca = perform_pca(X)
        st.sidebar.info(f"📊 PCA Explained Variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # --- Elbow Method and Silhouette Analysis ---
    st.subheader("🔍 Optimal Cluster Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Elbow Method")
        wcss, silhouette_scores = find_optimal_clusters(X)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(2, len(wcss) + 2), wcss, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
        ax.set_title("Elbow Method for Optimal K")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.markdown("#### Silhouette Analysis")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='s', linewidth=2, markersize=8, color='green')
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Score Analysis")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # --- Apply Clustering Algorithm ---
    st.subheader(f"🎯 {algorithm} Clustering Results")
    
    if algorithm == "K-Means":
        y_labels, centers, model = apply_kmeans(X, n_clusters=k)
    elif algorithm == "Hierarchical":
        y_labels = apply_hierarchical(X, n_clusters=k)
        centers = None
    else:  # DBSCAN
        y_labels = apply_dbscan(X, eps=eps, min_samples=min_samples)
        centers = None
    
    # --- Evaluate Clustering ---
    silhouette, calinski, davies = evaluate_clustering(X, y_labels)
    
    # Display evaluation metrics
    if silhouette is not None:
        st.markdown("### 📊 Clustering Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Silhouette Score", f"{silhouette:.3f}")
        with col2:
            st.metric("Calinski-Harabasz", f"{calinski:.1f}")
        with col3:
            st.metric("Davies-Bouldin", f"{davies:.3f}")
    
    # --- Visualize Clusters ---
    st.subheader("🧠 Cluster Visualization")
    
    # Create a larger figure for better visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors for clusters
    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'gray', 'pink',
              'olive', 'lime', 'navy', 'maroon', 'teal']
    
    # Plot clusters
    unique_labels = np.unique(y_labels)
    for i, label in enumerate(unique_labels):
        if label == -1:  # Noise points in DBSCAN
            ax.scatter(X[y_labels == label, 0], X[y_labels == label, 1],
                      s=100, c='black', marker='x', label='Noise', alpha=0.7)
        else:
            ax.scatter(X[y_labels == label, 0], X[y_labels == label, 1],
                      s=100, c=colors[i % len(colors)], label=f'Cluster {label}', alpha=0.7)
    
    # Plot centroids for K-Means
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1],
                  s=300, c='yellow', marker='*', edgecolors='black', linewidth=1, label='Centroids')
    
    ax.set_title(f"Customer Segmentation using {algorithm}")
    ax.set_xlabel("Feature 1" if not pca_analysis else "Principal Component 1")
    ax.set_ylabel("Feature 2" if not pca_analysis else "Principal Component 2")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # --- Cluster Statistics ---
    st.subheader("📋 Cluster Statistics")
    try:
        cluster_stats = get_cluster_statistics(data, y_labels)
        st.dataframe(cluster_stats, use_container_width=True)
    except Exception as e:
        st.warning("Could not calculate cluster statistics for this algorithm.")
    
    # --- 3D Visualization (if applicable) ---
    if X.shape[1] >= 3:
        st.subheader("🌐 3D Cluster Visualization")
        try:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            for i, label in enumerate(unique_labels):
                if label == -1:  # Noise points in DBSCAN
                    ax.scatter(X[y_labels == label, 0], X[y_labels == label, 1], X[y_labels == label, 2],
                              s=100, c='black', marker='x', label='Noise', alpha=0.7)
                else:
                    ax.scatter(X[y_labels == label, 0], X[y_labels == label, 1], X[y_labels == label, 2],
                              s=100, c=colors[i % len(colors)], label=f'Cluster {label}', alpha=0.7)
            
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_zlabel("Feature 3")
            ax.set_title(f"3D {algorithm} Clustering")
            ax.legend()
            
            st.pyplot(fig)
        except:
            st.info("3D visualization not available for this dataset.")
    
    # --- Download Results ---
    st.subheader("💾 Export Results")
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = y_labels
    
    csv = data_with_clusters.to_csv(index=False)
    st.download_button(
        label="Download Clustered Data as CSV",
        data=csv,
        file_name="clustered_mall_customers.csv",
        mime="text/csv"
    )
    
    st.success("✅ Clustering analysis completed successfully!")
    
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please make sure your dataset has the correct format with columns for CustomerID, Gender, Age, Annual Income, and Spending Score.")

# Footer
st.markdown("---")
st.markdown("### 📝 Notes")
st.markdown("""
- **K-Means**: Best for spherical clusters of similar size
- **Hierarchical**: Good for discovering nested clusters
- **DBSCAN**: Excellent for identifying outliers and clusters of varying shapes
- **Scaling**: Recommended for algorithms sensitive to feature magnitudes
- **PCA**: Useful for dimensionality reduction and visualization
""")
