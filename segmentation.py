import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def get_rfm_data(df):
    """Extract RFM metrics from cleaned data."""
    rfm = df.groupby('CustomerID').agg({
        'Recency': 'first',
        'Frequency': 'first',
        'Monetary': 'first'
    }).reset_index()
    return rfm

def normalize_rfm(rfm_df):
    """Normalize RFM features using StandardScaler."""
    scaler = StandardScaler()
    rfm_normalized = rfm_df.copy()
    rfm_normalized[['Recency', 'Frequency', 'Monetary']] = scaler.fit_transform(
        rfm_normalized[['Recency', 'Frequency', 'Monetary']]
    )
    return rfm_normalized

def apply_kmeans(rfm_normalized, n_clusters=5):
    """Apply K-Means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(rfm_normalized[['Recency', 'Frequency', 'Monetary']])
    silhouette = silhouette_score(rfm_normalized[['Recency', 'Frequency', 'Monetary']], labels)
    return labels, silhouette

def apply_cah(rfm_normalized, n_clusters=5):
    """Apply complete hierarchical clustering."""
    cah = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = cah.fit_predict(rfm_normalized[['Recency', 'Frequency', 'Monetary']])
    silhouette = silhouette_score(rfm_normalized[['Recency', 'Frequency', 'Monetary']], labels)
    return labels, silhouette

def prepare_cah(rfm_normalized, sample_size=100):
    """Prepare data for hierarchical clustering dendrogram."""
    sampled_data = rfm_normalized.sample(sample_size)
    Z = linkage(sampled_data[['Recency', 'Frequency', 'Monetary']], method='ward')
    return Z, sampled_data['CustomerID'].astype(str).values

def find_optimal_clusters(rfm_normalized, max_clusters=10):
    """
    Determine optimal number of clusters using elbow method and silhouette analysis.
    Returns metrics for K-Means and CAH.
    """
    # Prepare data for clustering
    X = rfm_normalized[['Recency', 'Frequency', 'Monetary']]
    
    # Initialize metrics dictionaries
    metrics = {
        'cluster_range': list(range(2, max_clusters+1)),
        'kmeans': {
            'inertia': [],
            'silhouette': [],
            'calinski': [],
            'davies': []
        },
        'cah': {
            'silhouette': [],
            'calinski': [],
            'davies': []
        }
    }
    
    for n in metrics['cluster_range']:
        # K-Means
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans_labels = kmeans.fit_predict(X)
        metrics['kmeans']['inertia'].append(kmeans.inertia_)
        metrics['kmeans']['silhouette'].append(silhouette_score(X, kmeans_labels))
        metrics['kmeans']['calinski'].append(calinski_harabasz_score(X, kmeans_labels))
        metrics['kmeans']['davies'].append(davies_bouldin_score(X, kmeans_labels))
        
        # CAH
        cah_labels, _ = apply_cah(rfm_normalized, n_clusters=n)
        metrics['cah']['silhouette'].append(silhouette_score(X, cah_labels))
        metrics['cah']['calinski'].append(calinski_harabasz_score(X, cah_labels))
        metrics['cah']['davies'].append(davies_bouldin_score(X, cah_labels))
    
    return metrics


def compare_clustering_algorithms(rfm_normalized, n_clusters=5):
    """Compare K-Means and CAH clustering results."""
    # Apply both algorithms
    kmeans_labels, kmeans_silhouette = apply_kmeans(rfm_normalized, n_clusters)
    cah_labels, cah_silhouette = apply_cah(rfm_normalized, n_clusters)
    
    # Calculate comparison metrics
    X = rfm_normalized[['Recency', 'Frequency', 'Monetary']]
    comparison = pd.DataFrame({
        'Algorithm': ['K-Means', 'CAH'],
        'Silhouette': [kmeans_silhouette, cah_silhouette],
        'Calinski-Harabasz': [
            calinski_harabasz_score(X, kmeans_labels),
            calinski_harabasz_score(X, cah_labels)
        ],
        'Davies-Bouldin': [
            davies_bouldin_score(X, kmeans_labels),
            davies_bouldin_score(X, cah_labels)
        ]
    })
    
    return {
        'comparison': comparison,
        'kmeans_labels': kmeans_labels,
        'cah_labels': cah_labels
    }
    
 