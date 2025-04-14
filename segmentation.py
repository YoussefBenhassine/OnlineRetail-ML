import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage

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

def prepare_cah(rfm_normalized, sample_size=100):
    """Prepare data for hierarchical clustering."""
    sampled_data = rfm_normalized.sample(sample_size)
    Z = linkage(sampled_data[['Recency', 'Frequency', 'Monetary']], method='ward')
    return Z, sampled_data['CustomerID'].astype(str).values







"""def normalize_data():
    #Normalize numerical columns in the cleaned data using Min-Max scaling.
    #Returns a normalized DataFrame and saves it to 'OnlineRetail_normalized.csv'.
    df = load_data()
    
    # Select numerical columns to normalize
    numerical_cols = ['Quantity', 'UnitPrice', 'TotalPrice', 'Recency', 'Frequency', 'Monetary']
    
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    
    # Normalize the numerical columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Save the normalized data
    df.to_csv('OnlineRetail_normalized.csv', sep=',', index=False)
    
    st.subheader("Normalized Data Preview")
    st.write(df.head())
    
    return df"""