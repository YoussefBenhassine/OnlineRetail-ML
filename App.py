import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from data_prep import preprocess_data, statistique_descriptive, visualize_data, quantity_outliers, unitPrice_outliers, load_data
from segmentation import get_rfm_data, normalize_rfm, apply_kmeans, prepare_cah
from scipy.cluster.hierarchy import dendrogram

st.set_page_config(layout="wide")
st.title("Online Retail Data Analysis")

def preparation_exploration(): 
    df = preprocess_data()
    st.subheader("Peak")
    st.write(df.head())
    statistique_descriptive()
    visualize_data()
    unitPrice_outliers()
    quantity_outliers()


def segmentation():
    st.title("RFM Customer Segmentation")
    df = load_data()
    rfm = get_rfm_data(df)
    rfm_normalized = normalize_rfm(rfm)

    # K-Means Section
    st.subheader("K-Means Clustering")
    n_clusters = st.slider("Number of clusters", 2, 10, 5, key="kmeans")
    
    if st.button("Run K-Means"):
        labels, silhouette = apply_kmeans(rfm_normalized, n_clusters)
        st.success(f"Silhouette Score: `{silhouette:.2f}`")
        
        # Display cluster averages
        rfm['Cluster'] = labels
        st.write("Cluster Averages:")
        st.write(rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean())
        
        # Plot clusters
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        sns.scatterplot(x='Recency', y='Frequency', hue=labels, palette='viridis', data=rfm_normalized, ax=axes[0])
        sns.scatterplot(x='Frequency', y='Monetary', hue=labels, palette='viridis', data=rfm_normalized, ax=axes[1])
        st.pyplot(fig)

    # CAH Section
    st.subheader("Hierarchical Clustering")
    sample_size = st.slider("Sample size", 50, 500, 100, key="cah")
    
    if st.button("Run CAH"):
        Z, customer_labels = prepare_cah(rfm_normalized, sample_size)
        plt.figure(figsize=(12, 6))
        dendrogram(Z, orientation='top', labels=customer_labels, distance_sort='descending')
        plt.title("Dendrogram (CAH)")
        st.pyplot(plt)








# Create sidebar navigation
selected_page = st.sidebar.radio(
    "Navigation",
    options=[
        "Préparation et exploration des données",
        "Segmentation client"
    ]
)

# Page routing
if selected_page == "Préparation et exploration des données":
    preparation_exploration()
elif selected_page == "Segmentation client":
    segmentation()