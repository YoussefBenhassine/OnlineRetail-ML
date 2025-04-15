import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from data_prep import preprocess_data, statistique_descriptive, visualize_data, quantity_outliers, unitPrice_outliers, load_data
from segmentation import get_rfm_data, normalize_rfm, apply_kmeans, prepare_cah, apply_cah, find_optimal_clusters, compare_clustering_algorithms
from scipy.cluster.hierarchy import dendrogram
import numpy as np  
from association_rules import apply_apriori, plot_association_rules


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

    # Section 1: Application des algorithmes de clustering
    st.header("1. Application des algorithmes de clustering (K-means et CAH)")
    
    # Sous-section K-Means
    st.subheader("K-Means Clustering")
    n_clusters = st.slider("Nombre de clusters", 2, 10, 5, key="kmeans")
    
    if st.button("Appliquer K-Means"):
        labels, silhouette = apply_kmeans(rfm_normalized, n_clusters)
        st.success(f"Score Silhouette: `{silhouette:.2f}`")
        
        # Affichage des caractéristiques des clusters
        rfm['Cluster'] = labels
        st.write("Caractéristiques moyennes par cluster:")
        st.write(rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean())
        
        # Visualisation des clusters
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        sns.scatterplot(x='Recency', y='Frequency', hue=labels, palette='viridis', data=rfm_normalized, ax=axes[0])
        sns.scatterplot(x='Frequency', y='Monetary', hue=labels, palette='viridis', data=rfm_normalized, ax=axes[1])
        sns.scatterplot(x='Recency', y='Monetary', hue=labels, palette='viridis', data=rfm_normalized, ax=axes[2])
        st.pyplot(fig)

    # Sous-section CAH
    st.subheader("Hierarchical Clustering (CAH)")
    sample_size = st.slider("Taille de l'échantillon", 50, 500, 100, key="cah")
    
    if st.button("Appliquer CAH"):
        # CAH complète pour tous les points
        cah_labels, cah_silhouette = apply_cah(rfm_normalized, n_clusters)
        st.success(f"Score Silhouette: `{cah_silhouette:.2f}`")
        
        # Dendrogramme avec échantillon
        Z, customer_labels = prepare_cah(rfm_normalized, sample_size)
        plt.figure(figsize=(12, 6))
        dendrogram(Z, orientation='top', labels=customer_labels, distance_sort='descending')
        plt.title("Dendrogramme (CAH)")
        st.pyplot(plt)
        
        # Affichage des caractéristiques des clusters CAH
        rfm['Cluster_CAH'] = cah_labels
        st.write("Caractéristiques moyennes par cluster (CAH):")
        st.write(rfm.groupby('Cluster_CAH')[['Recency', 'Frequency', 'Monetary']].mean())

    # Section 2: Détermination du nombre optimal de clusters
    st.header("2. Détermination du nombre optimal de clusters")
    if st.button("Analyser le nombre optimal de clusters"):
        metrics = find_optimal_clusters(rfm_normalized)
        
        # Graphique de la méthode Elbow
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        ax1.plot(metrics['cluster_range'], metrics['kmeans']['inertia'], 'bo-')
        ax1.set_title('Méthode Elbow (K-Means)')
        ax1.set_xlabel('Nombre de clusters')
        ax1.set_ylabel('Inertie')
        
        # Graphique des scores de silhouette
        ax2.plot(metrics['cluster_range'], metrics['kmeans']['silhouette'], 'bo-', label='K-Means')
        ax2.plot(metrics['cluster_range'], metrics['cah']['silhouette'], 'go-', label='CAH')
        ax2.set_title('Scores de Silhouette')
        ax2.set_xlabel('Nombre de clusters')
        ax2.set_ylabel('Score Silhouette')
        ax2.legend()
        
        st.pyplot(fig1)
        
        # Affichage des valeurs numériques
        st.write("Valeurs optimales:")
        optimal_kmeans = metrics['cluster_range'][np.argmax(metrics['kmeans']['silhouette'])]
        optimal_cah = metrics['cluster_range'][np.argmax(metrics['cah']['silhouette'])]
        st.write(f"- K-Means: nombre optimal de clusters = {optimal_kmeans} (silhouette max)")
        st.write(f"- CAH: nombre optimal de clusters = {optimal_cah} (silhouette max)")

    # Section 3: Comparaison des approches de clustering
    st.header("3. Comparaison des approches de clustering")
    if st.button("Comparer K-Means et CAH"):
        comparison_result = compare_clustering_algorithms(rfm_normalized, n_clusters)
        
        st.subheader("Métriques de comparaison")
        st.dataframe(comparison_result['comparison'].style.highlight_max(axis=0))
        
        st.subheader("Visualisation des clusters")
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # K-Means
        sns.scatterplot(x='Frequency', y='Monetary', hue=comparison_result['kmeans_labels'], 
                        palette='viridis', data=rfm_normalized, ax=axes[0])
        axes[0].set_title('K-Means Clustering')
        
        # CAH
        sns.scatterplot(x='Frequency', y='Monetary', hue=comparison_result['cah_labels'], 
                        palette='viridis', data=rfm_normalized, ax=axes[1])
        axes[1].set_title('Hierarchical Clustering (CAH)')
        
        st.pyplot(fig)
        
        st.subheader("Interprétation")
        st.markdown("""
        - **K-Means**: Généralement plus rapide, mieux adapté aux grands jeux de données
        - **CAH**: Donne une hiérarchie des clusters, permet de voir les relations entre groupes
        - Choix dépend des métriques et de l'objectif business
        """)


def association_analysis():
    st.title("Analyse des règles d'association")
    df = load_data()
    
    st.header("1. Paramètres de l'analyse")
    col1, col2 = st.columns(2)
    with col1:
        min_support = st.slider("Support minimum", 0.01, 0.1, 0.05, step=0.01)
    with col2:
        min_confidence = st.slider("Confiance minimum", 0.1, 1.0, 0.5, step=0.1)
    
    if st.button("Générer les règles d'association"):
        with st.spinner('Calcul en cours...'):
            rules = apply_apriori(df, min_support, min_confidence)
            
            if rules.empty:
                st.warning("Aucune règle trouvée avec ces paramètres. Essayez de réduire le support ou la confiance minimum.")
            else:
                st.success(f"{len(rules)} règles trouvées!")
                
                
                # Tableau interactif
                st.header("3. Tableau détaillé des règles")
                st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                           .sort_values('lift', ascending=False)
                           .style.background_gradient(cmap='Blues'))
                
                # Top produits
                st.header("4. Produits les plus fréquents")
                top_products = df['Description'].value_counts().head(10)
                fig, ax = plt.subplots()
                sns.barplot(x=top_products.values, y=top_products.index, ax=ax)
                plt.title('Top 10 des produits les plus fréquents')
                st.pyplot(fig)

# Ajouter dans la navigation sidebar
selected_page = st.sidebar.radio(
    "Navigation",
    options=[
        "Préparation et exploration des données",
        "Segmentation client",
        "Analyse des règles d'association"
    ]
)



# Page routing
if selected_page == "Préparation et exploration des données":
    preparation_exploration()
elif selected_page == "Segmentation client":
    segmentation()
elif selected_page == "Analyse des règles d'association":
    association_analysis()