import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from data_prep import preprocess_data, statistique_descriptive, visualize_data, quantity_outliers, unitPrice_outliers, load_data
from segmentation import get_rfm_data, normalize_rfm, apply_kmeans, prepare_cah, apply_cah, find_optimal_clusters, compare_clustering_algorithms
from scipy.cluster.hierarchy import dendrogram
import numpy as np  
from association_rules import apply_apriori, plot_association_rules
from analysis import top_customer, customer_segmentation, product_performance
import plotly.express as px


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
    col1, col2, col3 = st.columns(3)
    with col1:
        min_support = st.slider("Support minimum", 0.01, 0.1, 0.02, step=0.01, 
                               help="Fréquence minimale des itemsets dans le dataset")
    with col2:
        min_confidence = st.slider("Confiance minimum", 0.1, 1.0, 0.5, step=0.1,
                                 help="Probabilité minimale que le conséquent soit acheté si l'antécédent l'est")
    with col3:
        top_n = st.slider("Nombre de règles à afficher", 5, 50, 10)
    
    if st.button("Générer les règles d'association"):
        with st.spinner('Calcul en cours...'):
            rules = apply_apriori(df, min_support, min_confidence)
            
            if rules.empty:
                st.warning("Aucune règle trouvée avec ces paramètres. Essayez de réduire le support ou la confiance minimum.")
            else:
                st.success(f"{len(rules)} règles trouvées!")
                
               
                
                
                # Section 2: Top règles
                st.header("2. Meilleures règles par métrique")
                

                tab1, tab2, tab3 = st.tabs(["Par Lift", "Par Confiance", "Par Support"])
                
                with tab1:
                    st.subheader(f"Top {top_n} règles par Lift")
                    top_lift = rules.sort_values('lift', ascending=False).head(top_n)
                    st.dataframe(top_lift[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
                    
                    
                with tab2:
                    st.subheader(f"Top {top_n} règles par Confiance")
                    top_conf = rules.sort_values('confidence', ascending=False).head(top_n)
                    st.dataframe(top_conf[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
                
                with tab3:
                    st.subheader(f"Top {top_n} règles par Support")
                    top_supp = rules.sort_values('support', ascending=False).head(top_n)
                    st.dataframe(top_supp[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
                
                # Section 3: Produits fréquents
                st.header("3. Analyse des produits")
                
                # Top produits
                st.subheader("Produits les plus fréquents")
                top_products = df['Description'].value_counts().head(10) #############################################""
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                sns.barplot(x=top_products.values, y=top_products.index, ax=ax2, palette='Blues_d')
                plt.title('Top 10 des produits les plus fréquents')
                st.pyplot(fig2)
                
                # Matrice de co-occurrence
                st.subheader("Produits fréquemment achetés ensemble")
                product_pairs = rules[['antecedents', 'consequents']].apply(lambda x: f"{x[0]} → {x[1]}", axis=1)
                pair_counts = product_pairs.value_counts().head(10)
                
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                sns.barplot(x=pair_counts.values, y=pair_counts.index, ax=ax3, palette='Greens_d')
                plt.title('Top 10 des associations de produits')
                plt.xlabel('Nombre d\'occurrences')
                st.pyplot(fig3)
                
                
                st.header("4. Tableau complet des règles")
                st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                           .sort_values('lift', ascending=False)
                           .style.background_gradient(subset=['support', 'confidence', 'lift'], cmap='Blues')
                           .format({'support': '{:.3f}', 'confidence': '{:.3f}', 'lift': '{:.3f}'}))

def analysis():
    st.title("Advanced Customer Analytics Dashboard")
    
    tab1, tab2 = st.tabs([
        "Customer Segmentation",
        "Product Performance"
    ])
    
    with tab1:
        st.subheader('Customer Segmentation by Total Spending')
        segments = customer_segmentation()
            
        # Segment statistics
        st.markdown("Segment Statistics")
        segment_stats = segments.groupby('segment').agg(
            customer_count=('CustomerID', 'count'),
            avg_spending=('total_spent', 'mean'),
            total_spending=('total_spent', 'sum')
        ).reset_index()
            
        st.dataframe(segment_stats.style.format({
            'avg_spending': '€{:,.2f}',
            'total_spending': '€{:,.2f}'
        }))
            
        col1, col2 = st.columns(2)
            
        with col1:
            st.markdown("### Segment Distribution")
            fig = px.pie(segments, names='segment', 
                    title='Customer Segments Distribution')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("### Spending by Segment")
            fig = px.box(segments, x='segment', y='total_spent',
                        title='Total Spending Distribution by Segment',
                        labels={'total_spent': 'Total Spent (€)'})
            st.plotly_chart(fig, use_container_width=True)
            
        # Customers by segment
        st.markdown("Customers by Segment")
        selected_segment = st.selectbox(
            "Select segment to view:",
            ['All'] + sorted(segments['segment'].unique()),
            key='segment_select'
        )
            
        if selected_segment == 'All':
            display_df = segments
        else:
            display_df = segments[segments['segment'] == selected_segment]
            
        st.dataframe(display_df.sort_values('total_spent', ascending=False))
    
    
    with tab2:
        st.subheader("Product Performance Analysis")
        products = product_performance()
        
        # Top products selector
        top_n = st.slider("Number of top products to display", 5, 50, 10, key='product_slider')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"Top {top_n} Products by Revenue")
            fig = px.bar(products.head(top_n), x='revenue', y='Description',
                        orientation='h', color='revenue',
                        labels={'revenue': 'Revenue (€)'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f"Top {top_n} Products by Quantity Sold")
            fig = px.bar(products.head(top_n), x='quantity_sold', y='Description',
                        orientation='h', color='quantity_sold',
                        labels={'quantity_sold': 'Quantity Sold'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Product details table
        st.markdown("Product Performance Details")
        st.dataframe(products.head(top_n).style.format({
            'revenue': '€{:,.2f}',
            'quantity_sold': '{:,}'
        }))
    


selected_page = st.sidebar.radio(
    "Navigation",
    options=[
        "Préparation et exploration des données",
        "Segmentation client",
        "Analyse des règles d'association",
        "Dashboard"
    ]
)


if selected_page == "Préparation et exploration des données":
    preparation_exploration()
elif selected_page == "Segmentation client":
    segmentation()
elif selected_page == "Analyse des règles d'association":
    association_analysis()
elif selected_page == "Dashboard":
    analysis()