import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

def apply_apriori(df, min_support=0.05, min_confidence=0.5):
    # Préparation des données
    basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum()
              .unstack()
              .reset_index()
              .fillna(0)
              .set_index('InvoiceNo'))
    
    # Encodage one-hot
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    # Itemsets fréquents
    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
    
    # Règles d'association
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(by=['lift', 'confidence'], ascending=False)
    
    return rules

def plot_association_rules(rules, top_n=10):
    """Génère des visualisations pour les règles d'association"""
    from mpl_toolkits.mplot3d import Axes3D  # nécessaire pour les graphes 3D
    top_rules = rules.head(top_n)
    
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Graphique 3D
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    sc = ax1.scatter(top_rules['support'], top_rules['confidence'], top_rules['lift'], 
                     c=top_rules['lift'], cmap='viridis', s=100)
    ax1.set_xlabel('Support')
    ax1.set_ylabel('Confidence')
    ax1.set_zlabel('Lift')
    ax1.set_title('Relation entre Support, Confiance et Lift')
    fig.colorbar(sc, ax=ax1, label='Lift')
    
    # 2. Top règles par lift
    ax2 = fig.add_subplot(2, 2, 2)
    sns.barplot(data=top_rules, y='antecedents', x='lift', palette='coolwarm', ax=ax2)
    ax2.set_title(f'Top {top_n} Règles par Lift')
    ax2.set_ylabel('Produits Antécédents')
    ax2.set_xlabel('Lift')
    
    # 3. Nuage de points
    ax3 = fig.add_subplot(2, 2, 3)
    sns.scatterplot(data=top_rules, x='support', y='confidence', size='lift', 
                    hue='lift', palette='viridis', sizes=(50, 300), ax=ax3)
    ax3.set_title('Support vs Confidence (taille par Lift)')
    
    # 4. Relations texte
    ax4 = fig.add_subplot(2, 2, 4)
    for i, rule in top_rules.iterrows():
        antecedents = ', '.join(list(rule['antecedents']))
        consequents = ', '.join(list(rule['consequents']))
        ax4.plot([0, 1], [i, i], 'k-', alpha=0.1)
        ax4.text(0, i, antecedents, ha='right', va='center', fontsize=8)
        ax4.text(1, i, consequents, ha='left', va='center', fontsize=8)
    ax4.axis('off')
    ax4.set_title('Relations entre Antécédents et Conséquents')
    
    fig.tight_layout()
    return fig
