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
    # Sélection des top règles
    top_rules = rules.head(top_n)
    
    # Création de la figure
    plt.figure(figsize=(15, 10))
    
   
    
    # 2. Top règles par lift
    plt.subplot(2, 2, 2)
    sns.barplot(data=top_rules, y='antecedents', x='lift', palette='coolwarm')
    plt.title('Top Règles par Lift')
    plt.ylabel('Produits Antécédents')
    plt.xlabel('Lift')
    
    # 3. Réseau de relations
    plt.subplot(2, 1, 2)
    for i, rule in top_rules.iterrows():
        plt.plot([0, 1], [i, i], 'k-', alpha=0.1)
        plt.text(0, i, str(rule['antecedents']), ha='right', va='center')
        plt.text(1, i, str(rule['consequents']), ha='left', va='center')
    plt.axis('off')
    plt.title('Relations entre Antécédents et Conséquents')
    
    plt.tight_layout()
    return plt