import pandas as pd 
from data_prep import load_data
import numpy as np

def top_customer(top_n) : 
    df = load_data()
    top_customers = df.groupby(['CustomerID'])['Frequency'].max().reset_index()
    top_customers = top_customers.sort_values('Frequency', ascending=False).head(top_n)
    return top_customers


def customer_segmentation():
    df = load_data()
    # Calcul du montant total dépensé par client
    customer_spending = df.groupby('CustomerID').agg(
        total_spent=('TotalPrice', 'sum'),
        country=('Country', 'first'),
        frequency=('Frequency', 'max')
    ).reset_index()
    
    # Application des segments
    conditions = [
        (customer_spending['total_spent'] >= 50000),
        (customer_spending['total_spent'] >= 10000) & (customer_spending['total_spent'] < 50000),
        (customer_spending['total_spent'] >= 5000) & (customer_spending['total_spent'] < 10000),
        (customer_spending['total_spent'] >= 1000) & (customer_spending['total_spent'] < 5000),
        (customer_spending['total_spent'] < 1000)
    ]
    
    choices = ['Diamond', 'Gold', 'Silver', 'Bronze', 'Iron']
    customer_spending['segment'] = np.select(conditions, choices)
    
    # Ordre des segments pour un tri cohérent
    segment_order = ['Diamond', 'Gold', 'Silver', 'Bronze', 'Iron']
    customer_spending['segment'] = pd.Categorical(customer_spending['segment'], 
                                                categories=segment_order, 
                                                ordered=True)
    
    return customer_spending.sort_values('total_spent', ascending=False)

def product_performance():
    df = load_data()
    top_products = df.groupby('Description').agg(
        quantity_sold=('Quantity', 'sum'),
        revenue=('TotalPrice', 'sum'),
        orders=('InvoiceNo', 'nunique')
    ).sort_values('revenue', ascending=False)
    
    return top_products.reset_index()

def top_sales_month():
    df = load_data()

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['year_month'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    
    monthly_sales = df.groupby('year_month').agg(
        total_sales=('TotalPrice', 'sum'),
        order_count=('InvoiceNo', 'nunique')
    ).reset_index()

    top_month = monthly_sales.loc[monthly_sales['total_sales'].idxmax()]
    
    return {
        'top_month': top_month['year_month'],
        'total_sales': top_month['total_sales'],
        'order_count': top_month['order_count'],
        'all_months': monthly_sales.sort_values('year_month')
    }