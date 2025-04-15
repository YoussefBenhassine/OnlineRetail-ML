import pandas as pd 
from data_prep import load_data
import matplotlib.pyplot as plt

def top_customer(top_n) : 
    df = load_data()
    top_customers = df.groupby(['CustomerID'])['Frequency'].max().reset_index()
    top_customers = top_customers.sort_values('Frequency', ascending=False).head(top_n)
    return top_customers

def total_spent():
    df = load_data()
    depense = df.groupby(['CustomerID']).agg(
        total_spent=('TotalPrice', 'sum'),
        nb_cmd=('InvoiceNo', 'nunique')
    ).reset_index().sort_values('total_spent', ascending=False)
    return depense