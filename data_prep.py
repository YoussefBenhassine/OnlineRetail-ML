import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    df = pd.read_csv('OnlineRetail_cleaned.csv', encoding='unicode_escape', sep=',')
    return df

def preprocess_data(): 
    df = pd.read_csv('OnlineRetail.csv', encoding='unicode_escape', sep=',')
    df = df.dropna()
    df = df.drop_duplicates()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = (df['Quantity'] * df['UnitPrice']).round(2)
    df['CustomerID'] = df['CustomerID'].astype('Int64')
    df['Country'] = df['Country'].replace({
        'EIRE': 'Ireland',
        'Israel': 'Palestine',
        'Unspecified': 'United Kingdom',
        'RSA': 'South Africa',
    })
    df = df[(df['UnitPrice'] > 0) & (df['UnitPrice'] <= 100)]
    df = df[(df['Quantity'] > 0) & (df['Quantity'] <= 1500)]
    df = df[(df['TotalPrice'] > 0) & (df['TotalPrice'] <= 7000)]
    df = df[(df['InvoiceDate'] >= '2010-12-01') & (df['InvoiceDate'] <= '2011-12-09')]
    df = df[df['Country'] == "United Kingdom"]
    rfm_df = rfm(df)
    df = df.merge(rfm_df, on='CustomerID', how='left')
    df.to_csv('OnlineRetail_cleaned.csv', sep=',', index=False)
    return df

def statistique_descriptive():
    df = load_data()
    st.subheader("Statistiques descriptives")
    st.write(df.describe())

def visualize_data(): 
    df = load_data()
    st.subheader("Nombre de commandes par pays")
    top_countries = df['Country'].value_counts().nlargest(15)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.countplot(x='Country', data=df, order=top_countries.index, ax=ax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def unitPrice_outliers():
    df = load_data()
    st.subheader("Outliers Unit Price")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(x=df['UnitPrice'], ax=ax)
    plt.title('Boxplot of Unit Price')
    st.pyplot(fig)

def quantity_outliers():
    df = load_data()
    st.subheader("Outliers Quantity")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(x=df['Quantity'], ax=ax)
    plt.title('Boxplot of Quantity')
    st.pyplot(fig)

def rfm(df):

    rfm_df = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalPrice': 'Monetary'
    })
    return rfm_df

