import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page setup
st.set_page_config(page_title="Customer Segmentation", page_icon="🎯", layout="wide")



# Train K-Means model on realistic clustered synthetic data
@st.cache_resource
def run_clustering(n_clusters):
    np.random.seed(42)
    n_per_cluster = 120

    # 1. Low Income, Low Spending ("Frugal")
    inc1 = np.random.normal(25, 6, n_per_cluster)
    sp1 = np.random.normal(20, 8, n_per_cluster)

    # 2. Low Income, High Spending ("Careless Spenders")
    inc2 = np.random.normal(25, 6, n_per_cluster)
    sp2 = np.random.normal(80, 8, n_per_cluster)

    # 3. High Income, Low Spending ("Miserly Savers")
    inc3 = np.random.normal(90, 12, n_per_cluster)
    sp3 = np.random.normal(20, 8, n_per_cluster)

    # 4. High Income, High Spending ("VIP Spenders")
    inc4 = np.random.normal(90, 12, n_per_cluster)
    sp4 = np.random.normal(82, 8, n_per_cluster)

    # 5. Average Income, Average Spending ("Middle Class")
    inc5 = np.random.normal(55, 10, n_per_cluster)
    sp5 = np.random.normal(50, 8, n_per_cluster)

    income = np.concatenate([inc1, inc2, inc3, inc4, inc5])
    spending = np.concatenate([sp1, sp2, sp3, sp4, sp5])
    age = np.random.randint(18, 70, len(income))


    # Keep within realistic bounds
    income = np.clip(income, 15, 140)
    spending = np.clip(spending, 1, 99)

    df = pd.DataFrame({
        'Age': age,
        'Annual Income ($k)': income,
        'Spending Score (1-100)': spending
    })

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[['Annual Income ($k)', 'Spending Score (1-100)']])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Segment'] = kmeans.fit_predict(scaled_data)
    df['Segment'] = 'Cluster ' + df['Segment'].astype(str)

    return df


# Title
st.title("🎯 Customer Segmentation App")
st.markdown("Group target customers into actionable personas using **K-Means Clustering**.")

# Sidebar Settings
st.sidebar.header("Segmentation Controls")
n_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=6, value=5)

df = run_clustering(n_clusters)

# Visualizations Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Customer Persona Clusters")
    fig = px.scatter(
        df,
        x='Annual Income ($k)',
        y='Spending Score (1-100)',
        color='Segment',
        hover_data=['Age'],
        template='plotly_white',
        title=f"K-Means Clustering Analysis (K={n_clusters})"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Cluster Averages")
    summary = df.groupby('Segment')[['Annual Income ($k)', 'Spending Score (1-100)', 'Age']].mean().reset_index()
    st.dataframe(
        summary.style.format({
            'Annual Income ($k)': '${:.1f}k',
            'Spending Score (1-100)': '{:.1f}',
            'Age': '{:.0f}'
        }),
        use_container_width=True
    )