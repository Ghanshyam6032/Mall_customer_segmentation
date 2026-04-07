import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

def app():
    st.title("🧠 Mall Customer Segmentation (DBSCAN)")

    # ==============================
    # Load Dataset
    # ==============================
    df = pd.read_csv("Mall_Customers.csv")
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    st.write("### 📂 Data Sample")
    st.dataframe(X.head())

    # ==============================
    # Sidebar Parameters
    # ==============================
    st.sidebar.header("⚙️ DBSCAN Parameters")
    eps = st.sidebar.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1)
    min_samples = st.sidebar.slider("Min Samples", 2, 20, 5)

    # ==============================
    # Scaling
    # ==============================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ==============================
    # DBSCAN Model
    # ==============================
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    # ==============================
    # Plot Clusters
    # ==============================
    st.write("### 📊 Cluster Visualization")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        X['Annual Income (k$)'],
        X['Spending Score (1-100)'],
        c=labels,
        cmap='rainbow'
    )

    # Highlight outliers
    outliers = X[labels == -1]
    ax.scatter(
        outliers['Annual Income (k$)'],
        outliers['Spending Score (1-100)'],
        color='black',
        label='Outliers',
        s=80
    )

    # ==============================
    # User Input
    # ==============================
    st.write("### 👤 Enter Customer Data")

    income = st.number_input("Annual Income (k$)", 0, 200, 50)
    score = st.number_input("Spending Score (1-100)", 1, 100, 50)

    # ==============================
    # Prediction (Indentation Fixed)
    # ==============================
    if st.button("Predict Customer Category"):
        new_point = np.array([[income, score]])

        # Scale input
        new_scaled = scaler.transform(new_point)

        distances = pairwise_distances(new_scaled, X_scaled)
        neighbors = np.where(distances[0] <= eps)[0]

        if len(neighbors) >= min_samples:
            cluster = labels[neighbors[0]]

            if cluster == -1:
                st.error("⚠️ Unusual Customer (Outlier)")
            else:
                # Category logic
                if income > 70 and score > 70:
                    name = "💎 Premium Customer"
                elif income > 70 and score <= 40:
                    name = "🧊 Conservative Customer"
                elif income <= 40 and score > 70:
                    name = "🎯 Target Customer"
                elif income <= 40 and score <= 40:
                    name = "🪙 Budget Customer"
                else:
                    name = "🔄 Average Customer"

                st.success(f"✅ Category: {name}")

                # Plot user point
                ax.scatter(income, score, color='red', s=200, marker='X', label='User')

        else:
            st.error("⚠️ Unusual Customer (Outlier)")

    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("DBSCAN Clustering")
    ax.legend()

    st.pyplot(fig)

    # ==============================
    # Summary
    # ==============================
    st.write("### 📈 Summary")

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = list(labels).count(-1)

    st.write(f"✅ Total Clusters: {n_clusters}")
    st.write(f"❌ Outliers: {n_outliers}")

# Run app
if __name__ == "__main__":
    app()