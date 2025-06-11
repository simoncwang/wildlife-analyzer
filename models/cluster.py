from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
import sys
import mlflow
from sklearn.metrics import silhouette_score
from collections import Counter

# Dynamically add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from pipeline.utils import load_config
from pipeline.feature_engineering import engineer_features, load_clean_data

def plot_clusters(df, labels, max_labels_per_cluster=5):
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df["lon"], df["lat"], c=labels, cmap="tab10", alpha=0.6)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Wildlife Clusters with Species Labels")
    plt.grid(True)
    plt.colorbar(scatter, label="Cluster")

    # Add labels for a few species per cluster
    df["cluster"] = labels
    for cluster in df["cluster"].unique():
        cluster_df = df[df["cluster"] == cluster]
        samples = cluster_df.sample(min(max_labels_per_cluster, len(cluster_df)))
        for _, row in samples.iterrows():
            plt.text(row["lon"], row["lat"], row["species"], fontsize=8, alpha=0.8)

    plt.show()

def plot_clusters_interactive(df, labels):
    df = df.copy()
    df["cluster"] = labels

    fig = px.scatter(
        df,
        x="lon",
        y="lat",
        color="cluster",
        hover_name="species",   # this will show species name when you hover
        hover_data=["observer", "date", "place_guess"],  # more fields if desired
        title="Wildlife Clusters (Interactive)",
        labels={"lon": "Longitude", "lat": "Latitude"},
        color_continuous_scale="Viridis"  # for numeric cluster labels
    )

    fig.update_traces(marker=dict(size=8, opacity=0.7), selector=dict(mode='markers'))
    fig.update_layout(legend_title_text="Cluster")
    fig.show()

def cluster_data(features, df, n_clusters):
    n_samples = features.shape[0]
    if n_samples < n_clusters:
        n_clusters = n_samples  # Adjust to number of samples if fewer than clusters
    
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(features)

    df["cluster"] = labels
    
    #  with mlflow.start_run(run_name="kmeans_clustering", nested=True):
        # model = KMeans(n_clusters=n_clusters, random_state=42)
        # labels = model.fit_predict(features)

        # df["cluster"] = labels

        # mlflow.log_param("n_clusters", n_clusters)
        # mlflow.log_metric("inertia", model.inertia_)

        # # Silhouette Score
        # try:
        #     sil_score = silhouette_score(features, labels)
        #     mlflow.log_metric("silhouette_score", sil_score)
        # except:
        #     pass  # Only works if >1 cluster and samples

        # Cluster counts
        # cluster_counts = {int(k): v for k, v in Counter(labels).items()}
        # mlflow.log_dict(cluster_counts, "cluster_distribution.json")
        # mlflow.log_dict(cluster_counts, "cluster_distribution.json")

        # PCA (optional for variance tracking)
        # pca = PCA(n_components=2)
        # _ = pca.fit(features)
        # mlflow.log_metric("pca_var_ratio_1", pca.explained_variance_ratio_[0])
        # mlflow.log_metric("pca_var_ratio_2", pca.explained_variance_ratio_[1])

        # mlflow.log_metric("num_features", features.shape[1])

    return df

if __name__ == "__main__":
    cfg = load_config()
    latest_file = sorted(os.listdir("data/processed"))[-1]
    
    print(f"🔍 Loading data from {latest_file}")
    df = load_clean_data(os.path.join("data/processed", latest_file))
    features, original_df = engineer_features(df)

    clustered_df = cluster_data(features, original_df, cfg.get("n_clusters", 5))
    output_path = os.path.join("data/clustered", "clustered_observations.csv")
    clustered_df.to_csv(output_path, index=False)
    print(f"✅ Saved clustered data to {output_path}")
