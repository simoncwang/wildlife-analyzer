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
from datetime import datetime
import json

# Dynamically add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from pipeline.utils import load_config
from pipeline.feature_engineering import engineer_features, load_clean_data
from cloud.upload import upload_to_mock_cloud, upload_to_s3

def calculate_metrics(model, features, labels):
    metrics = {}
    
    # Inertia
    metrics["inertia"] = model.inertia_
    
    # Silhouette Score
    try:
        metrics["silhouette_score"] = silhouette_score(features, labels)
    except ValueError:
        metrics["silhouette_score"] = None  # Not enough clusters or samples
    
    # Cluster counts
    cluster_counts = Counter(labels)
    metrics["cluster_distribution"] = {int(k): v for k, v in cluster_counts.items()}

    # PCA (optional for variance tracking)
    pca = PCA(n_components=2)
    _ = pca.fit(features)
    metrics["pca_var_ratio_1"] = pca.explained_variance_ratio_[0]
    metrics["pca_var_ratio_2"] = pca.explained_variance_ratio_[1]
    metrics["num_features"] = features.shape[1]
    
    return metrics

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

    return model, df, model.cluster_centers_

if __name__ == "__main__":
    cfg = load_config()
    latest_file = sorted(os.listdir("data/processed"))[-1]
    
    # load preprocessed data
    print(f"🔍 Loading data from {latest_file}")
    df = load_clean_data(os.path.join("data/processed", latest_file))

    # engineer features
    features, original_df = engineer_features(df)

    model, clustered_df, centroids = cluster_data(features, original_df, cfg.get("n_clusters", 5))

    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cluster_path = os.path.join("data/clustered", f"clustered_observations_{now_str}.csv")
    centroids_path = os.path.join("data/clustered", f"centroids_{now_str}.npy")

    # Save data
    clustered_df.to_csv(cluster_path, index=False)
    np.save(centroids_path, centroids)

    print(f"✅ Saved clustered data to {cluster_path}")
    print(f"✅ Saved centroids to {centroids_path}")

    # get metrics and save to cloud
    # Ensure metrics directory exists
    os.makedirs("data/metrics", exist_ok=True)
    metrics = calculate_metrics(model, features, clustered_df["cluster"])
    metrics_path = os.path.join("data/metrics", f"clustering_metrics_{now_str}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved clustering metrics to {metrics_path}")

    # uploading to mock cloud or s3 bucket
    if cfg.get("cloud_backend") == "s3":
        # Generate relative S3 key based on local `data/` structure
        s3_key = os.path.relpath(metrics_path, start="data")
        upload_to_s3(metrics_path, cfg["s3_bucket"], s3_key)
        print(f"\n☁️ Uploaded metrics s3 bucket {cfg['s3_bucket']}")
    else:
        uploaded_path = upload_to_mock_cloud(metrics_path)
        print(f"\n☁️ Uploaded metrics to mock cloud")
