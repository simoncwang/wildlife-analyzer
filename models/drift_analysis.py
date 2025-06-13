import os
import numpy as np
import json
from scipy.spatial.distance import cdist

def load_centroids(centroid_dir):
    files = sorted(
        [f for f in os.listdir(centroid_dir) if f.startswith("centroids_") and f.endswith(".npy")]
    )
    if len(files) < 2:
        raise ValueError("At least two centroid files are required for drift analysis.")
    return files[-2], files[-1]  # previous, latest

def compute_drift(file1, file2, centroid_dir):
    c1 = np.load(os.path.join(centroid_dir, file1))
    c2 = np.load(os.path.join(centroid_dir, file2))

    if c1.shape[1] != c2.shape[1]:
        raise ValueError(f"Centroid dimension mismatch: {file1} has shape {c1.shape}, {file2} has shape {c2.shape}. Make sure features are consistent across runs.")

    dist_matrix = cdist(c1, c2)
    matched_indices = np.argmin(dist_matrix, axis=1)
    drifts = dist_matrix[np.arange(len(matched_indices)), matched_indices]
    return {
        "previous_file": file1,
        "latest_file": file2,
        "mean_drift": float(np.mean(drifts)),
        "max_drift": float(np.max(drifts)),
        "drift_per_cluster": drifts.tolist()
    }

if __name__ == "__main__":
    centroid_dir = "data/clustered"
    drift_dir = "data/drift"
    os.makedirs(drift_dir, exist_ok=True)

    prev_file, latest_file = load_centroids(centroid_dir)
    print(f"🔍 Comparing centroids: {prev_file} vs {latest_file}")
    drift_metrics = compute_drift(prev_file, latest_file, centroid_dir)

    output_path = os.path.join(
        drift_dir,
        f"drift_{latest_file.replace('centroids_', '').replace('.npy', '')}.json"
    )
    with open(output_path, "w") as f:
        json.dump(drift_metrics, f, indent=2)

    print(f"✅ Drift analysis complete.\nSaved to: {output_path}")
    print(json.dumps(drift_metrics, indent=2))
