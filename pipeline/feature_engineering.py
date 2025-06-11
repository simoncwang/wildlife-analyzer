import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
import sys

# Dynamically add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from pipeline.utils import load_config

def load_clean_data(path):
    return pd.read_csv(path)

def engineer_features(df):
    # Drop rows with missing lat/lon
    df = df.dropna(subset=["lat", "lon"])

    # Encode species
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    species_encoded = encoder.fit_transform(df[["species"]])  # Ensures 2D input and output

    coords = df[["lat", "lon"]].values  # shape: (n_samples, 2)

    # Now safely combine
    features = np.hstack([coords, species_encoded])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    return scaled, df

if __name__ == "__main__":
    cfg = load_config()

    latest_file = sorted(os.listdir("data/processed"))[-1]
    print(f"🔍 Loading data from {latest_file}")
    df = load_clean_data(os.path.join("data/processed", latest_file))
    features, original_df = engineer_features(df)
    print(f"✅ Engineered features for {len(original_df)} entries.")
