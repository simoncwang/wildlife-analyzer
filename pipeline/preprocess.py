import os
import json
import pandas as pd
from datetime import datetime
import yaml
import sys
import glob

# Dynamically add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from pipeline.utils import load_config
from cloud.upload import upload_to_mock_cloud, upload_to_s3

def clean_observations(raw_obs):
    cleaned = []
    print(f"🔍 Cleaning {len(raw_obs)} observations...")
    for obs in raw_obs:
        taxon = obs.get("taxon") or {}
        cleaned.append({
            "id": obs.get("id"),
            "species": taxon.get("preferred_common_name", "Unknown"),
            "scientific_name": taxon.get("name", "Unknown"),
            "observer": obs.get("user", {}).get("login", "Anonymous"),
            "date": obs.get("observed_on"),
            "lat": obs.get("geojson", {}).get("coordinates", [None, None])[1],
            "lon": obs.get("geojson", {}).get("coordinates", [None, None])[0],
            "place_guess": obs.get("place_guess"),
            "image_url": obs["photos"][0]["url"].replace("square", "medium") if obs.get("photos") else "",
        })
    return pd.DataFrame(cleaned)

if __name__ == "__main__":
    cfg = load_config()
    timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")

    # This assumes a single run output is saved to this file
    # input_path = "data/logs/latest_observations.json"

    # get latest log file
    input_path = max(glob.glob("data/logs/*.json"), default=None, key=os.path.getmtime)

    if not os.path.exists(input_path):
        print(f"❌ No observation file found at {input_path}")
        exit(1)

    print(f"🔍 Loading data from {input_path}")
    with open(input_path) as f:
        data = json.load(f)

    raw_obs = data.get("results", [])
    
    num_obs = len(raw_obs)
    if num_obs == 0:
        print("❌ No observations found in the input file")
        exit(1)

    df = clean_observations(raw_obs)

    output_dir = "data/processed/"

    output_path = os.path.join(output_dir, f"{timestamp}.csv")
    df.to_csv(output_path, index=False)

    print(f"✅ Cleaned data saved to {output_path} ({len(df)} rows)")

    # uploading to mock cloud or s3 bucket
    if cfg.get("cloud_backend") == "s3":
        # Generate relative S3 key based on local `data/` structure
        s3_key = os.path.relpath(output_path, start="data")
        upload_to_s3(output_path, cfg["s3_bucket"], s3_key)
        print(f"\n☁️ Uploaded data s3 bucket {cfg['s3_bucket']}")
    else:
        uploaded_path = upload_to_mock_cloud(output_path)
        print(f"\n☁️ Uploaded data to mock cloud")
    
