from openai import OpenAI
import pandas as pd
import os
import streamlit as st
import sys
import mlflow
import time
from datetime import datetime

# Dynamically add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from pipeline.utils import load_config
from pipeline.feature_engineering import load_clean_data
from cloud.upload import upload_to_mock_cloud, upload_to_s3

def summarize_observations(observations, model="gpt-4o"):
    client = OpenAI(api_key=st.secrets["openai"]["api_key"])

    obs_text = "\n".join([f"{o['species']} at {o['location']} on {o['observed_on']}" for o in observations])
    prompt = f"Here are some recent wildlife observations:\n\n{obs_text}\n\nSummarize what was observed and any interesting patterns."

    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model, messages=messages
    )

    return response.choices[0].message.content

def process_df(df: pd.DataFrame) -> list:
    observations = []
    for _, row in df.iterrows():
        obs = {
            "species": row["species"],
            "scientific_name": row["scientific_name"],
            "location": row["place_guess"],
            "observed_on": row["date"],
            "lat": row["lat"],
            "lon": row["lon"],
            "observer": row["observer"]
        }
        if pd.notna(row["image_url"]):
            obs["image_url"] = row["image_url"]
        observations.append(obs)
    return observations

if __name__ == "__main__":
    cfg = load_config()
    latest_file = sorted(os.listdir("data/processed"))[-1]

    print(f"🔍 Loading data from {latest_file}")
    df = load_clean_data(os.path.join("data/processed", latest_file))

    observations = process_df(df)

    summary = summarize_observations(observations)
    
    # save and log summary
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = os.path.join("data/summary", f"llm_summary_{now_str}.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    
    # uploading to mock cloud or s3 bucket
    if cfg.get("cloud_backend") == "s3":
        # Generate relative S3 key based on local `data/` structure
        s3_key = os.path.relpath(summary_path, start="data")
        upload_to_s3(summary_path, cfg["s3_bucket"], s3_key)
        print(f"\n☁️ Uploaded metrics s3 bucket {cfg['s3_bucket']}")
    else:
        uploaded_path = upload_to_mock_cloud(summary_path)
        print(f"\n☁️ Uploaded metrics to mock cloud")
    
    # COMMENT OUT ABOVE AND UNCOMMENT BELOW TO USE MLFLOW
    # cfg = load_config()
    # latest_file = sorted(os.listdir("data/processed"))[-1]

    # print(f"🔍 Loading data from {latest_file}")
    # df = load_clean_data(os.path.join("data/processed", latest_file))

    # observations = process_df(df)
    # with mlflow.start_run(run_name="llm_summary"):
    #     mlflow.set_tag("stage", "llm_summary")
    #     mlflow.log_params({
    #         "location": cfg["location_name"],
    #         "taxon": cfg["taxon_name"],
    #         "run_mode": cfg["run_mode"]
    #     })

    #     start = time.time()
        # summary = summarize_observations(observations)
        # mlflow.log_metric("llm_latency", time.time() - start)
        # mlflow.log_metric("summary_length", len(summary.split()))
    
        # save and log summary
        # summary_path = "data/summary/latest_summary.txt"
        # with open(summary_path, "w") as f:
        #     f.write(summary)
        # mlflow.log_artifact(summary_path)