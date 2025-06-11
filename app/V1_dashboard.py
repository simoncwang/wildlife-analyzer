import streamlit as st
import pandas as pd
import os
import glob
import json
from datetime import datetime
import plotly.express as px
import yaml
import subprocess

# Set up layout
st.set_page_config(page_title="Wildlife MLOps Dashboard", layout="wide")
st.title("🦉 Wildlife Observation MLOps Dashboard")

# Helper to load the most recent file from a folder
def get_latest_file(folder, extension="csv"):
    files = glob.glob(os.path.join(folder, f"*.{extension}"))
    return max(files, key=os.path.getmtime) if files else None

# === Pipeline Stage Status ===
st.header("📈 Pipeline Status")
cols = st.columns(4)
status_icons = {True: "✅", False: "❌"}

# Check existence of files at each stage
raw_file = get_latest_file("data/raw", "json")
clean_file = get_latest_file("data/processed", "csv")
feature_file = clean_file  # Assuming same source
cluster_file = get_latest_file("data/clustered", "csv")

cols[0].markdown(f"**Fetched:** {status_icons[bool(raw_file)]}")
cols[1].markdown(f"**Cleaned:** {status_icons[bool(clean_file)]}")
cols[2].markdown(f"**Engineered:** {status_icons[bool(feature_file)]}")
cols[3].markdown(f"**Clustered:** {status_icons[bool(cluster_file)]}")

# === Logs Section ===
st.header("📜 Recent Logs")
log_file = get_latest_file("data/logs", "json")

if log_file:
    with open(log_file) as f:
        log_data = json.load(f)

    st.markdown(f"**🗂 Latest Log File:** `{os.path.basename(log_file)}`")
    st.markdown(f"- **Timestamp:** {log_data.get('timestamp', 'Unknown')}")
    st.markdown(f"- **Location Queried:** {log_data.get('location_input', 'N/A')}")
    st.markdown(f"- **Species Queried:** {log_data.get('taxon_name', 'Any')}")
    st.markdown(f"- **# Observations:** {log_data.get('total_results', 'N/A')}")

    with open(log_file, "rb") as f:
        st.download_button(
            label="⬇️ Download Full Log",
            data=f,
            file_name=os.path.basename(log_file),
            mime="application/json"
        )
else:
    st.info("No logs found.")

# === Visualization Section ===
st.header("📍 Cluster Map or Scatter")
if cluster_file:
    df = pd.read_csv(cluster_file)
    map_type = st.radio("Choose visualization type:", ["Scatter Plot", "Map (if coordinates available)"])

    if map_type == "Scatter Plot":
        fig = px.scatter(
            df, x="lon", y="lat", color="cluster",
            hover_data=["species", "observer", "date"],
            title="Clustered Observations"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif map_type == "Map (if coordinates available)":
        fig = px.scatter_map(
            df,
            lat="lat",
            lon="lon",
            color="cluster",
            hover_data=["species", "observer", "date"],
            zoom=5,
            height=600,
            color_continuous_scale=px.colors.qualitative.Vivid
        )
        fig.update_traces(marker=dict(size=8))

        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":40})
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Run clustering to see results here.")

# === Optional Preview ===
with st.expander("📄 Preview Latest Cleaned Data"):
    if clean_file:
        st.dataframe(pd.read_csv(clean_file))
    else:
        st.info("No cleaned data available.")
