import streamlit as st
import pandas as pd
import os
import glob
import json
import time
import plotly.express as px
import yaml
import subprocess
import sys

# Dynamically add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from pipeline.utils import search_places, search_taxa, load_config, save_config

# TITLE
st.set_page_config(page_title="Wildlife Analyzer", layout="wide")
st.title("🦉 Wildlife Analyzer Dashboard")

# SIDEBAR

st.sidebar.header("⚙️ Pipeline Parameters")
cfg = load_config()

# select location
st.sidebar.markdown("### 🌍 Location")
location_mode = st.sidebar.radio("Input mode for location", ["Enter manually", "Search by name"])

if location_mode == "Enter manually":
    location = st.sidebar.text_input("Location name", cfg.get("location_name", ""))
else:
    place_query = st.sidebar.text_input("🔍 Search place name")
    location = ""
    if place_query:
        matching_places = search_places(place_query)
        place_options = [f"{p['display_name']} (id={p['id']})" for p in matching_places]
        if place_options:
            selected = st.sidebar.selectbox("Matching places", place_options)
            selected_index = place_options.index(selected)
            location = matching_places[selected_index]["display_name"]

# select species
st.sidebar.markdown("### 🐾 Species")
species_mode = st.sidebar.radio("Species input mode", ["Any species", "Enter manually", "Search by name"])

taxon = None  # default to None if "Any species"

if species_mode == "Enter manually":
    taxon = st.sidebar.text_input("Species (taxon_name)", cfg.get("taxon_name") or "")
elif species_mode == "Search by name":
    taxon_query = st.sidebar.text_input("🔍 Search species name")
    if taxon_query:
        matching_taxa = search_taxa(taxon_query)
        taxon_options = [
            f"{t.get('preferred_common_name', '')} ({t['name']})" if t.get("preferred_common_name")
            else t["name"]
            for t in matching_taxa
        ]
        if taxon_options:
            selected = st.sidebar.selectbox("Matching species", taxon_options)
            selected_index = taxon_options.index(selected)
            taxon = matching_taxa[selected_index]["name"]

# other params
per_page = st.sidebar.number_input("Observations (max 200)", min_value=1, max_value=200, value=cfg.get("per_page", 50))
n_clusters = st.sidebar.number_input("# Clusters for KMeans", min_value=1, max_value=20, value=cfg.get("n_clusters", 5))
run_mode = st.sidebar.selectbox(
    "Pipeline Run Mode",
    options=["clustering", "llm_summary", "both"],
    index=["clustering", "llm_summary", "both"].index(cfg.get("run_mode", "clustering"))
)

# save parameters
if st.sidebar.button("💾 Save Parameters"):
    new_cfg = {
        "location_name": location,
        "taxon_name": taxon or None,
        "per_page": int(per_page),
        "n_clusters": int(n_clusters),
        "run_mode": run_mode
    }
    save_config(new_cfg)
    st.sidebar.success("Parameters updated!")

# PIPELINE RUN
st.header("🚀 Run Pipeline")

if st.button("▶️ **Run Full Pipeline**"):
    with st.status("Running pipeline...", expanded=True) as status:

        st.write("📥 **Fetching observations from iNaturalist API...**")
        result = subprocess.run(["python", "pipeline/fetch_and_log.py"], capture_output=True, text=True)
        st.write(result.stdout)
        if result.returncode != 0:
            st.error("❌ Fetch failed.")
            status.update(label="Pipeline failed", state="error")
            st.stop()

        st.write("🧹 **Preprocessing observations...**")
        result = subprocess.run(["python", "pipeline/preprocess.py"], capture_output=True, text=True)
        st.write(result.stdout)
        if result.returncode != 0:
            st.error("❌ Preprocessing failed.")
            status.update(label="Pipeline failed", state="error")
            st.stop()
        
        if run_mode in ["clustering", "both"]:
            st.write("🧠 **Feature engineering...**")
            result = subprocess.run(["python", "pipeline/feature_engineering.py"], capture_output=True, text=True)
            st.write(result.stdout)
            if result.returncode != 0:
                st.error("❌ Feature engineering failed.")
                status.update(label="Pipeline failed", state="error")
                st.stop()

            st.write("🔗 **Clustering observations (location, species)...**")
            result = subprocess.run(["python", "models/cluster.py"], capture_output=True, text=True)
            st.write(result.stdout)
            if result.returncode != 0:
                st.error("❌ Clustering failed.")
                status.update(label="Pipeline failed", state="error")
                st.stop()

        if run_mode in ["llm_summary", "both"]:
            st.write("🧠 **Generating LLM summary...**")
            result = subprocess.run(["python", "models/llm_summary.py"], capture_output=True, text=True)
            st.write(result.stdout)
            if result.returncode != 0:
                st.error("❌ LLM summary failed.")
                status.update(label="Pipeline failed", state="error")
                st.stop()

        # log most recent run metadata
        st.write("📅 **Logging run metadata...**")
        run_metadata = {
            "mode": run_mode,
            "timestamp": time.time()
        }
        with open("data/last_run.json", "w") as f:
            json.dump(run_metadata, f)

        st.success("✅ All pipeline stages completed successfully!")
        status.update(label="Pipeline complete", state="complete")

# RUN SUMMARY
st.write("📋 **Latest Run Summary**")
log_path = max(glob.glob("data/logs/*.json"), default=None, key=os.path.getmtime)
if log_path:
    with open(log_path) as f:
        log = json.load(f)
    
    total_results = log.get("total_results", 0)
    per_page = log.get("per_page", 20)

    fetched = total_results if total_results < per_page else per_page

    st.write(f"**Location:** {log.get('input_location')} | **Resolved:** {log.get('resolved_location')}")
    st.write(f"**Species:** {taxon or 'Any'}")
    st.write(f"**Total Observations:** {total_results} | **Fetched:** {fetched} observations")
    st.download_button(
        "⬇️ Download Raw Log",
        data=open(log_path,"rb").read(),
        file_name=os.path.basename(log_path),
        mime="application/json"
    )
else:
    st.info("No logs found. Run the pipeline first.")

# VISUALIZATION
if run_mode in ["clustering", "both"]:

    # check most recent run metadata to see if anything should be displayed
    show_visualization = False
    run_info_path = "data/last_run.json"
    if os.path.exists(run_info_path):
        with open(run_info_path) as f:
            run_info = json.load(f)
        last_mode = run_info.get("mode")
        if last_mode in ["clustering", "both"]:
            show_visualization = True

    if show_visualization:
        st.header("📍 Cluster Map or Scatter")
        cluster_file = max(glob.glob("data/clustered/*.csv"), default=None, key=os.path.getmtime)
        if cluster_file:
            df = pd.read_csv(cluster_file)
            map_type = st.radio("Choose visualization type:", ["Map (if coordinates available)", "Scatter Plot"])
            color_scale = st.selectbox(
                "Color Scale",
                ["Agsunset_r", "Viridis", "Cividis", "Plasma", "Inferno", "Magma"],
                index=0,
                help="Choose a color scale for the clusters."
            )

            if map_type == "Map (if coordinates available)":
                fig = px.scatter_map(
                    df,
                    lat="lat",
                    lon="lon",
                    color="cluster",
                    hover_data=["species", "observer", "date"],
                    zoom=5,
                    height=600,
                    color_continuous_scale=color_scale,
                )
                fig.update_traces(marker=dict(size=8, opacity=0.8))

                fig.update_layout(mapbox_style="open-street-map")
                fig.update_layout(margin={"r":0,"t":0,"l":0,"b":40})
                st.plotly_chart(fig, use_container_width=True)
            
            elif map_type == "Scatter Plot":
                fig = px.scatter(
                    df, x="lon", y="lat", color="cluster",
                    hover_data=["species", "observer", "date"],
                    title="Clustered Observations",
                    color_continuous_scale=color_scale
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Clustering was not part of the last pipeline run. Run the pipeline with 'clustering' or 'both' mode.")

# LLM SUMMARY
if run_mode in ["llm_summary", "both"]:
    st.header("🧠 LLM Summary")

    summary_path = "data/summary/latest_summary.txt"
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary_text = f.read()

        # Scrollable markdown container using HTML/CSS
        st.markdown(summary_text, unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download Summary",
            data=summary_text,
            file_name="latest_summary.txt",
            mime="text/plain"
        )
        st.success("Summary generated successfully!")
    else:
        st.info("No summary generated yet. Run the pipeline to produce one.")

# PREVIEW CLEANED DATA
st.header("🔍 Cleaned Data Preview")
clean_file = max(glob.glob("data/processed/*.csv"), default=None, key=os.path.getmtime)
if clean_file:
    st.dataframe(pd.read_csv(clean_file).head(20))
else:
    st.info("No cleaned data available.")