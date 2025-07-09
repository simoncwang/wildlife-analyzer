import streamlit as st
import pandas as pd
import os
import glob
import json
import time
import plotly.express as px
import subprocess
import sys
import mlflow

# Dynamically add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from pipeline.utils import search_places, search_taxa, load_config, save_config

# Set up layout
st.set_page_config(page_title="Wildlife Analyzer", layout="wide")
st.title("🦉 Wildlife Analyzer Dashboard")
st.markdown("*Simon Wang | [Website](https://simoncwang.github.io/)* 🌐 *| [GitHub](https://github.com/simoncwang)* 👨‍💻 *| [LinkedIn](https://www.linkedin.com/in/simon-wang-519902193/)* 🔗")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Settings")
cfg = load_config()

st.sidebar.markdown("### Location")
location_mode = st.sidebar.radio("Input mode for location", ["Search by name", "Enter manually (must be exact as on iNaturalist)"])

if location_mode == "Enter manually (must be exact as on iNaturalist)":
    location = st.sidebar.text_input("Location name", cfg.get("location_name", ""))
else:
    place_query = st.sidebar.text_input("🔍 Search place name", cfg.get("location_name", ""))
    location = ""
    if place_query:
        matching_places = search_places(place_query)
        place_options = [f"{p['display_name']} (id={p['id']})" for p in matching_places]
        if place_options:
            selected = st.sidebar.selectbox("Matching places", place_options)
            selected_index = place_options.index(selected)
            location = matching_places[selected_index]["display_name"]

st.sidebar.markdown("### Species")
species_mode = st.sidebar.radio("Species input mode", ["Any species", "Search by name", "Enter manually (must be exact as on iNaturalist)"])

taxon = None
if species_mode == "Enter manually (must be exact as on iNaturalist)":
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

st.sidebar.markdown("### Pipeline Parameters")

per_page = st.sidebar.number_input("Number of Observations (max 200)", min_value=1, max_value=200, value=cfg.get("per_page", 50))

all_modes = ["clustering", "llm_summary", "drift_analysis"]
default_modes = cfg.get("run_mode", ["clustering"])  # default as list
run_mode = st.sidebar.multiselect(
    "Pipeline Run Modes (select one or more)",
    options=all_modes,
    default=default_modes
)

openai_api_key = None
if "llm_summary" in run_mode:
    st.sidebar.markdown("### LLM Summary Parameters")
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key"
    )

if "clustering" in run_mode or "drift_analysis" in run_mode:
    n_clusters = st.sidebar.number_input("Number of Clusters for KMeans", min_value=1, max_value=20, value=cfg.get("n_clusters", 5))

# Show date range inputs if enabled
start_date, end_date = None, None
date_range = st.sidebar.checkbox("📅 Date Range", value=cfg.get("date_range") is not None)
if date_range:
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")

if start_date and end_date and start_date > end_date:
    st.sidebar.error("Start date must be before end date.")

# cloud backend selection
cloud_backend = st.sidebar.selectbox(
    "Cloud Backend (only use mock for demo, s3 requires local AWS credentials)",
    options=["mock", "s3"],
    index=0 if cfg.get("cloud_backend") == "mock" else 1,
    help="Select the cloud backend for storing data. 'mock' uses a local directory, 's3' requires AWS credentials."
)

if cloud_backend == "s3":
    s3_bucket = st.sidebar.text_input("S3 Bucket Name", value=cfg.get("s3_bucket", ""))
    if not s3_bucket:
        st.sidebar.error("S3 Bucket Name is required for S3 backend.")

if st.sidebar.button("💾 Save Parameters"):
    new_cfg = {
        "location_name": location,
        "taxon_name": taxon or None,
        "per_page": int(per_page),
        "run_mode": run_mode,
        "cloud_backend": cloud_backend,
        "s3_bucket": s3_bucket if cloud_backend == "s3" else None
    }
    new_cfg["date_range"] = {
        "start": str(start_date),
        "end": str(end_date)
    }

    if "clustering" in run_mode or "drift_analysis" in run_mode:
        new_cfg["n_clusters"] = int(n_clusters)

    save_config(new_cfg)
    st.sidebar.success("Parameters updated!")

# --- TABS ---
tabs = st.tabs([
    "📖 Instructions",
    "🚀 Run Pipeline",
    "📋 Summary",
    "📍 Cluster Visualization",
    "🧠 LLM Summary",
    "🔍 Cleaned Data",
    "☁️ Mock Cloud Files"
])

# INSTRUCTIONS
with tabs[0]:
    st.header("📖 Instructions")
    st.markdown("""
    Welcome to the Wildlife Analyzer Dashboard! This tool allows you to fetch, preprocess, and analyze wildlife observations from [iNaturalist API](https://api.inaturalist.org/v1/docs/).
                
    ### Usage:
    1. **Set Parameters**: Use the sidebar to configure your pipeline parameters:
        - **Location**: Specify the area you want to analyze.
        - **Species**: Choose a specific species or leave it as "Any species".
        - **Observations**: Set the maximum number of observations to fetch (up to 200).
        - **Clustering**: Set the number of clusters for KMeans (default is 5).
        - **Run Mode**: Choose between:
            - **Clustering**: Cluster the observations based on location and species.
            - **LLM Summary**: Generate a summary using a language model.
            - **Both**: Run both clustering and LLM summary.
            - If using LLM summary, either set the OpenAI API key in Streamlit secrets or enter it into the input field.
        - Click "Save Parameters" to apply your settings.
    2. **Run Pipeline**: Click the "Run Full Pipeline" button in the "Run Pipeline" tab to execute the pipeline stages:
        - The pipeline stages include fetching data, preprocessing, feature engineering, clustering (if selected), and generating an LLM summary (if selected).
    3. **View Results**: After the pipeline completes, you can view:
        - A brief summary of the latest run in the "Summary" tab.
        - A visualization of the clustered data in the "Cluster Visualization" tab (if clustering was run).
        - The generated LLM summary in the "LLM Summary" tab (if LLM summary was run).
        - A preview of cleaned data in the "Cleaned Data Preview" tab.
    
    **Have fun exploring wildlife data! 🫎**
                
    #### Note (using MLflow):
    - The pipeline stages are set up to be logged using [MLflow](https://mlflow.org/), which allows you to track runs, parameters, and artifacts. Currently all mlflow code is commented out, but you can uncomment it to enable logging.
    - To use MLflow with this dashboard, you must install and run the project locally by cloning and following the instructions in the [GitHub repository](https://github.com/simoncwang/wildlife-analyzer)
    - Once set up, you can view and track various runs, parameters, artifacts, and model metrics in the MLflow UI.
    - Currently, I have only set up some basic logging, but you can extend it to log more detailed metrics, artifacts, and parameters as needed by inspecting the code in the UI and models directory scripts.
                
    ### Have questions or feedback?
    - Check out the [GitHub repository](https://github.com/simoncwang/wildlife-analyzer) for documentation and issues.
    - Contact me via email at wang.c.simon@gmail.com, or check out my [Personal Website](https://simoncwang.github.io/) for more projects and info!
                
    ### Resources/Tools Used:
    - [Streamlit](https://streamlit.io/) for the dashboard interface.
    - [iNaturalist API](https://api.inaturalist.org/v1/docs/) for fetching wildlife observations.
    - [OpenAI API](https://platform.openai.com/docs/api-reference) for generating summaries.
    - [Plotly](https://plotly.com/python/) for interactive visualizations.
    - [Pandas](https://pandas.pydata.org/) for data manipulation.
    - [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) from scikit-learn for clustering observations.
    - [MLflow](https://mlflow.org/) for tracking runs and logging model metrics.
    """)

# RUN PIPELINE
with tabs[1]:
    st.header("🚀 Run Pipeline")

    if st.button("▶️ **Run**"):
        with st.status("Running pipeline...", expanded=True) as status:
            # with mlflow.start_run(run_name="wildlife_pipeline") as run:

                st.write("📥 **Fetching observations from iNaturalist API...**")
                result = subprocess.run([sys.executable, "pipeline/fetch_and_log.py"], capture_output=True, text=True)
                st.write(result.stdout)
                # mlflow.log_text(result.stdout, "logs/fetch.log")
                if result.returncode != 0:
                    # mlflow.set_tag("status", "failed_fetch")
                    st.error("❌ Fetch failed.")
                    st.write(result.stdout)
                    st.write(result.stderr)
                    status.update(label="Pipeline failed", state="error")
                    st.stop()

                st.write("🧹 **Preprocessing observations...**")
                result = subprocess.run([sys.executable, "pipeline/preprocess.py"], capture_output=True, text=True)
                st.write(result.stdout)
                # mlflow.log_text(result.stdout, "logs/preprocess.log")
                if result.returncode != 0:
                    # mlflow.set_tag("status", "failed_preprocess")
                    st.error("❌ Preprocessing failed.")
                    status.update(label="Pipeline failed", state="error")
                    st.stop()

                if "clustering" in run_mode:
                    # st.write("🧠 **Feature engineering...**")
                    # result = subprocess.run([sys.executable, "pipeline/feature_engineering.py"], capture_output=True, text=True)
                    # st.write(result.stdout)
                    # # mlflow.log_text(result.stdout, "logs/features.log")
                    # if result.returncode != 0:
                    #     # mlflow.set_tag("status", "failed_feature_eng")
                    #     st.error("❌ Feature engineering failed.")
                    #     status.update(label="Pipeline failed", state="error")
                    #     st.stop()

                    st.write("🔗 **Clustering observations (location, species)...**")
                    result = subprocess.run([sys.executable, "models/cluster.py"], capture_output=True, text=True)
                    st.write(result.stdout)
                    # mlflow.log_text(result.stdout, "logs/clustering.log")
                    if result.returncode != 0:
                        # mlflow.set_tag("status", "failed_clustering")
                        st.write(result.stderr)
                        st.error("❌ Clustering failed.")
                        status.update(label="Pipeline failed", state="error")
                        st.stop()
                    
                    st.write("📊 **Check out the results in the Cluster Visualization tab!**")

                    # latest_cluster = max(glob.glob("data/clustered/*.csv"), default=None, key=os.path.getmtime)
                    # if latest_cluster:
                    #     mlflow.log_artifact(latest_cluster, artifact_path="clustered")

                if "llm_summary" in run_mode:
                    st.write("🧠 **Generating LLM summary...**")
                    result = subprocess.run([sys.executable,
                        "models/llm_summary.py",
                        "--api_key", openai_api_key
                        ], capture_output=True, text=True)
                    st.write(result.stdout)
                    # mlflow.log_text(result.stdout, "logs/llm_summary.log")
                    if result.returncode != 0:
                        # st.write(result.stderr)
                        # mlflow.set_tag("status", "failed_llm_summary")
                        st.error("❌ LLM summary failed.")
                        status.update(label="Pipeline failed", state="error")
                        st.stop()

                    st.write("📊 **Check out the results in the LLM Summary tab!**")

                    # summary_path = "data/summary/latest_summary.txt"
                    # if os.path.exists(summary_path):
                    #     mlflow.log_artifact(summary_path, artifact_path="summary")
                
                if "drift_analysis" in run_mode:
                    st.write("📈 **Running drift analysis...**")
                    st.write("⚠️ This feature is currently in progress and will be available soon.")
                    # result = subprocess.run([sys.executable, "models/drift_analysis.py"], capture_output=True, text=True)
                    # st.write(result.stdout)
                    # # mlflow.log_text(result.stdout, "logs/drift_analysis.log")
                    # if result.returncode != 0:
                    #     st.write(result.stderr)
                    #     # mlflow.set_tag("status", "failed_drift_analysis")
                    #     st.error("❌ Drift analysis failed.")
                    #     status.update(label="Pipeline failed", state="error")
                    #     st.stop()

                    # st.write("📊 **Drift analysis completed!**")

                run_metadata = {
                    "mode": run_mode,
                    "timestamp": time.time()
                }
                with open("data/last_run.json", "w") as f:
                    json.dump(run_metadata, f)

                # mlflow.log_params({
                #     "location": location,
                #     "species": taxon or "Any",
                #     "per_page": per_page,
                #     "n_clusters": n_clusters,
                #     "run_mode": run_mode
                # })

                # mlflow.set_tag("status", "completed")
                st.success("✅ All pipeline stages completed successfully!")
                status.update(label="Pipeline complete", state="complete")

# SUMMARY
with tabs[2]:
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

# CLUSTER VISUALIZATION
with tabs[3]:
    if "clustering" in run_mode:

        # check most recent run metadata to see if anything should be displayed
        show_visualization = False
        run_info_path = "data/last_run.json"
        if os.path.exists(run_info_path):
            with open(run_info_path) as f:
                run_info = json.load(f)
            last_mode = run_info.get("mode")
            if isinstance(last_mode, str):
                last_mode = [last_mode]
            if "clustering" in last_mode:
                show_visualization = True

        if show_visualization:
            st.header("📍 Cluster Map or Scatter")
            st.markdown("**NOTE:** The cluster color scale is currently using a continuous scale, which is not ideal for categorical clusters. This will be improved in future versions.")
            # get latest clustered data file
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
with tabs[4]:
    if "llm_summary" in run_mode:
        st.header("🧠 LLM Summary")

        # get latest llm summary file
        summary_path = max(glob.glob("data/summary/*.txt"), default=None, key=os.path.getmtime)
        # summary_path = "data/summary/latest_summary.txt"
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
        st.info("No LLM summary generated yet. Run the pipeline with 'llm_summary' or 'both' mode.")

# PREVIEW CLEANED DATA
with tabs[5]:
    st.header("🔍 Cleaned Data Viewer")
    clean_file = max(glob.glob("data/processed/*.csv"), default=None, key=os.path.getmtime)
    if clean_file:
        st.dataframe(pd.read_csv(clean_file))
    else:
        st.info("No cleaned data available.")

# CLOUD FILE BROWSER
with tabs[6]:
    st.header("☁️ View Mock Cloud Files")
    st.markdown("This section allows you to browse and preview files stored in the mock cloud directory. This demo version uses a local directory to simulate a cloud storage system. In a real application, this would connect to an actual cloud service like AWS S3, Google Cloud Storage, etc. **To use real cloud storage**, clone the repository and set up as instructed in the README.")

    mock_cloud_root = "mock_cloud"

    if not os.path.exists(mock_cloud_root):
        st.info("No mock cloud files found.")
    else:
        # Group files by their immediate subfolder
        grouped_files = {}

        for root, _, files in os.walk(mock_cloud_root):
            rel_dir = os.path.relpath(root, mock_cloud_root)
            if rel_dir == ".":
                rel_dir = "root"

            for f in files:
                full_path = os.path.join(root, f)
                grouped_files.setdefault(rel_dir, []).append((f, full_path))

        if not grouped_files:
            st.info("No files found in mock cloud.")
        else:
            for folder, files in sorted(grouped_files.items()):
                with st.expander(f"📁 {folder}/", expanded=False):
                    file_names = [f for f, _ in files]
                    selected_file = st.selectbox(
                        f"Select a file in '{folder}'", file_names, key=folder
                    )

                    selected_path = dict(files)[selected_file]

                    st.markdown(f"**🗂 Previewing:** `{selected_file}`")

                    # checking if file is empty
                    if os.path.getsize(selected_path) == 0:
                        st.warning("This file is empty.")
                        continue
                    else:

                        # previewing the file content based on its type
                        if selected_file.endswith(".json"):
                            with open(selected_path) as f:
                                content = f.read()
                                st.json(json.loads(content))
                        elif selected_file.endswith(".txt"):
                            with open(selected_path) as f:
                                content = f.read()
                                st.text(content)
                        elif selected_file.endswith(".csv"):
                            df = pd.read_csv(selected_path)
                            st.dataframe(df)
                            content = df.to_csv(index=False)
                        else:
                            content = None
                            st.warning("Unsupported file type for preview.")

                        # downloading the file
                        if content is not None:
                            st.download_button(
                                label="⬇️ Download File",
                                data=content,
                                file_name=selected_file,
                                mime="text/plain" if selected_file.endswith(".txt") else "application/octet-stream"
                            )
