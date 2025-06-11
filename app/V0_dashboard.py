import streamlit as st
import sys
import os
import json

# Dynamically add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from pipeline.utils import fetch_observations, search_taxa
from models.llm_summary import summarize_observations

st.set_page_config(page_title="🦉 Wildlife Observation Summarizer", layout="wide")
st.title("🦉 Wildlife Observation Summarizer")

# Sidebar settings
with st.sidebar:
    st.header("Settings")

    # Location Input
    location_name = st.text_input("🌍 Enter location name", value="Yellowstone National Park")

    num_observations = st.sidebar.slider("Number of observations", 5, 50, 10)

    # Species Selection
    st.markdown("### 🐾 Species Filter")
    species_filter = st.radio("Choose a filter:", ["Any species", "Specify species"])

    taxon_name = None
    if species_filter == "Specify species":
        species_input_mode = st.radio("Input method", ["Select from examples", "Search by name"])
        example_species = {
            "Bald Eagle": "Haliaeetus leucocephalus",
            "American Bison": "Bison bison",
            "Red Fox": "Vulpes vulpes",
            "Monarch Butterfly": "Danaus plexippus",
        }

        if species_input_mode == "Select from examples":
            selected = st.selectbox("Choose a species:", list(example_species.keys()))
            taxon_name = example_species[selected]
        else:
            query = st.text_input("🔍 Enter partial name (common or scientific)")
            matching_taxa = []
            if query:
                matching_taxa = search_taxa(query)
                taxon_options = [
                    f"{t.get('preferred_common_name', '')} ({t['name']})" if t.get("preferred_common_name")
                    else t["name"]
                    for t in matching_taxa
                ]
                if taxon_options:
                    selected_option = st.selectbox("Matching results", taxon_options)
                    selected_index = taxon_options.index(selected_option)
                    taxon_name = matching_taxa[selected_index]["name"]

# Fetch + Summarize
run_button = st.button("🔍 Fetch & Summarize")

if run_button:
    try:
        with st.spinner("Fetching observations..."):
            data, resolved_location = fetch_observations(place_name=location_name, taxon_name=taxon_name, per_page=num_observations)
            results = data.get("results", [])
            st.session_state["raw_data"] = results

            formatted = []
            for result in results:
                formatted.append({
                    "species": result.get("taxon", {}).get("preferred_common_name", "Unknown"),
                    "image_url": result["photos"][0]["url"].replace("square", "large") if result.get("photos") else "",
                    "observed_on": result.get("observed_on", "Unknown"),
                    "location": result.get("place_guess", "Unknown"),
                    "observer": result.get("user", {}).get("login", "Anonymous"),
                })
            st.session_state["observations"] = formatted
            st.success(f"✅ Fetched {len(results)} observations for: {resolved_location}")

        with st.spinner("Generating summary..."):
            summary = summarize_observations(st.session_state["observations"], taxon_name)
            st.session_state["summary"] = summary
            st.success("✅ Summary complete!")

    except Exception as e:
        st.error(f"Error fetching observations: {e}")

# Summary section
if "summary" in st.session_state:
    st.subheader("📋 LLM Summary")
    st.write(st.session_state["summary"])

# Display observations with scrollable container
st.subheader("📸 Recent Observations")

if "observations" in st.session_state and st.session_state["observations"]:
    st.markdown(
        """
        <div style="max-height: 600px; overflow-y: auto; padding: 1rem; border: 1px solid #ccc; border-radius: 8px;">
        """,
        unsafe_allow_html=True
    )

    for obs in st.session_state["observations"]:
        col1, col2 = st.columns([1, 3])
        with col1:
            if obs["image_url"]:
                st.image(obs["image_url"], width=250)
            else:
                st.text("No image")
        with col2:
            st.markdown(f"**Species**: {obs['species']}")
            st.markdown(f"**Observer**: {obs['observer']}")
            st.markdown(f"**Date**: {obs['observed_on']}")
            st.markdown(f"**Location**: {obs['location']}")

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("No observations to display. Enter a location and click the button.")
