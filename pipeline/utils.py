import requests
import os
import yaml

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pipeline_config.yaml"))

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

def search_taxa(query, limit=10):
    url = "https://api.inaturalist.org/v1/taxa"
    params = {
        "q": query,
        "per_page": limit
    }
    response = requests.get(url, params=params)
    return response.json().get("results", [])

def search_places(query, limit=10):
    url = "https://api.inaturalist.org/v1/places/autocomplete"
    params = {
        "q": query,
        "per_page": limit
    }
    response = requests.get(url, params=params)
    return response.json().get("results", [])

def get_place_id(place_name):
    """Look up iNaturalist place ID using the autocomplete API."""
    url = "https://api.inaturalist.org/v1/places/autocomplete"
    params = {"q": place_name}
    response = requests.get(url, params=params)
    results = response.json().get("results", [])
    
    if results:
        return results[0]["id"], results[0]["display_name"]
    else:
        return None, None

def fetch_observations(place_name=None, place_id=None, taxon_name=None, per_page=20, date_start=None, date_end=None):
    """Fetch observations from iNaturalist using either place_id or place_name.
       Optionally filter by date range using ISO format: YYYY-MM-DD.
    """
    
    # If no place_id is provided, try to resolve it from place_name
    resolved_place_name = None
    if not place_id and place_name:
        place_id, resolved_place_name = get_place_id(place_name)
        if not place_id:
            raise ValueError(f"Could not find place ID for: {place_name}")
    
    url = "https://api.inaturalist.org/v1/observations"
    params = {
        "per_page": per_page,
        "order_by": "observed_on",
        "order": "desc",
        "verifiable": "true",
        "photos": "true",
        "place_id": place_id
    }

    if taxon_name:
        params["taxon_name"] = taxon_name
    if date_start:
        params["d1"] = date_start
    if date_end:
        params["d2"] = date_end

    response = requests.get(url, params=params)
    data = response.json()

    return data, resolved_place_name or place_name
