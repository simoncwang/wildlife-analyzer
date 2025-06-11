import os
import json
import sys
from datetime import datetime

# Dynamically add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from pipeline.utils import fetch_observations, load_config

LATEST_OUTPUT = "data/logs/latest_observations.json"
TIMESTAMPED_DIR = "data/raw"

def main():
    # Load configuration
    cfg = load_config()
    loc = cfg["location_name"]
    taxon = cfg.get("taxon_name")
    per_page = cfg.get("per_page",20)

    print(f"🔍 Fetching {per_page} observations for: {loc}")
    data, resolved = fetch_observations(place_name=loc, taxon_name=taxon, per_page=per_page)

    # Save latest
    os.makedirs(os.path.dirname(LATEST_OUTPUT), exist_ok=True)

    # adding timestamp and location to the json data
    data["timestamp"] = datetime.now().isoformat()
    data["input_location"] = loc
    data["resolved_location"] = resolved
    with open(LATEST_OUTPUT, "w") as f:
        json.dump(data, f, indent=2)

    # Save timestamped raw
    os.makedirs(TIMESTAMPED_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = os.path.join(TIMESTAMPED_DIR, f"observations_{timestamp}.json")
    with open(timestamped_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Saved {len(data.get('results', []))} observations to {LATEST_OUTPUT}")
    print(f" Also saved to: {timestamped_path}")
    print(f"📍 Resolved location: {resolved}")

if __name__ == "__main__":
    cfg = load_config()
    main()
