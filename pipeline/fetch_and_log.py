import os
import sys
import json
from datetime import datetime

# Dynamically add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from pipeline.utils import fetch_observations, load_config
from cloud.upload import upload_to_mock_cloud

def main():
    cfg = load_config()
    location = cfg["location_name"]
    taxon = cfg.get("taxon_name")
    per_page = cfg.get("per_page", 50)
    date_range = cfg.get("date_range")
    start = None
    end = None
    

    print(f"📅 Fetching observations for {location} (taxon: {taxon or 'any'})")
    
    if date_range:
        start = date_range.get("start")
        end = date_range.get("end")
        print(f"📆 Between dates: {start} and {end}")
        date_tag = f"{start}_{end}"
    else:
        print("📆 No date range specified, fetching latest observations")
        date_tag = datetime.now().strftime("%Y-%m-%d")

    result, resolved_place_name = fetch_observations(
        place_name=cfg.get("location_name"),
        taxon_name=cfg.get("taxon_name"),
        per_page=cfg.get("per_page", 20),
        date_start=start,
        date_end=end
    )

    log_path = os.path.join("data/logs", f"observations_{date_tag}.json")
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)

    # save_log(result, log_path)
    print(f"✅ Saved log to {log_path}")

    # # uploading to mock cloud (can be replaced with actual cloud upload logic)
    # uploaded_path = upload_to_mock_cloud(log_path)
    # print(f"✅ Uploaded log to mock cloud")

if __name__ == "__main__":
    cfg = load_config()
    main()