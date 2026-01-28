import json
import os
from pathlib import Path
from datetime import datetime
import requests
from dotenv import load_dotenv

# ----------------------------
# Setup
# ----------------------------

load_dotenv()

API_URL = "https://cruise-api1.p.rapidapi.com/cruises/search"
RAPIDAPI_HOST = "cruise-api1.p.rapidapi.com"

RAW_DATA_DIR = Path(__file__).parent / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Core function
# ----------------------------

def cache_cruises(api_key: str, payload: dict, max_pages: int = None) -> Path:
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": RAPIDAPI_HOST,
        "Content-Type": "application/json",
    }

    all_cruises = []
    page = 1
    
    while True:
        payload["page"] = page
        payload["pageSize"] = 10  # API limit is 10
        
        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code != 200:
            print("STATUS:", response.status_code)
            print("RESPONSE:", response.text)
            response.raise_for_status()

        data = response.json()
        all_cruises.extend(data.get("data", []))
        
        total_pages = data.get("total_pages", 1)
        print(f"[PAGE {page}/{total_pages}] Fetched {len(data.get('data', []))} cruises")
        
        # Stop if we've fetched all pages or reached max_pages limit
        if page >= total_pages or (max_pages and page >= max_pages):
            break
        
        page += 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RAW_DATA_DIR / f"cruise_snapshot_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump({"data": all_cruises, "total_results": len(all_cruises)}, f, indent=2)

    print(f"[OK] Cached {len(all_cruises)} total cruises → {output_path}")
    return output_path


# ----------------------------
# CLI entry point
# ----------------------------

if __name__ == "__main__":
    api_key = os.getenv("RAPIDAPI_KEY")
    if not api_key:
        raise RuntimeError("RAPIDAPI_KEY not found in environment")

    payload = {
        "cruiseName": "Denali",
        "cruiseLineCodes": ["HA"],
        "shipCodes": ["WE"],
        "startPortCodes": ["CAVAN"],
        "endPortCodes": ["CAVAN"],
        "destinations": ["AK"],
        "earliestStartDate": "2026-05-01",
        "latestStartDate": "2026-05-31",
        "durationMin": 10,
        "durationMax": 14,
        "numberOfGuests": [2],
        "roomTypeCategoryCodes": ["B"],
        "cruiseTypes": ["OCEAN_TOUR"],
        "removeSoldOut": False,
        "sortBy": "departureDate",
        "sortOrder": "asc",
    }

    cache_cruises(api_key, payload, max_pages=None)  # Set to None for all pages, or a number to limit