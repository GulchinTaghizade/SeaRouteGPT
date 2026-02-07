import json
import os
from pathlib import Path
from datetime import datetime
import requests
from dotenv import load_dotenv
import time

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

    # Define multiple search payloads to get diverse cruise data
    search_payloads = [
        #Alaska peak season
        {
            "destinations": ["AK"],
            "earliestStartDate": "2026-06-01",
            "latestStartDate": "2026-08-31",
            "durationMin": 7,
            "durationMax": 14,
            "numberOfGuests": [2]
        },
        #Arctic - Shoulder
        {
            "destinations": ["AR"],
            "earliestStartDate": "2026-05-01",
            "latestStartDate": "2026-06-15",
            "durationMin": 7,
            "durationMax": 14,
            "numberOfGuests": [2]
        },
        #Caribbean - Winter Peak
        {
            "destinations": ["CA"],
            "earliestStartDate": "2026-02-07",
            "latestStartDate": "2026-03-31",
            "durationMin": 5,
            "durationMax": 10,
            "numberOfGuests": [2]
        },

        #Caribbean - Summer Low Season
        {
            "destinations": ["CA"],
            "earliestStartDate": "2026-08-01",
            "latestStartDate": "2026-10-15",
            "durationMin": 5,
            "durationMax": 9,
            "numberOfGuests": [2]
        },
        #Europe - Summer Peak
        {
            "destinations": ["EA"],
            "earliestStartDate": "2026-06-01",
            "latestStartDate": "2026-09-15",
            "durationMin": 7,
            "durationMax": 14,
            "numberOfGuests": [2]
        },
        #Europe - Shoulder
        {
            "destinations": ["EA"],
            "earliestStartDate": "2026-04-01",
            "latestStartDate": "2026-05-15",
            "durationMin": 7,
            "durationMax": 12,
            "numberOfGuests": [2]
        },
        #Mediterranean - Classic
        {
            "destinations": ["MA"],
            "earliestStartDate": "2026-05-01",
            "latestStartDate": "2026-09-30",
            "durationMin": 7,
            "durationMax": 12,
            "numberOfGuests": [2]
        },
        # Asia - Long-haul
        {
            "destinations": ["AS"],
            "earliestStartDate": "2026-10-01",
            "latestStartDate": "2026-12-15",
            "durationMin": 10,
            "durationMax": 21,
            "numberOfGuests": [2]
        },
        #Antarctida - Expedition
        {
            "destinations": ["AN"],
            "earliestStartDate": "2026-11-01",
            "latestStartDate": "2027-02-28",
            "durationMin": 10,
            "durationMax": 20,
            "numberOfGuests": [2]
        },
        #South America - Seasonal FLip
        {
            "destinations": ["SA"],
            "earliestStartDate": "2026-11-01",
            "latestStartDate": "2027-03-31",
            "durationMin": 7,
            "durationMax": 16,
            "numberOfGuests": [2]
        },
        #Transatlantic - Repositioning
        {
            "destinations": ["TC"],
            "earliestStartDate": "2026-04-01",
            "latestStartDate": "2026-06-30",
            "durationMin": 12,
            "durationMax": 21,
            "numberOfGuests": [2]
        },
        # World - Control group
        {
            "destinations": ["WO"],
            "earliestStartDate": "2026-02-07",
            "latestStartDate": "2026-12-31",
            "durationMin": 7,
            "durationMax": 30,
            "numberOfGuests": [2]
        }

    ]

    # Collect all cruises from multiple searches
    all_cruises_combined = []
    seen_cruise_ids = set()

    for i, payload in enumerate(search_payloads, 1):
        print(f"\n{'='*60}")
        print(f"SEARCH {i}/{len(search_payloads)}: {payload.get('destinations', 'N/A')} | Guests: {payload.get('numberOfGuests', 'N/A')}")
        print(f"{'='*60}")

        headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": RAPIDAPI_HOST,
            "Content-Type": "application/json",
        }

        page = 1
        max_pages_for_search = 10  # Limit pages per search to respect rate limits

        while page <= max_pages_for_search:
            payload["page"] = page
            payload["pageSize"] = 10  # API limit is 10

            try:
                print(f"  ⏳ Fetching page {page}...", end="", flush=True)
                response = requests.post(API_URL, headers=headers, json=payload, timeout=10)

                if response.status_code == 429:
                    # Rate limited - wait and retry
                    print(f"\r  ⏳ Rate limit hit. Waiting 60 seconds before retry...     ")
                    time.sleep(60)
                    continue
                elif response.status_code != 200:
                    print(f"\r  ❌ STATUS: {response.status_code}                           ")
                    print(f"  RESPONSE: {response.text}")
                    break

                data = response.json()
                cruises = data.get("data", [])

                # Only add cruises we haven't seen before (by cruiseId)
                new_count = 0
                for cruise in cruises:
                    cruise_id = cruise.get("cruiseId")
                    if cruise_id and cruise_id not in seen_cruise_ids:
                        all_cruises_combined.append(cruise)
                        seen_cruise_ids.add(cruise_id)
                        new_count += 1

                total_pages = data.get("total_pages", 1)
                print(f"\r  [PAGE {page}/{min(total_pages, max_pages_for_search)}] Fetched {len(cruises)} cruises (new: {new_count})         ")

                if page >= total_pages or page >= max_pages_for_search:
                    break

                page += 1
                # Respect rate limit with 5 seconds delay
                print(f"  ⏳ Waiting 5 seconds before next request...", end="", flush=True)
                time.sleep(5)
                print(f"\r" + " " * 50 + "\r", end="", flush=True)

            except requests.exceptions.Timeout:
                print(f"\r  ⏱️  Request timeout. Waiting 15 seconds...                   ")
                time.sleep(15)
            except requests.exceptions.RequestException as e:
                print(f"\r  ❌ Request error: {e}                                      ")
                break

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RAW_DATA_DIR / f"cruise_snapshot_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump({"data": all_cruises_combined, "total_results": len(all_cruises_combined)}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"[OK] Cached {len(all_cruises_combined)} UNIQUE total cruises → {output_path}")
    print(f"{'='*60}")
