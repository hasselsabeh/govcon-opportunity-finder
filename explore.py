"""Week 1: make ONE call to SAM.gov and save the raw response to data/raw_sample.json.

Personal SAM.gov keys allow roughly 10 requests per day, so we call the API once
and do all further development against the saved file (caching).
"""
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env into environment variables
API_KEY = os.environ.get("SAM_API_KEY", "").strip()
if not API_KEY:
    sys.exit("SAM_API_KEY is empty. Paste your key into the .env file first.")

URL = "https://api.sam.gov/opportunities/v2/search"

today = date.today()
params = {
    "api_key": API_KEY,
    "postedFrom": (today - timedelta(days=30)).strftime("%m/%d/%Y"),
    "postedTo": today.strftime("%m/%d/%Y"),
    "state": "MI",  # place of performance: Michigan
    "limit": 1000,  # max records per page
    "offset": 0,  # pagination: start at record 0
}

response = requests.get(URL, params=params, timeout=60)
print("HTTP status:", response.status_code)

if response.status_code != 200:
    # Print only the body. The full URL contains the key, so don't print it.
    print(response.text[:500])
    sys.exit(1)

data = response.json()
Path("data").mkdir(exist_ok=True)
Path("data/raw_sample.json").write_text(json.dumps(data, indent=2))

print("Total matching records:", data.get("totalRecords"))
print("Records in this page:  ", len(data.get("opportunitiesData", [])))
print("Saved to data/raw_sample.json")
