"""GovCon Opportunity Finder: Extract + Transform steps of the pipeline.

    extract()   -> pull raw IT/cloud opportunities from SAM.gov (nationwide)
    transform() -> keep only biddable, still-open ones and reshape them into
                   the fields a business owner needs

Week 2 moves these same functions into AWS Lambda and adds a load() step
that writes to DynamoDB.

Usage:
    python pipeline.py            # calls SAM.gov (one request per NAICS code)
    python pipeline.py --offline  # re-runs transform on the last saved raw data, no API calls
"""
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

URL = "https://api.sam.gov/opportunities/v2/search"
DAYS_BACK = 7
PAGE_SIZE = 1000
MAX_REQUESTS = 8  # personal keys allow ~10/day; stop before hitting the limit

# NAICS industry codes for IT and cloud work
IT_NAICS = {
    "541511": "Custom Computer Programming",
    "541512": "Computer Systems Design",
    "541513": "Computer Facilities Management",
    "541519": "Other Computer Related Services",
    "518210": "Data Processing, Hosting (Cloud)",
}

# Notice types a business can still act on. Award notices etc. are already decided.
BIDDABLE_TYPES = {
    "Solicitation",
    "Combined Synopsis/Solicitation",
    "Presolicitation",
    "Sources Sought",
}

RAW_FILE = Path("data/raw_it.json")
CLEAN_FILE = Path("data/opportunities.json")


def extract(api_key):
    """Fetch every opportunity posted in the last DAYS_BACK days for each IT NAICS code."""
    today = date.today()
    base_params = {
        "api_key": api_key,
        "postedFrom": (today - timedelta(days=DAYS_BACK)).strftime("%m/%d/%Y"),
        "postedTo": today.strftime("%m/%d/%Y"),
        "limit": PAGE_SIZE,
    }
    records, requests_made = [], 0

    for naics in IT_NAICS:
        offset = 0
        while True:
            if requests_made >= MAX_REQUESTS:
                print(f"Stopped at {MAX_REQUESTS} requests to protect the daily limit.")
                return records
            params = {**base_params, "ncode": naics, "offset": offset}
            response = requests.get(URL, params=params, timeout=60)
            requests_made += 1
            if response.status_code != 200:
                # Print only the body. The URL contains the key.
                sys.exit(f"SAM.gov returned {response.status_code}: {response.text[:300]}")

            body = response.json()
            page = body.get("opportunitiesData", [])
            records.extend(page)
            print(f"  NAICS {naics}: got {len(page)} (total available {body.get('totalRecords')})")

            offset += PAGE_SIZE
            if offset >= body.get("totalRecords", 0):
                break

    print(f"Used {requests_made} API request(s).")
    return records


def _agency(path):
    """'DEPT OF DEFENSE.DEPT OF THE ARMY.AMC...' -> 'DEPT OF DEFENSE > DEPT OF THE ARMY'"""
    parts = (path or "").split(".")
    return " > ".join(p for p in parts[:2] if p) or "Not listed"


def _place(pop):
    """Nested placeOfPerformance object -> 'Fort Gordon, GA'"""
    if not pop:
        return "Not specified"
    city = (pop.get("city") or {}).get("name")
    state = (pop.get("state") or {}).get("code")
    return ", ".join(p for p in (city, state) if p) or "Not specified"


def _contact(contacts):
    """List of contacts -> the primary contact's name and email."""
    if not contacts:
        return {"name": None, "email": None}
    primary = next((c for c in contacts if c.get("type") == "primary"), contacts[0])
    return {"name": primary.get("fullName"), "email": primary.get("email")}


def _is_expired(deadline, now):
    if not deadline:
        return False  # no deadline listed: keep it rather than hide a possible lead
    try:
        parsed = datetime.fromisoformat(deadline)
    except ValueError:
        return False
    if parsed.tzinfo is None:  # some deadlines come without a time zone; assume UTC
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed < now


def transform(raw_records):
    """Filter to biddable, open IT opportunities and reshape to the digest fields."""
    now = datetime.now(timezone.utc)
    clean, seen = [], set()
    dropped = {"not biddable": 0, "expired": 0, "duplicate": 0}

    for r in raw_records:
        notice_id = r.get("noticeId")
        if notice_id in seen:
            dropped["duplicate"] += 1
            continue
        if r.get("type") not in BIDDABLE_TYPES:
            dropped["not biddable"] += 1
            continue
        if _is_expired(r.get("responseDeadLine"), now):
            dropped["expired"] += 1
            continue
        seen.add(notice_id)

        naics = r.get("naicsCode")
        clean.append({
            "notice_id": notice_id,  # unique ID: becomes the DynamoDB primary key
            "title": r.get("title"),
            "agency": _agency(r.get("fullParentPathName")),
            "type": r.get("type"),
            "response_deadline": r.get("responseDeadLine"),
            "posted_date": r.get("postedDate"),
            "naics_code": naics,
            "naics_description": IT_NAICS.get(naics, "Other"),
            "set_aside": r.get("typeOfSetAsideDescription") or "None",
            "place_of_performance": _place(r.get("placeOfPerformance")),
            "point_of_contact": _contact(r.get("pointOfContact")),
            "link": r.get("uiLink"),
        })

    print(f"Kept {len(clean)} of {len(raw_records)}. Dropped: {dropped}")
    return clean


def main():
    offline = "--offline" in sys.argv
    if offline:
        raw = json.loads(RAW_FILE.read_text())
        print(f"Offline mode: loaded {len(raw)} raw records from {RAW_FILE}")
    else:
        load_dotenv()
        api_key = os.environ.get("SAM_API_KEY", "").strip()
        if not api_key:
            sys.exit("SAM_API_KEY is empty. Paste your key into the .env file first.")
        raw = extract(api_key)
        RAW_FILE.parent.mkdir(exist_ok=True)
        RAW_FILE.write_text(json.dumps(raw, indent=2))

    clean = transform(raw)
    CLEAN_FILE.write_text(json.dumps(clean, indent=2))
    print(f"Saved {len(clean)} opportunities to {CLEAN_FILE}")


if __name__ == "__main__":
    main()
