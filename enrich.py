"""Week 3: AI enrichment with Claude Haiku 4.5 on Amazon Bedrock.

Two-stage funnel (cheap filter first, expensive analysis on the shortlist):
    Stage 1  quick_score()    every new opportunity, metadata only -> fit_score + one-line reason
    Stage 2  full_analysis()  top few by score, full SAM.gov description -> plain-English summary

Set-aside eligibility is checked with a plain rule *before* any AI call: if the company
legally can't bid, there's nothing for the model to judge, and the call would be wasted money.
"""
import html
import json
import re
from pathlib import Path

import anthropic
import requests

PROFILE = (Path(__file__).parent / "company_profile.md").read_text()

# Set-asides this company does NOT qualify for (see company_profile.md).
# "Sole Source" means the agency already picked a specific vendor.
INELIGIBLE_MARKERS = ["8(a)", "SDVOSB", "Service-Disabled", "Veteran", "WOSB", "Women", "HUBZone", "Sole Source"]

DESCRIPTION_URL = "https://api.sam.gov/prod/opportunities/v1/noticedesc"
MAX_DESCRIPTION_CHARS = 20_000  # cost guard: ~5K tokens; longer notices are cut and flagged

SYSTEM_PROMPT = f"""You evaluate U.S. federal contract opportunities for a small IT company.
Your readers are busy small-business owners who decide in seconds whether an opportunity
is worth their time, so be concrete, plain-spoken, and honest about poor fits.

Score fit from 0 to 100:
  80-100  strong match to the company's strengths, realistic to win
  50-79   partial match or notable uncertainty
  20-49   weak match
  0-19    outside what the company does, or work it can't take on

The company profile:
{PROFILE}"""

QUICK_TOOL = {
    "name": "record_fit",
    "description": "Record the fit assessment for one opportunity.",
    "input_schema": {
        "type": "object",
        "properties": {
            "fit_score": {"type": "integer", "minimum": 0, "maximum": 100},
            "fit_reason": {"type": "string", "description": "One sentence explaining the score."},
        },
        "required": ["fit_score", "fit_reason"],
    },
}

FULL_TOOL = {
    "name": "record_analysis",
    "description": "Record the full analysis of one opportunity.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "2-3 plain-English sentences: what this is and why it matters."},
            "what_they_want": {"type": "string", "description": "The actual work or product being bought, in one sentence."},
            "key_requirements": {
                "type": "array", "items": {"type": "string"},
                "description": "Up to 5 requirements a bidder must meet (certifications, clearances, location, experience).",
            },
            "fit_score": {"type": "integer", "minimum": 0, "maximum": 100},
            "fit_reasons": {"type": "string", "description": "2-3 sentences on why the score is what it is."},
            "red_flags": {
                "type": "array", "items": {"type": "string"},
                "description": "Anything that would stop or seriously hurt this company's bid. Empty if none.",
            },
        },
        "required": ["summary", "what_they_want", "key_requirements", "fit_score", "fit_reasons", "red_flags"],
    },
}


class Enricher:
    def __init__(self, model_id):
        self.model_id = model_id
        self.client = anthropic.AnthropicBedrock()  # region comes from AWS_REGION, set automatically in Lambda
        self.input_tokens = 0
        self.output_tokens = 0

    def _call(self, tool, user_text, max_tokens):
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},  # always answer through the tool's schema
            messages=[{"role": "user", "content": user_text}],
        )
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens

        result = next((b.input for b in response.content if b.type == "tool_use"), None)
        if not result or "fit_score" not in result:
            raise ValueError(f"Model returned no usable assessment (stop_reason={response.stop_reason})")
        result["fit_score"] = max(0, min(100, int(result["fit_score"])))
        return result

    def quick_score(self, opp):
        """Stage 1: score from metadata only. Costs no SAM.gov requests."""
        return self._call(QUICK_TOOL, "Assess this opportunity:\n" + _metadata(opp), max_tokens=1024)

    def full_analysis(self, opp, description):
        """Stage 2: detailed analysis using the full notice text."""
        text = (
            "Assess this opportunity using its full description.\n\n"
            f"{_metadata(opp)}\n\n<description>\n{description}\n</description>"
        )
        return self._call(FULL_TOOL, text, max_tokens=2048)

    def estimated_cost_usd(self, input_price_per_m=1.0, output_price_per_m=5.0):
        """Rough cost of this run's AI calls (Haiku 4.5 list prices; Bedrock's may differ slightly)."""
        return round(self.input_tokens / 1e6 * input_price_per_m + self.output_tokens / 1e6 * output_price_per_m, 4)


def check_eligibility(set_aside):
    """Hard rule, no AI: can this company legally bid?"""
    for marker in INELIGIBLE_MARKERS:
        if marker.lower() in (set_aside or "").lower():
            return False, f"Not eligible: reserved set-aside ({set_aside})."
    return True, None


def fetch_description(notice_id, api_key):
    """One SAM.gov request. Returns the notice text as plain text (HTML stripped), or None."""
    response = requests.get(DESCRIPTION_URL, params={"noticeid": notice_id, "api_key": api_key}, timeout=30)
    if response.status_code != 200:
        print(f"  Description fetch for {notice_id} failed: HTTP {response.status_code}")
        return None
    raw = response.json().get("description") or ""
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_DESCRIPTION_CHARS:
        print(f"  Description for {notice_id} truncated from {len(text)} chars")
        text = text[:MAX_DESCRIPTION_CHARS] + " [...truncated]"
    return text or None


def _metadata(opp):
    fields = {
        "Title": opp.get("title"),
        "Agency": opp.get("agency"),
        "Notice type": opp.get("type"),
        "Industry (NAICS)": f"{opp.get('naics_code')} {opp.get('naics_description')}",
        "Set-aside": opp.get("set_aside"),
        "Place of performance": opp.get("place_of_performance"),
        "Response deadline": opp.get("response_deadline") or "Not listed",
    }
    return json.dumps(fields, indent=2)
