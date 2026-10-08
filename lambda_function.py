"""AWS Lambda entry point: Extract (SAM.gov) -> Transform -> Load (DynamoDB) -> Enrich (Bedrock).

Configuration comes from Lambda environment variables, not code:
    SSM_PARAM_NAME  Parameter Store name of the SAM.gov key (SecureString)
    TABLE_NAME      DynamoDB table to write to
    DAYS_BACK       How many days of postings to pull (overlap is safe: writes are idempotent)
    MODEL_ID        Bedrock inference profile for the AI model
    MAX_SCORES      Cost guard: most Stage 1 scores per run
    MAX_DETAILS     Stage 2 analyses per run (each costs one SAM.gov request)
    MIN_DETAIL_SCORE  Only opportunities scoring at least this get a Stage 2 analysis
"""
import os
from datetime import datetime, timezone

import boto3  # AWS SDK for Python; preinstalled in the Lambda runtime

from enrich import Enricher, check_eligibility, fetch_description
from pipeline import _is_expired, extract, transform

SSM_PARAM_NAME = os.environ.get("SSM_PARAM_NAME", "/govcon/sam-api-key")
TABLE_NAME = os.environ.get("TABLE_NAME", "govcon-opportunities")
DAYS_BACK = int(os.environ.get("DAYS_BACK", "3"))
MODEL_ID = os.environ.get("MODEL_ID", "global.anthropic.claude-haiku-4-5-20251001-v1:0")
MAX_SCORES = int(os.environ.get("MAX_SCORES", "40"))
MAX_DETAILS = int(os.environ.get("MAX_DETAILS", "3"))
MIN_DETAIL_SCORE = int(os.environ.get("MIN_DETAIL_SCORE", "50"))

# Created once per container and reused across warm invocations
ssm = boto3.client("ssm")
table = boto3.resource("dynamodb").Table(TABLE_NAME)


def get_api_key():
    """Read and decrypt the SAM.gov key from Parameter Store."""
    response = ssm.get_parameter(Name=SSM_PARAM_NAME, WithDecryption=True)
    return response["Parameter"]["Value"]


def load(opportunities):
    """Upsert each opportunity by notice_id. Returns how many were new.

    Upsert = insert if new, update if it already exists. Running twice gives the
    same result as running once (idempotent), so overlapping days are harmless.
    first_seen is only set the first time; fields added later (AI summary in
    Week 3) are never overwritten because we only SET the pipeline's fields.
    """
    now = datetime.now(timezone.utc).isoformat()
    new_count = 0

    for opp in opportunities:
        fields = {k: v for k, v in opp.items() if k != "notice_id"}
        names = {f"#{k}": k for k in fields}  # '#' aliases avoid DynamoDB reserved words (e.g. "type")
        values = {f":{k}": v for k, v in fields.items()}
        updates = [f"#{k} = :{k}" for k in fields]
        updates += ["first_seen = if_not_exists(first_seen, :now)", "last_seen = :now"]
        values[":now"] = now

        result = table.update_item(
            Key={"notice_id": opp["notice_id"]},
            UpdateExpression="SET " + ", ".join(updates),
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
            ReturnValues="UPDATED_OLD",
        )
        if "first_seen" not in result.get("Attributes", {}):
            new_count += 1

    return new_count


def scan_all():
    """Read every item. Fine for a few thousand items; at larger scale use a Query on an index."""
    items, kwargs = [], {}
    while True:
        page = table.scan(**kwargs)
        items.extend(page["Items"])
        if "LastEvaluatedKey" not in page:
            return items
        kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]


def enrich(api_key):
    """Stage 1 on anything not yet scored, then Stage 2 on the best unsummarized items.

    Works from what's in the table (not just today's new items), so a failed
    run is picked up automatically by the next one.
    """
    now = datetime.now(timezone.utc)
    stamp = now.isoformat()
    enricher = Enricher(MODEL_ID)
    items = [i for i in scan_all() if not _is_expired(i.get("response_deadline"), now)]
    stats = {"scored": 0, "ineligible": 0, "detailed": 0, "ai_errors": 0}

    # Stage 1: quick score (metadata only)
    for item in [i for i in items if "fit_score" not in i][:MAX_SCORES]:
        eligible, reason = check_eligibility(item.get("set_aside"))
        try:
            if eligible:
                result = enricher.quick_score(item)
                stats["scored"] += 1
            else:
                result = {"fit_score": 0, "fit_reason": reason}
                stats["ineligible"] += 1
        except Exception as e:  # one bad item shouldn't stop the run
            print(f"  Stage 1 failed for {item['notice_id']}: {e}")
            stats["ai_errors"] += 1
            continue
        table.update_item(
            Key={"notice_id": item["notice_id"]},
            UpdateExpression="SET fit_score = :s, fit_reason = :r, scored_at = :t",
            ExpressionAttributeValues={":s": result["fit_score"], ":r": result["fit_reason"], ":t": stamp},
        )
        item.update(fit_score=result["fit_score"])

    # Stage 2: full analysis on the top unsummarized candidates
    candidates = sorted(
        (i for i in items if "ai_summary" not in i and int(i.get("fit_score", 0)) >= MIN_DETAIL_SCORE),
        key=lambda i: int(i["fit_score"]),
        reverse=True,
    )
    for item in candidates[:MAX_DETAILS]:
        description = fetch_description(item["notice_id"], api_key)
        if not description:
            continue
        try:
            result = enricher.full_analysis(item, description)
        except Exception as e:
            print(f"  Stage 2 failed for {item['notice_id']}: {e}")
            stats["ai_errors"] += 1
            continue
        table.update_item(
            Key={"notice_id": item["notice_id"]},
            UpdateExpression=(
                "SET ai_summary = :sm, what_they_want = :w, key_requirements = :k, "
                "fit_score = :s, fit_reasons = :fr, red_flags = :rf, summarized_at = :t"
            ),
            ExpressionAttributeValues={
                ":sm": result["summary"], ":w": result["what_they_want"],
                ":k": result["key_requirements"], ":s": result["fit_score"],
                ":fr": result["fit_reasons"], ":rf": result["red_flags"], ":t": stamp,
            },
        )
        stats["detailed"] += 1

    stats.update(
        input_tokens=enricher.input_tokens,
        output_tokens=enricher.output_tokens,
        est_ai_cost_usd=enricher.estimated_cost_usd(),
    )
    return stats


def lambda_handler(event, context):
    """Lambda calls this function. EventBridge passes the schedule event as `event`.

    Test event {"enrich_only": true} skips the SAM.gov search and only runs the AI
    stage on what's already in the table (saves 5 SAM.gov requests).
    """
    api_key = get_api_key()
    summary = {}
    if not (event or {}).get("enrich_only"):
        raw = extract(api_key, days_back=DAYS_BACK)
        clean = transform(raw)
        summary = {"fetched": len(raw), "kept": len(clean), "new": load(clean)}

    summary.update(enrich(api_key))
    print(f"Run complete: {summary}")  # shows up in CloudWatch Logs
    return summary
