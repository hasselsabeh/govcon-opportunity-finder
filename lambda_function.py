"""AWS Lambda entry point: Extract (SAM.gov) -> Transform -> Load (DynamoDB).

Configuration comes from Lambda environment variables, not code:
    SSM_PARAM_NAME  Parameter Store name of the SAM.gov key (SecureString)
    TABLE_NAME      DynamoDB table to write to
    DAYS_BACK       How many days of postings to pull (overlap is safe: writes are idempotent)
"""
import os
from datetime import datetime, timezone

import boto3  # AWS SDK for Python; preinstalled in the Lambda runtime

from pipeline import extract, transform

SSM_PARAM_NAME = os.environ.get("SSM_PARAM_NAME", "/govcon/sam-api-key")
TABLE_NAME = os.environ.get("TABLE_NAME", "govcon-opportunities")
DAYS_BACK = int(os.environ.get("DAYS_BACK", "3"))

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


def lambda_handler(event, context):
    """Lambda calls this function. EventBridge passes the schedule event as `event`."""
    api_key = get_api_key()
    raw = extract(api_key, days_back=DAYS_BACK)
    clean = transform(raw)
    new_count = load(clean)

    summary = {"fetched": len(raw), "kept": len(clean), "new": new_count}
    print(f"Run complete: {summary}")  # shows up in CloudWatch Logs
    return summary
