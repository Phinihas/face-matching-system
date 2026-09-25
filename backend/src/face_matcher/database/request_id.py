"""
request_id.py — Timestamp-based integer ID generator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generates a 16-digit integer from the current datetime
(YYYYMMDDHHMMSSffffff → first 16 digits).
Matches the format used across all other PII projects.
"""

from datetime import datetime


def generate_request_id() -> int:
    """Return a unique 16-digit timestamp-based integer request ID."""
    return int(datetime.now().strftime("%Y%m%d%H%M%S%f")[:16])


def generate_short_request_id() -> str:
    """
    Return an 8-char hex string for endpoints that do NOT persist to the DB
    (e.g. /compare, /search).  These are transient log-only IDs.
    """
    import uuid
    return uuid.uuid4().hex[:8]
