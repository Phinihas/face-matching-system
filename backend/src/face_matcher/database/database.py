"""
database.py — Database operations for FaceMatching
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Uses shared pii_status and pii_results tables.
Type identifier: 'FaceMatching'

Status code mapping:
    DB code  ←→  human-readable
    'p'      ←→  pending / processing
    's'      ←→  completed
    'e'      ←→  failed
    'c'      ←→  cancelled
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .db import get_db
from .request_id import generate_request_id

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
PROJECT = "FaceMatching"

# Map DB single-char code → human-readable string (for API responses)
_STATUS_TO_DISPLAY = {
    "p": "pending",
    "s": "completed",
    "e": "failed",
    "c": "cancelled",
}

# Map human-readable / legacy strings → DB single-char code
_STATUS_TO_DB = {
    "pending":    "p",
    "processing": "p",   # no separate "processing" state; stays 'p'
    "completed":  "s",
    "failed":     "e",
    "cancelled":  "c",
}


# ═══════════════════════════════════════════════════════════════════
#  DATA CLASS  —  returned by get_batch_job / get_all_batch_jobs
#  Mirrors the old BatchJob ORM object attribute names so that
#  main.py requires no changes.
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BatchJobResult:
    """
    Backward-compatible wrapper around pii_status + pii_results data.
    Attribute names intentionally match the old SQLAlchemy BatchJob model
    so that main.py endpoints need zero changes.
    """
    batch_id:         str                   # = str(request_id)
    status:           str                   # human-readable display string
    folder_path:      Optional[str] = None
    max_workers:      int           = 4
    created_at:       Optional[datetime] = None
    started_at:       Optional[datetime] = None   # mapped from created_at
    completed_at:     Optional[datetime] = None   # mapped from updated_at when done
    total_images:     int           = 0           # = total_count
    processed_images: int           = 0           # = processed_count
    error_message:    Optional[str] = None


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

def _row_to_batch_job(status_row: dict, request_row: dict = None) -> BatchJobResult:
    """
    Convert a pii_status row (+ optional pii_results row) into a
    BatchJobResult with backward-compatible attribute names.
    """
    db_status    = status_row["status"]
    display_status = _STATUS_TO_DISPLAY.get(db_status, db_status)

    # Derive started_at / completed_at from created_at / updated_at
    started_at    = status_row.get("created_at")      # job starts when created
    completed_at  = None
    if db_status in ("s", "e"):                        # finished states
        completed_at = status_row.get("updated_at")

    # Extract folder_path and max_workers from pii_results row
    folder_path = None
    max_workers = 4
    if request_row:
        folder_path = request_row.get("filepath")
        raw_result  = request_row.get("result")
        if raw_result:
            try:
                result_dict = (
                    json.loads(raw_result)
                    if isinstance(raw_result, str)
                    else raw_result
                )
                max_workers = result_dict.get("max_workers", 4)
            except (json.JSONDecodeError, TypeError):
                pass

    return BatchJobResult(
        batch_id         = str(status_row["request_id"]),
        status           = display_status,
        folder_path      = folder_path,
        max_workers      = max_workers,
        created_at       = status_row.get("created_at"),
        started_at       = started_at,
        completed_at     = completed_at,
        total_images     = status_row.get("total_count",     0),
        processed_images = status_row.get("processed_count", 0),
        error_message    = status_row.get("error_message"),
    )


# ═══════════════════════════════════════════════════════════════════
#  PUBLIC API  —  same function signatures as the old database.py
# ═══════════════════════════════════════════════════════════════════

def create_batch_job(folder_path: str, max_workers: int = 4) -> str:
    """
    Create a new batch job.

    Inserts a row into pii_status (type='FaceMatching', status='p')
    and a row into pii_results (filepath=folder_path).

    Returns:
        request_id as a string  (used as batch_id throughout the app)
    """
    request_id = generate_request_id()

    with get_db() as conn:
        with conn.cursor() as cur:
            # ── pii_status row ───────────────────────────────────
            cur.execute(
                """
                INSERT INTO pii_status
                    (request_id, type, status, processed_count, total_count, request_type)
                VALUES
                    (%s, %s, 'p', 0, 0, 'bulk')
                """,
                (request_id, PROJECT),
            )

            # ── pii_results row (stores folder_path + max_workers) ─
            result_id = generate_request_id()
            cur.execute(
                """
                INSERT INTO pii_results
                    (id, request_id, result, filepath)
                VALUES
                    (%s, %s, %s, %s)
                """,
                (
                    result_id,
                    request_id,
                    json.dumps({"max_workers": max_workers}),
                    folder_path,
                ),
            )

    logger.info(f"[{request_id}] Batch job created — folder={folder_path}, workers={max_workers}")
    return str(request_id)


def update_batch_status(batch_id: str, status: str, **kwargs):
    """
    Update the status of a batch job in pii_status.

    Args:
        batch_id : the request_id (as string)
        status   : 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled'
                   (legacy strings are mapped to single-char DB codes automatically)
        **kwargs:
            total_images     (int) → total_count
            processed_images (int) → processed_count
            error_message    (str) → error_message
    """
    db_status = _STATUS_TO_DB.get(status, status)   # already a code? pass through

    fields = ["status = %s", "updated_at = NOW()"]
    values = [db_status]

    if "total_images" in kwargs:
        fields.append("total_count = %s")
        values.append(kwargs["total_images"])

    if "processed_images" in kwargs:
        fields.append("processed_count = %s")
        values.append(kwargs["processed_images"])

    if "error_message" in kwargs and kwargs["error_message"] is not None:
        fields.append("error_message = %s")
        values.append(str(kwargs["error_message"])[:65535])

    # WHERE clause params
    values.extend([int(batch_id), PROJECT])

    sql = f"UPDATE pii_status SET {', '.join(fields)} WHERE request_id = %s AND type = %s"

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, values)
        logger.info(f"[{batch_id}] Status updated → {status} ({db_status})")
    except Exception as e:
        logger.warning(f"[{batch_id}] Failed to update status to {status}: {e}")


def get_batch_job(batch_id: str) -> Optional[BatchJobResult]:
    """
    Fetch a single batch job by its request_id.

    Returns a BatchJobResult (same attributes as old BatchJob ORM model)
    or None if not found.
    """
    try:
        rid = int(batch_id)
    except (ValueError, TypeError):
        logger.warning(f"get_batch_job: invalid batch_id '{batch_id}'")
        return None

    with get_db() as conn:
        with conn.cursor() as cur:
            # Fetch status row
            cur.execute(
                "SELECT * FROM pii_status WHERE request_id = %s AND type = %s",
                (rid, PROJECT),
            )
            status_row = cur.fetchone()

            if not status_row:
                return None

            # Fetch request details row (folder_path, max_workers)
            cur.execute(
                "SELECT * FROM pii_results WHERE request_id = %s ORDER BY created_at ASC LIMIT 1",
                (rid,),
            )
            request_row = cur.fetchone()

    return _row_to_batch_job(status_row, request_row)


def get_all_batch_jobs() -> list:
    """
    Fetch all FaceMatching batch jobs, newest first.

    Returns a list of BatchJobResult objects.
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.*, r.filepath, r.result AS req_result
                FROM pii_status s
                LEFT JOIN pii_results r ON r.request_id = s.request_id
                WHERE s.type = %s AND s.request_type = 'bulk'
                ORDER BY s.created_at DESC
                """,
                (PROJECT,),
            )
            rows = cur.fetchall()

    jobs = []
    for row in rows:
        # Separate status fields from pii_results fields
        request_row = None
        if row.get("filepath") is not None:
            request_row = {
                "filepath": row["filepath"],
                "result":   row.get("req_result"),
            }
        jobs.append(_row_to_batch_job(row, request_row))

    return jobs


# ── get_db passthrough (used by main.py lifespan / health checks) ─
def get_db_session():
    """
    Yields a raw DB connection for direct use.
    Prefer the helpers above; use this only for one-off queries.
    """
    return get_db()


def cancel_batch_job(batch_id: str) -> dict:
    """
    Cancel a pending or in-progress batch job.

    Returns:
        {"success": True, "processed_count": N, "total_count": N}
        {"error": "not_found" | "already_completed" | "already_failed" | "already_cancelled"}
    """
    try:
        rid = int(batch_id)
    except (ValueError, TypeError):
        return {"error": "not_found"}

    # Fetch current status
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status, processed_count, total_count FROM pii_status WHERE request_id = %s AND type = %s",
                (rid, PROJECT),
            )
            row = cur.fetchone()

    if not row:
        return {"error": "not_found"}
    if row["status"] == "s":
        return {"error": "already_completed"}
    if row["status"] == "e":
        return {"error": "already_failed"}
    if row["status"] == "c":
        return {"error": "already_cancelled"}

    # Mark as cancelled
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE pii_status SET status = 'c', updated_at = NOW() WHERE request_id = %s AND type = %s",
                (rid, PROJECT),
            )

    logger.info(f"[{batch_id}] Job cancelled.")
    return {
        "success":         True,
        "processed_count": row["processed_count"],
        "total_count":     row["total_count"],
    }


def create_single_upload_job(image_filename: str) -> str:
    """
    Create a pii_status row for a single_upload request so that it appears
    in /jobs alongside bulk_upload batches.

    Returns:
        request_id as a string (16-digit timestamp integer)
    """
    request_id = generate_request_id()

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pii_status
                    (request_id, type, status, processed_count, total_count, request_type)
                VALUES
                    (%s, %s, 'p', 0, 1, 'single')
                """,
                (request_id, PROJECT),
            )

            result_id = generate_request_id()
            cur.execute(
                """
                INSERT INTO pii_results
                    (id, request_id, result, filepath)
                VALUES
                    (%s, %s, %s, %s)
                """,
                (
                    result_id,
                    request_id,
                    '{"job_type": "single_upload"}',
                    image_filename,
                ),
            )

    logger.info(f"[{request_id}] Single upload job created — file={image_filename}")
    return str(request_id)


def complete_single_upload_job(request_id: str, success: bool, error_msg: str = None):
    """
    Mark a single_upload job as completed or failed in pii_status.
    """
    status = "completed" if success else "failed"
    update_batch_status(
        request_id,
        status,
        total_images=1,
        processed_images=1 if success else 0,
        **({"error_message": error_msg} if error_msg else {}),
    )
def _db_insert_compare_pending(request_id: int, image1: str, image2: str) -> None:
    """Insert pending row for a compare request."""
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO pii_status
                        (request_id, type, status, processed_count, total_count, request_type)
                    VALUES
                        (%s, %s, 'p', 0, 1, 'single')
                    """,
                    (request_id, PROJECT),
                )
        logger.info(f"[{request_id}] compare pending inserted")
    except Exception as e:
        logger.warning(f"[DB] _db_insert_compare_pending failed: {e}")


def _db_insert_compare_success(
    request_id: int,
    image1: str,
    image2: str,
    result: str,
    match_percentage: float,
    execution_time: float,
) -> None:
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE pii_status
                    SET status = 's', processed_count = 1, updated_at = NOW()
                    WHERE request_id = %s
                    """,
                    (request_id,),
                )
                result_id = generate_request_id()
                cur.execute(
                    """
                    INSERT INTO pii_results
                        (id, request_id, filepath, result)
                    VALUES
                        (%s, %s, %s, %s)
                    """,
                    (
                        result_id,
                        request_id,
                        f"{image1}|{image2}",
                        json.dumps({
                            "result":           result,
                            "match_percentage": match_percentage,
                            "execution_time":   execution_time,
                            "image1":           image1,
                            "image2":           image2,
                        }),
                    ),
                )
        logger.info(f"[{request_id}] compare success updated")
    except Exception as e:
        logger.warning(f"[DB] _db_insert_compare_success failed: {e}")

def _db_insert_compare_error(
    request_id: int,
    image1: str,
    image2: str,
    error_msg: str,
) -> None:
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE pii_status
                    SET status = 'e', error_message = %s, updated_at = NOW()
                    WHERE request_id = %s
                    """,
                    (error_msg[:65535], request_id),
                )
        logger.info(f"[{request_id}] compare error updated")
    except Exception as e:
        logger.warning(f"[DB] _db_insert_compare_error failed: {e}")