"""
db.py — Database connection manager for FaceMatching
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reads host/port/database from settings.ini (same directory).
Reads credentials from .env (DB_USER, DB_PASSWORD).

Place this file at:
    backend/src/face_matcher/database/db.py
"""

import pymysql
import pymysql.cursors
from contextlib import contextmanager
import logging
from face_matcher import settings

logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host":        settings.DB_HOST,
    "port":        settings.DB_PORT,
    "user":        settings.DB_USER,
    "password":    settings.DB_PASSWORD,
    "database":    settings.DB_NAME,
    "charset":     settings.DB_CHARSET,
    "cursorclass": pymysql.cursors.DictCursor,
}



@contextmanager
def get_db():
    """
    Context manager that yields a DB connection.
    Auto-commits on success, rolls back on error, always closes.

    Usage:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
    """
    conn = None
    try:
        conn = pymysql.connect(**DB_CONFIG)
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"DB error: {e}")
        raise
    finally:
        if conn:
            conn.close()