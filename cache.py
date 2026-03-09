"""
SQLite cache layer.
Stores raw API responses so historical data is never re-fetched unnecessarily.
"""

import sqlite3
import json
import os
import time
from config import CACHE_DIR, CACHE_DB, CACHE_TTL_HOURS


def _ensure_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _connect() -> sqlite3.Connection:
    _ensure_dir()
    conn = sqlite3.connect(CACHE_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cache (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            fetched_at REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def get(key: str):
    """Return cached value (parsed JSON) or None if missing/expired."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT value, fetched_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        value_json, fetched_at = row
        age_hours = (time.time() - fetched_at) / 3600
        if age_hours > CACHE_TTL_HOURS:
            return None
        return json.loads(value_json)
    finally:
        conn.close()


def get_permanent(key: str):
    """Return cached value (parsed JSON) or None if missing — no TTL check."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT value FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])
    finally:
        conn.close()


def set(key: str, value) -> None:
    """Store value (serialised as JSON) with current timestamp."""
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO cache (key, value, fetched_at)
            VALUES (?, ?, ?)
            """,
            (key, json.dumps(value), time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def invalidate(key: str) -> None:
    """Remove a single cache entry."""
    conn = _connect()
    try:
        conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        conn.commit()
    finally:
        conn.close()


def clear_all() -> None:
    """Wipe the entire cache (useful for testing)."""
    conn = _connect()
    try:
        conn.execute("DELETE FROM cache")
        conn.commit()
    finally:
        conn.close()
