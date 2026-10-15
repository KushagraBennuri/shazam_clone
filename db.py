"""MySQL connection pooling. Configure via environment variables:
DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_POOL_SIZE
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager

from mysql.connector import Error as MySQLError
from mysql.connector import pooling

_pool: pooling.MySQLConnectionPool | None = None
_lock = threading.Lock()


class DatabaseError(RuntimeError):
    """Raised when the database is unreachable or misconfigured."""


def _config() -> dict:
    return {
        "host": os.getenv("DB_HOST", "127.0.0.1"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
        "database": os.getenv("DB_NAME", "shazam_clone"),
        "charset": "utf8mb4",
        "autocommit": False,
        "connection_timeout": 10,
    }


def get_pool() -> pooling.MySQLConnectionPool:
    global _pool
    if _pool is None:
        with _lock:
            if _pool is None:
                try:
                    _pool = pooling.MySQLConnectionPool(
                        pool_name="shazam_pool",
                        pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
                        pool_reset_session=True,
                        **_config(),
                    )
                except MySQLError as exc:
                    raise DatabaseError(
                        f"Cannot connect to MySQL ({exc}). Did you run schema.sql "
                        "and set DB_* environment variables?"
                    ) from exc
    return _pool


def get_connection():
    """Borrow a connection; calling .close() returns it to the pool."""
    try:
        return get_pool().get_connection()
    except MySQLError as exc:
        raise DatabaseError(f"Connection pool error: {exc}") from exc


@contextmanager
def cursor(commit: bool = False, dictionary: bool = False):
    conn = get_connection()
    cur = conn.cursor(dictionary=dictionary)
    try:
        yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def check_connection() -> None:
    with cursor() as cur:
        cur.execute("SELECT 1")
        cur.fetchone()
