"""
Singleton Snowflake connection for TariffIQ (TARIFFIQ.RAW).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import snowflake.connector
from dotenv import load_dotenv
from snowflake.connector import SnowflakeConnection

logger = logging.getLogger(__name__)

_conn: Optional[SnowflakeConnection] = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_env() -> None:
    root = _project_root()
    env_file = root / ".env"
    if env_file.is_file():
        load_dotenv(env_file)
    load_dotenv()


def _connect_kwargs() -> dict:
    _load_env()
    kwargs: dict = {
        "user": os.environ["SNOWFLAKE_USER"],
        "password": os.environ["SNOWFLAKE_PASSWORD"],
        "account": os.environ["SNOWFLAKE_ACCOUNT"].strip(),
        "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
        "database": os.environ["SNOWFLAKE_DATABASE"],
        "schema": os.environ["SNOWFLAKE_SCHEMA"],
    }
    # If login returns 404, your account value is usually wrong/incomplete. Prefer the full
    # account identifier from Snowflake UI (e.g. xy12345.us-east-1.aws or orgname-accountname).
    region = os.environ.get("SNOWFLAKE_REGION", "").strip()
    if region:
        kwargs["region"] = region
    cloud = os.environ.get("SNOWFLAKE_CLOUD", "").strip().lower()
    if cloud in ("aws", "azure", "gcp"):
        kwargs["cloud"] = cloud
    host = os.environ.get("SNOWFLAKE_HOST", "").strip()
    if host:
        kwargs["host"] = host
    return kwargs


def _connect() -> SnowflakeConnection:
    return snowflake.connector.connect(**_connect_kwargs())


def _connection_usable(conn: Optional[SnowflakeConnection]) -> bool:
    if conn is None:
        return False
    try:
        return not conn.is_closed()
    except Exception:
        logger.warning("snowflake_connection_state_check_failed")
        return False


def get_connection() -> SnowflakeConnection:
    """
    Return the process-wide Snowflake connection, creating or replacing it if needed.
    """
    global _conn
    _load_env()
    if not _connection_usable(_conn):
        if _conn is not None:
            logger.info("snowflake_reconnecting")
        else:
            logger.info("snowflake_connecting")
        _conn = _connect()
    return _conn
