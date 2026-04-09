"""
Idempotent HTS load into Snowflake (INSERT ... WHERE NOT EXISTS), two USITC passes.

Used by Airflow DAG `hts_ingest`; batch size 500 for Day 3 spec.
"""

from __future__ import annotations

import structlog
from typing import Any, Dict, List, Tuple

from ingestion.load_hts_to_snowflake import raw_row_to_params
from ingestion.usitc_client import _get_export_list
from snowflake.connection import get_connection

logger = structlog.get_logger(__name__)

BATCH_SIZE = 500

CREATE_STAGING_SQL = """
CREATE OR REPLACE TEMPORARY TABLE hts_ingest_staging (
    hts_id VARCHAR(32),
    hts_code VARCHAR(32),
    stat_suffix VARCHAR(2),
    chapter VARCHAR(2),
    level VARCHAR(20),
    indent_level INTEGER,
    description TEXT,
    general_rate VARCHAR(100),
    special_rate VARCHAR(500),
    other_rate VARCHAR(100),
    units VARCHAR(50),
    footnotes_json VARCHAR(16777216),
    is_chapter99 BOOLEAN,
    is_header_row BOOLEAN,
    raw_json VARCHAR(16777216)
)
"""

TRUNCATE_STAGING_SQL = "TRUNCATE TABLE hts_ingest_staging"

STAGING_INSERT_SQL = """
INSERT INTO hts_ingest_staging (
    hts_id, hts_code, stat_suffix, chapter, level, indent_level,
    description, general_rate, special_rate, other_rate, units,
    footnotes_json, is_chapter99, is_header_row, raw_json
) VALUES (
    %(hts_id)s, %(hts_code)s, %(stat_suffix)s, %(chapter)s, %(level)s, %(indent_level)s,
    %(description)s, %(general_rate)s, %(special_rate)s, %(other_rate)s, %(units)s,
    %(footnotes_json)s, %(is_chapter99)s, %(is_header_row)s, %(raw_json)s
)
"""

INSERT_WHERE_NOT_EXISTS_SQL = """
INSERT INTO HTS_CODES (
    hts_id,
    hts_code,
    stat_suffix,
    chapter,
    level,
    indent_level,
    description,
    general_rate,
    special_rate,
    other_rate,
    units,
    footnotes,
    is_chapter99,
    is_header_row,
    raw_json,
    loaded_at
)
SELECT
    s.hts_id,
    s.hts_code,
    s.stat_suffix,
    s.chapter,
    s.level,
    s.indent_level,
    s.description,
    s.general_rate,
    s.special_rate,
    s.other_rate,
    s.units,
    PARSE_JSON(s.footnotes_json),
    s.is_chapter99,
    s.is_header_row,
    PARSE_JSON(s.raw_json),
    CURRENT_TIMESTAMP()
FROM hts_ingest_staging s
WHERE NOT EXISTS (
    SELECT 1 FROM HTS_CODES t WHERE t.hts_code = s.hts_code
)
"""


def fetch_two_pass_raw() -> Tuple[List[dict], List[dict]]:
    """USITC schedule (0101–9999) and Chapter 99 window (9903–9904)."""
    main = _get_export_list("0101", "9999")
    ch99 = _get_export_list("9903", "9904")
    return main, ch99


def merge_rows_two_passes(main_raw: List[dict], ch99_raw: List[dict]) -> Tuple[List[dict[str, Any]], int]:
    """
    Parse both passes; merge by hts_code with Chapter 99 pass winning (is_chapter99 forced True).
    Returns (ordered param dicts, skipped_empty_htsno_count).
    """
    skipped = 0
    by_code: Dict[str, dict[str, Any]] = {}

    for raw in main_raw:
        p = raw_row_to_params(raw)
        if p is None:
            skipped += 1
            continue
        by_code[p["hts_code"]] = p

    for raw in ch99_raw:
        p = raw_row_to_params(raw)
        if p is None:
            skipped += 1
            continue
        p = dict(p)
        p["is_chapter99"] = True
        by_code[p["hts_code"]] = p

    return list(by_code.values()), skipped


def load_batches_idempotent(rows: List[dict[str, Any]], *, batch_size: int = BATCH_SIZE) -> int:
    """
    Insert rows in batches using staging + WHERE NOT EXISTS. Returns total new rows inserted.
    """
    if not rows:
        return 0

    conn = get_connection()
    total_loaded = 0
    try:
        cur = conn.cursor()
        cur.execute(CREATE_STAGING_SQL)

        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            cur.execute(TRUNCATE_STAGING_SQL)
            cur.executemany(STAGING_INSERT_SQL, chunk)
            cur.execute(INSERT_WHERE_NOT_EXISTS_SQL)
            rc = cur.rowcount
            if rc is not None and rc >= 0:
                total_loaded += rc
            else:
                logger.warning(
                    "hts_insert_rowcount_unknown",
                    batch_start=i,
                    batch_len=len(chunk),
                )

        conn.commit()
        return total_loaded
    except Exception:
        conn.rollback()
        logger.exception("hts_idempotent_load_failed")
        raise


def run_hts_ingest_airflow() -> Dict[str, int]:
    """
    Entry point for Airflow: two-pass fetch, merge, idempotent load, structlog summary.
    """
    main_raw, ch99_raw = fetch_two_pass_raw()
    total_fetched = len(main_raw) + len(ch99_raw)

    merged, skipped = merge_rows_two_passes(main_raw, ch99_raw)
    total_parsed = len(merged)

    total_loaded = load_batches_idempotent(merged)

    logger.info(
        "hts_ingest_metrics",
        total_fetched=total_fetched,
        total_parsed=total_parsed,
        total_loaded=total_loaded,
        total_skipped=skipped,
    )
    return {
        "total_fetched": total_fetched,
        "total_parsed": total_parsed,
        "total_loaded": total_loaded,
        "total_skipped": skipped,
    }
