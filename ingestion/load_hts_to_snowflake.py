"""
Fetch HTS rows from USITC exportList and load into TARIFFIQ.RAW.HTS_CODES.

Usage (from repo root):
  PYTHONPATH=. python -m ingestion.load_hts_to_snowflake
  PYTHONPATH=. python -m ingestion.load_hts_to_snowflake --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from typing import Any, List, Optional

from ingestion.usitc_client import _get_export_list
from snowflake.connection import get_connection

logger = logging.getLogger(__name__)

# Staging uses only types allowed in bulk VALUES; VARIANT is built via INSERT...SELECT PARSE_JSON.
CREATE_STAGING_SQL = """
CREATE OR REPLACE TEMPORARY TABLE hts_bulk_load_staging (
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

TRUNCATE_STAGING_SQL = "TRUNCATE TABLE hts_bulk_load_staging"

STAGING_INSERT_SQL = """
INSERT INTO hts_bulk_load_staging (
    hts_id, hts_code, stat_suffix, chapter, level, indent_level,
    description, general_rate, special_rate, other_rate, units,
    footnotes_json, is_chapter99, is_header_row, raw_json
) VALUES (
    %(hts_id)s, %(hts_code)s, %(stat_suffix)s, %(chapter)s, %(level)s, %(indent_level)s,
    %(description)s, %(general_rate)s, %(special_rate)s, %(other_rate)s, %(units)s,
    %(footnotes_json)s, %(is_chapter99)s, %(is_header_row)s, %(raw_json)s
)
"""

INSERT_FROM_STAGING_SQL = """
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
FROM hts_bulk_load_staging s
"""

BATCH_SIZE = 3000


def _ensure_hts_column_width(cur) -> None:
    """Widen HTS_CODES keys if the table was created with legacy VARCHAR(12)."""
    alters = (
        "ALTER TABLE HTS_CODES ALTER COLUMN hts_code SET DATA TYPE VARCHAR(32)",
        "ALTER TABLE HTS_CODES ALTER COLUMN hts_id SET DATA TYPE VARCHAR(32)",
    )
    for sql in alters:
        try:
            cur.execute(sql)
        except Exception as e:
            logger.warning("hts_alter_skipped sql=%s err=%s", sql[:60], e)


def _chapter_from_htsno(htsno: str) -> Optional[str]:
    """First two digits of the tariff chapter (e.g. 01, 99)."""
    if not htsno:
        return None
    head = htsno.split(".", 1)[0].strip()
    digits = re.sub(r"\D", "", head)
    if len(digits) >= 2:
        return digits[:2]
    if len(digits) == 1:
        return digits.zfill(2)
    return None


def _indent_level(raw: dict) -> Optional[int]:
    ind = raw.get("indent")
    if ind is None or ind == "":
        return None
    try:
        return int(str(ind).strip())
    except ValueError:
        return None


def _units_string(raw: dict, max_len: int = 50) -> str:
    units = raw.get("units")
    if units is None:
        return ""
    if isinstance(units, list):
        s = ", ".join(str(u) for u in units if u is not None)
    else:
        s = str(units)
    if len(s) > max_len:
        logger.warning("units_truncated htsno=%s", raw.get("htsno"))
        return s[: max_len - 3] + "..."
    return s


def _is_chapter99(htsno: str) -> bool:
    ch = _chapter_from_htsno(htsno)
    return ch == "99"


def _is_header_row(raw: dict) -> bool:
    sup = raw.get("superior")
    if sup is True:
        return True
    if isinstance(sup, str) and sup.lower() == "true":
        return True
    return False


def _hts_id(hts_code: str) -> str:
    """Stable id: strip dots; cap at 32 chars for Snowflake column."""
    compact = re.sub(r"[^\w]", "", hts_code) or hts_code.replace(".", "")[:32]
    return compact[:32]


def raw_row_to_params(raw: dict) -> Optional[dict[str, Any]]:
    htsno = raw.get("htsno")
    if htsno is None or str(htsno).strip() == "":
        return None
    hts_code = str(htsno).strip()
    description = str(raw.get("description") or "").strip() or "(no description)"

    general = str(raw.get("general") or "")
    special = str(raw.get("special") or "")
    other = str(raw.get("other") or "")
    if len(special) > 500:
        special = special[:497] + "..."

    footnotes_obj = raw.get("footnotes")
    if footnotes_obj is None:
        footnotes_obj = []

    try:
        raw_json_str = json.dumps(raw, default=str)
    except TypeError:
        raw_json_str = "{}"

    return {
        "hts_id": _hts_id(hts_code),
        "hts_code": hts_code[:32],
        "stat_suffix": None,
        "chapter": _chapter_from_htsno(hts_code),
        "level": str(raw.get("indent") or "")[:20] or None,
        "indent_level": _indent_level(raw),
        "description": description,
        "general_rate": general[:100] if general else None,
        "special_rate": special[:500] if special else None,
        "other_rate": other[:100] if other else None,
        "units": _units_string(raw, 50) or None,
        "footnotes_json": json.dumps(footnotes_obj, default=str),
        "is_chapter99": _is_chapter99(hts_code),
        "is_header_row": _is_header_row(raw),
        "raw_json": raw_json_str,
    }


def fetch_all_hts_raw() -> List[dict]:
    """Single USITC window covering the working schedule (includes chapter 99)."""
    return _get_export_list("0101", "9999")


def load_hts_codes(*, dry_run: bool = False) -> int:
    """
    Truncate HTS_CODES and reload from USITC. Returns number of rows inserted.
    """
    raw_rows = fetch_all_hts_raw()
    batches: List[dict[str, Any]] = []
    skipped = 0
    for raw in raw_rows:
        p = raw_row_to_params(raw)
        if p is None:
            skipped += 1
            continue
        batches.append(p)

    logger.info(
        "hts_fetch_complete raw=%s usable=%s skipped_empty_htsno=%s",
        len(raw_rows),
        len(batches),
        skipped,
    )

    if dry_run:
        logger.info("dry_run_no_snowflake_write")
        return len(batches)

    conn = get_connection()
    try:
        cur = conn.cursor()
        _ensure_hts_column_width(cur)
        cur.execute("TRUNCATE TABLE IF EXISTS HTS_CODES")
        cur.execute(CREATE_STAGING_SQL)
        inserted = 0
        for i in range(0, len(batches), BATCH_SIZE):
            chunk = batches[i : i + BATCH_SIZE]
            cur.execute(TRUNCATE_STAGING_SQL)
            cur.executemany(STAGING_INSERT_SQL, chunk)
            cur.execute(INSERT_FROM_STAGING_SQL)
            inserted += len(chunk)
            logger.info("hts_batch_inserted total_so_far=%s", inserted)
        conn.commit()
        logger.info("hts_load_complete inserted=%s", inserted)
        return inserted
    except Exception:
        conn.rollback()
        logger.exception("hts_load_failed")
        raise


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Load USITC HTS into Snowflake HTS_CODES")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse only; do not write to Snowflake",
    )
    args = parser.parse_args()
    n = load_hts_codes(dry_run=args.dry_run)
    logger.info("done rows=%s dry_run=%s", n, args.dry_run)


if __name__ == "__main__":
    main()
