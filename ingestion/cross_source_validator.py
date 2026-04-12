import logging
from typing import List

logger = logging.getLogger(__name__)


def validate_hts_codes(extracted_codes: List[str], conn) -> List[dict]:
    """
    Validates extracted HTS codes against the Snowflake HTS_CODES table.

    Uses a single IN clause — not N round-trips.
    If HTS_CODES is empty (e.g. friend hasn't run USITC ingest yet),
    all codes return UNMATCHED. Never drops anything — only flags.

    Returns: [{"hts_code": str, "match_status": "VERIFIED" | "UNMATCHED"}]
    """
    if not extracted_codes:
        return []

    # Deduplicate before hitting Snowflake
    unique_codes = list(dict.fromkeys(extracted_codes))

    placeholders = ", ".join(["%s"] * len(unique_codes))
    cur = conn.cursor()

    try:
        cur.execute(
            f"SELECT hts_code FROM HTS_CODES WHERE hts_code IN ({placeholders})",
            unique_codes,
        )
        verified_set = {row[0] for row in cur.fetchall()}
    except Exception as e:
        logger.warning("validate_hts_codes_query_failed: %s — marking all UNMATCHED", e)
        verified_set = set()
    finally:
        cur.close()

    results = [
        {
            "hts_code": code,
            "match_status": "VERIFIED" if code in verified_set else "UNMATCHED",
        }
        for code in unique_codes
    ]

    verified_count = sum(1 for r in results if r["match_status"] == "VERIFIED")
    logger.info(
        "validate_hts_codes_done total=%d verified=%d unmatched=%d",
        len(results),
        verified_count,
        len(results) - verified_count,
    )
    return results
