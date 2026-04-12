import logging
from typing import List

logger = logging.getLogger(__name__)


def write_notice_hts_codes(
    conn,
    document_number: str,
    entities: List[dict],
    validation_results: List[dict],
    full_text: str = "",
) -> int:
    """
    MERGE-upserts (document_number, hts_code) pairs into NOTICE_HTS_CODES.
    Only writes HTS_CODE and HTS_RANGE entities (skips HTS_CHAPTER, HTS_HEADING).
    Never overwrites existing rows — WHEN NOT MATCHED only.

    Returns number of rows attempted.
    """
    if not entities:
        return 0

    status_map = {v["hts_code"]: v["match_status"] for v in validation_results}

    cur = conn.cursor()
    written = 0

    for entity in entities:
        if entity["label"] not in ("HTS_CODE", "HTS_RANGE"):
            continue

        hts_code = entity["entity_text"].strip()
        chapter = hts_code[:2] if hts_code and hts_code[0].isdigit() else ""
        match_status = status_map.get(hts_code, "UNMATCHED")

        # Pull ~150 chars of context around the entity from full text
        if full_text:
            s = max(0, entity["start_char"] - 75)
            e = min(len(full_text), entity["end_char"] + 75)
            context_snippet = full_text[s:e].replace("\n", " ").strip()
        else:
            context_snippet = ""

        cur.execute(
            """
            MERGE INTO NOTICE_HTS_CODES AS t
            USING (
                SELECT %s AS document_number, %s AS hts_code
            ) AS s
            ON t.document_number = s.document_number
               AND t.hts_code = s.hts_code
            WHEN NOT MATCHED THEN INSERT
                (document_number, hts_code, hts_chapter, context_snippet, match_status)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                document_number, hts_code,
                document_number, hts_code, chapter, context_snippet, match_status,
            ),
        )
        written += 1

    cur.close()
    logger.info(
        "write_notice_hts_codes doc=%s written=%d", document_number, written
    )
    return written
