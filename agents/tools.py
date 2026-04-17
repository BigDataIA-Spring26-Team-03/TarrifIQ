"""
TariffIQ — Agent Tool Registry  (agents/tools.py)

Single source of truth for all data access in the pipeline.
No agent imports snowflake.connector, chromadb, or census_client directly.

TOOLS
─────
  1.  hts_base_rate_lookup(hts_code)
  2.  hts_keyword_search(query, limit, chapter_filter, heading_filter)
  3.  hts_chapter_lookup(chapter, limit)
  4.  hts_verify(hts_code)
  5.  hts_description(hts_code)
  6.  alias_lookup(product)
  7.  alias_write(product, hts_code, confidence)
  8.  fetch_doc_numbers_for_hts(hts_code)
  9.  fetch_bm25_corpus(hts_code, hts_chapter)
  10. verify_fr_doc(doc_number)
  11. write_hitl_record(query, reason, hts, conf)
  12. census_trade_flow(hts_code)

Note: policy vector search and HTS vector search are handled by
services/retrieval/hybrid.py (HybridRetriever) — not in this file.
"""

import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def _sf():
    import snowflake.connector
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
    )


# ── TOOL 1 — hts_base_rate_lookup ─────────────────────────────────────────────
# Fetches general_rate, special_rate, and footnotes from HTS_CODES.
# If the country qualifies for an FTA rate, returns that instead of the MFN rate.
# This is the only place in the pipeline that handles FTA / special rates.

# FTA program codes used in HTS special_rate column
# e.g. "Free (A,AU,BH,CA,CL,CO,D,E,IL,JO,KR,MA,MX,OM,P,PA,PE,S,SG)"
COUNTRY_FTA_CODES: Dict[str, List[str]] = {
    "canada":             ["CA"],
    "mexico":             ["MX"],
    "australia":          ["AU"],
    "south korea":        ["KR"],
    "korea":              ["KR"],
    "israel":             ["IL"],
    "jordan":             ["JO"],
    "chile":              ["CL"],
    "singapore":          ["SG"],
    "morocco":            ["MA"],
    "bahrain":            ["BH"],
    "oman":               ["OM"],
    "peru":               ["PE"],
    "colombia":           ["CO"],
    "panama":             ["PA"],
    "guatemala":          ["GT"],
    "el salvador":        ["SV"],
    "honduras":           ["HN"],
    "nicaragua":          ["NI"],
    "costa rica":         ["CR"],
    "dominican republic": ["DR"],
    # GSP beneficiary countries use code "A" or "A*"
    # Major ones listed — not exhaustive
    "india":              ["A"],
    "indonesia":          ["A"],
    "thailand":           ["A"],
    "brazil":             ["A"],
    "philippines":        ["A"],
    "vietnam":            [],   # not an FTA partner — MFN applies
    "china":              [],   # Section 301 country — no FTA
    "taiwan":             [],   # no FTA
    "japan":              [],   # no FTA (as of 2025)
    "germany":            [],   # EU — no FTA
}


def _parse_rate_string(s: str) -> float:
    """Parse a rate string like '2.5%' or 'Free' to a float percentage."""
    if not s:
        return 0.0
    s = s.strip()
    if s.lower() in ("", "free"):
        return 0.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    if m:
        return float(m.group(1))
    try:
        return float(s.split()[0].replace("%", ""))
    except (ValueError, IndexError):
        return 0.0


def _parse_fta_rate(special_rate: str, country: Optional[str]) -> Optional[tuple]:
    """
    Check if country qualifies for an FTA rate in the special_rate column.
    special_rate looks like: "Free (A,AU,BH,CA,CL,CO,D,E,IL,JO,KR,MA,MX,OM,P,PA,PE,S,SG)"
    or "1.5% (CA,MX)" or "Free (AU)" etc.

    Returns (rate_float, fta_program_str) or None if no FTA applies.
    """
    if not special_rate or not country:
        return None

    country_lower = country.lower().strip()
    fta_codes = COUNTRY_FTA_CODES.get(country_lower, [])

    # Unknown country — try to find its code if it appears directly by name
    if not fta_codes:
        return None

    # Check if any of the country's FTA codes appear in special_rate
    for code in fta_codes:
        # Match code in the parenthesised list, e.g. "(A,AU,BH,CA,...)"
        if re.search(r'\b' + re.escape(code) + r'\b', special_rate):
            # Extract the rate text before the parenthesis
            # e.g. "Free (CA,MX)" → "Free"
            # e.g. "1.5% (CA,MX)" → "1.5%"
            rate_match = re.match(r'^\s*(Free|\d+\.?\d*%?)', special_rate.strip(), re.IGNORECASE)
            if rate_match:
                rate_str = rate_match.group(1)
                rate_val = 0.0 if rate_str.lower() == "free" else _parse_rate_string(rate_str)

                # Determine program name
                if code in ("CA", "MX"):
                    program = "USMCA"
                elif code == "AU":
                    program = "US-Australia FTA"
                elif code == "KR":
                    program = "KORUS FTA"
                elif code == "IL":
                    program = "US-Israel FTA"
                elif code in ("A", "A*"):
                    program = "GSP"
                else:
                    program = f"FTA ({code})"

                logger.info("fta_rate_found country=%s code=%s rate=%.2f program=%s",
                            country, code, rate_val, program)
                return (rate_val, program)

    return None


def _parse_footnotes(footnotes_json: Any) -> List[str]:
    """
    Extract readable footnote strings from the VARIANT footnotes column.
    Returns a list of strings like ["See 9903.88.01 for Section 301 duties"].
    """
    if not footnotes_json:
        return []
    try:
        import json
        if isinstance(footnotes_json, str):
            parsed = json.loads(footnotes_json)
        else:
            parsed = footnotes_json
        if isinstance(parsed, list):
            return [str(f).strip() for f in parsed if f and str(f).strip()]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
        return []
    except Exception:
        return []


def hts_base_rate_lookup(hts_code: str, country: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Returns rate info for an HTS code, country-aware.

    Result keys:
      hts_code         — resolved code (may be shorter than input)
      description      — HTS description
      base_rate        — effective rate to use (FTA if applicable, else MFN)
      general_rate_str — raw MFN rate string from DB
      mfn_rate         — parsed MFN rate (always the general rate)
      fta_rate         — FTA rate if applicable, else None
      fta_program      — e.g. "USMCA", "KORUS FTA", None
      fta_applied      — True if base_rate is the FTA rate
      footnotes        — list of footnote strings from HTS_CODES
    """
    conn = _sf()
    cur = conn.cursor()
    try:
        codes = [hts_code]
        parts = hts_code.split(".")
        while len(parts) > 2:
            parts = parts[:-1]
            codes.append(".".join(parts))

        for code in codes:
            cur.execute(
                """
                SELECT hts_code, description, general_rate, special_rate, footnotes
                FROM TARIFFIQ.RAW.HTS_CODES
                WHERE hts_code = %s
                  AND chapter NOT IN ('98','99')
                LIMIT 1
                """,
                (code,),
            )
            row = cur.fetchone()
            if row:
                if code != hts_code:
                    logger.info("hts_base_rate_fallback original=%s resolved=%s", hts_code, code)

                db_hts, description, general_rate_str, special_rate, footnotes_raw = row
                mfn_rate = _parse_rate_string(general_rate_str or "")
                footnotes = _parse_footnotes(footnotes_raw)

                # Check FTA
                fta_result = _parse_fta_rate(special_rate or "", country)
                if fta_result:
                    fta_rate, fta_program = fta_result
                    effective_rate = fta_rate
                    fta_applied = True
                    logger.info("hts_base_rate_fta hts=%s country=%s mfn=%.2f fta=%.2f program=%s",
                                code, country, mfn_rate, fta_rate, fta_program)
                else:
                    fta_rate = None
                    fta_program = None
                    fta_applied = False
                    effective_rate = mfn_rate

                return {
                    "hts_code": db_hts,
                    "description": description,
                    "base_rate": effective_rate,
                    "general_rate_str": general_rate_str or "",
                    "mfn_rate": mfn_rate,
                    "fta_rate": fta_rate,
                    "fta_program": fta_program,
                    "fta_applied": fta_applied,
                    "footnotes": footnotes,
                }

        return None
    except Exception as e:
        logger.error("hts_base_rate_lookup_error hts=%s error=%s", hts_code, e)
        return None
    finally:
        cur.close()
        conn.close()


# ── TOOL 2 — hts_keyword_search ───────────────────────────────────────────────

def hts_keyword_search(
    query: str,
    limit: int = 5,
    chapter_filter: Optional[str] = None,
    heading_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    conn = _sf()
    cur = conn.cursor()
    try:
        search_lower = query.lower().strip()
        conditions = [
            "LOWER(description) LIKE LOWER(%s)",
            "is_header_row = FALSE",
            "chapter NOT IN ('98','99')",
            "general_rate IS NOT NULL",
        ]
        params: List[Any] = [f"%{search_lower}%"]

        if heading_filter:
            digits = re.sub(r"\D", "", heading_filter)
            if len(digits) >= 6:
                conditions.append("LEFT(hts_code, 7) = %s")
                params.append(f"{digits[:4]}.{digits[4:6]}")
            else:
                conditions.append("LEFT(hts_code, 4) = %s")
                params.append(digits[:4])
        elif chapter_filter:
            conditions.append("chapter = %s")
            params.append(chapter_filter.zfill(2))

        cur.execute(
            f"""
            SELECT hts_code, description, general_rate
            FROM TARIFFIQ.RAW.HTS_CODES
            WHERE {" AND ".join(conditions)}
            ORDER BY
              CASE WHEN LOWER(description) = %s THEN 0 ELSE 1 END,
              CASE WHEN LOWER(description) LIKE %s THEN 0 ELSE 1 END,
              LENGTH(description) ASC
            LIMIT %s
            """,
            params + [search_lower, f"{search_lower}%", limit],
        )
        results = []
        for hts_code, description, general_rate in cur.fetchall():
            desc_lower = (description or "").lower()
            if desc_lower == search_lower:
                conf = 1.0
            elif desc_lower.startswith(search_lower):
                conf = 0.90
            elif f" {search_lower}" in desc_lower or f"{search_lower} " in desc_lower:
                conf = 0.85
            else:
                conf = 0.70
            results.append({
                "hts_code": hts_code,
                "description": description,
                "general_rate": general_rate or "Free",
                "chapter": hts_code[:2] if hts_code else "",
                "confidence": conf,
            })
        return results
    except Exception as e:
        logger.error("hts_keyword_search_error query=%s error=%s", query, e)
        return []
    finally:
        cur.close()
        conn.close()


# ── TOOL 3 — hts_chapter_lookup ───────────────────────────────────────────────

def hts_chapter_lookup(chapter: str, limit: int = 50) -> List[Dict[str, Any]]:
    conn = _sf()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT hts_code, description, general_rate
            FROM TARIFFIQ.RAW.HTS_CODES
            WHERE chapter = %s AND is_header_row = FALSE
              AND chapter NOT IN ('98','99')
            ORDER BY hts_code ASC LIMIT %s
            """,
            (chapter.zfill(2), limit),
        )
        return [{"hts_code": r[0], "description": r[1], "general_rate": r[2] or "Free"}
                for r in cur.fetchall()]
    except Exception as e:
        logger.error("hts_chapter_lookup_error chapter=%s error=%s", chapter, e)
        return []
    finally:
        cur.close()
        conn.close()


# ── TOOL 4 — hts_verify ───────────────────────────────────────────────────────

def hts_verify(hts_code: str) -> bool:
    conn = _sf()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT 1 FROM TARIFFIQ.RAW.HTS_CODES WHERE hts_code = %s LIMIT 1",
            (hts_code,),
        )
        return cur.fetchone() is not None
    except Exception as e:
        logger.error("hts_verify_error hts=%s error=%s", hts_code, e)
        return False
    finally:
        cur.close()
        conn.close()


# ── TOOL 5 — hts_description ──────────────────────────────────────────────────

def hts_description(hts_code: str) -> Optional[str]:
    conn = _sf()
    cur = conn.cursor()
    try:
        codes = [hts_code]
        parts = hts_code.split(".")
        while len(parts) > 2:
            parts = parts[:-1]
            codes.append(".".join(parts))
        for code in codes:
            cur.execute(
                "SELECT description FROM TARIFFIQ.RAW.HTS_CODES WHERE hts_code = %s LIMIT 1",
                (code,),
            )
            row = cur.fetchone()
            if row:
                return row[0]
        return None
    except Exception as e:
        logger.error("hts_description_error hts=%s error=%s", hts_code, e)
        return None
    finally:
        cur.close()
        conn.close()


# ── TOOL 6 — alias_lookup ─────────────────────────────────────────────────────

def alias_lookup(product: str) -> Optional[Tuple[str, float]]:
    conn = _sf()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT hts_code, confidence FROM TARIFFIQ.RAW.PRODUCT_ALIASES "
            "WHERE LOWER(alias) = LOWER(%s) LIMIT 1",
            (product.strip(),),
        )
        row = cur.fetchone()
        if row and row[0]:
            return row[0], float(row[1]) if row[1] else 0.95
        return None
    except Exception as e:
        logger.error("alias_lookup_error product=%s error=%s", product, e)
        return None
    finally:
        cur.close()
        conn.close()


# ── TOOL 7 — alias_write ──────────────────────────────────────────────────────

def alias_write(product: str, hts_code: str, confidence: float) -> None:
    stored = min(confidence, 0.95)
    conn = _sf()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT hts_code, confidence FROM TARIFFIQ.RAW.PRODUCT_ALIASES "
            "WHERE LOWER(alias) = LOWER(%s) LIMIT 1",
            (product.strip(),),
        )
        existing = cur.fetchone()
        if existing:
            if existing[0] == hts_code and stored > float(existing[1] or 0):
                cur.execute(
                    "UPDATE TARIFFIQ.RAW.PRODUCT_ALIASES SET confidence=%s, "
                    "updated_at=CURRENT_TIMESTAMP() WHERE LOWER(alias)=LOWER(%s)",
                    (stored, product.strip()),
                )
                logger.info("alias_write_updated product=%s hts=%s", product, hts_code)
            elif existing[0] != hts_code:
                logger.warning("alias_write_conflict product=%s existing=%s new=%s",
                               product, existing[0], hts_code)
        else:
            cur.execute(
                "INSERT INTO TARIFFIQ.RAW.PRODUCT_ALIASES "
                "(alias, hts_code, confidence, created_at, updated_at) "
                "VALUES (%s,%s,%s,CURRENT_TIMESTAMP(),CURRENT_TIMESTAMP())",
                (product.strip(), hts_code, stored),
            )
            logger.info("alias_write_inserted product=%s hts=%s conf=%.2f", product, hts_code, stored)
    except Exception as e:
        logger.error("alias_write_error product=%s error=%s", product, e)
    finally:
        cur.close()
        conn.close()


# ── TOOL 8 — fetch_doc_numbers_for_hts ───────────────────────────────────────

def fetch_doc_numbers_for_hts(hts_code: str) -> Set[str]:
    if not hts_code:
        return set()
    conn = _sf()
    cur = conn.cursor()
    docs: Set[str] = set()
    try:
        cur.execute(
            "SELECT DISTINCT document_number FROM TARIFFIQ.RAW.NOTICE_HTS_CODES WHERE hts_code=%s",
            (hts_code,),
        )
        docs.update(r[0] for r in cur.fetchall() if r[0])
        cur.execute(
            "SELECT DISTINCT document_number FROM TARIFFIQ.RAW.CBP_NOTICE_HTS_CODES WHERE hts_code=%s",
            (hts_code,),
        )
        docs.update(r[0] for r in cur.fetchall() if r[0])
        logger.info("fetch_doc_numbers hts=%s found=%d", hts_code, len(docs))
        return docs
    except Exception as e:
        logger.warning("fetch_doc_numbers_error hts=%s error=%s", hts_code, e)
        return docs
    finally:
        cur.close()
        conn.close()


# ── TOOL 9 — fetch_bm25_corpus ────────────────────────────────────────────────
# Only used if HybridRetriever BM25 needs to be scoped to specific HTS notices.
# In normal flow HybridRetriever builds its own corpus from all policy_notices.

def fetch_bm25_corpus(hts_code: str, hts_chapter: str) -> List[Dict[str, Any]]:
    conn = _sf()
    cur = conn.cursor()
    chunks: Dict[str, Dict] = {}
    try:
        cur.execute(
            """
            SELECT c.chunk_id, c.chunk_text, c.document_number, c.section,
                   f.title, f.publication_date::VARCHAR
            FROM TARIFFIQ.RAW.CHUNKS c
            INNER JOIN TARIFFIQ.RAW.NOTICE_HTS_CODES n ON c.document_number=n.document_number
            LEFT JOIN TARIFFIQ.RAW.FEDERAL_REGISTER_NOTICES f ON c.document_number=f.document_number
            WHERE n.hts_code=%s AND c.chunk_text IS NOT NULL LIMIT 150
            """,
            (hts_code,),
        )
        for r in cur.fetchall():
            if r[0] and r[0] not in chunks:
                chunks[r[0]] = {"chunk_id": r[0], "chunk_text": r[1], "document_number": r[2],
                                "section": r[3], "title": r[4] or "", "publication_date": r[5] or "",
                                "source": "USTR"}

        cur.execute(
            """
            SELECT cb.chunk_id, cb.chunk_text, cb.document_number, cb.section,
                   cbf.title, cbf.publication_date::VARCHAR
            FROM TARIFFIQ.RAW.CBP_CHUNKS cb
            INNER JOIN TARIFFIQ.RAW.CBP_NOTICE_HTS_CODES n ON cb.document_number=n.document_number
            LEFT JOIN TARIFFIQ.RAW.CBP_FEDERAL_REGISTER_NOTICES cbf ON cb.document_number=cbf.document_number
            WHERE n.hts_code=%s AND cb.chunk_text IS NOT NULL LIMIT 100
            """,
            (hts_code,),
        )
        for r in cur.fetchall():
            if r[0] and r[0] not in chunks:
                chunks[r[0]] = {"chunk_id": r[0], "chunk_text": r[1], "document_number": r[2],
                                "section": r[3], "title": r[4] or "", "publication_date": r[5] or "",
                                "source": "CBP"}

        if len(chunks) >= 10:
            return list(chunks.values())

        cur.execute(
            """
            SELECT c.chunk_id, c.chunk_text, c.document_number, c.section,
                   f.title, f.publication_date::VARCHAR
            FROM TARIFFIQ.RAW.CHUNKS c
            LEFT JOIN TARIFFIQ.RAW.FEDERAL_REGISTER_NOTICES f ON c.document_number=f.document_number
            LEFT JOIN TARIFFIQ.RAW.NOTICE_HTS_CODES n ON c.document_number=n.document_number
            WHERE n.hts_chapter=%s AND c.chunk_text IS NOT NULL LIMIT 150
            """,
            (hts_chapter,),
        )
        for r in cur.fetchall():
            if r[0] and r[0] not in chunks:
                chunks[r[0]] = {"chunk_id": r[0], "chunk_text": r[1], "document_number": r[2],
                                "section": r[3], "title": r[4] or "", "publication_date": r[5] or "",
                                "source": "USTR"}
        return list(chunks.values())
    except Exception as e:
        logger.error("fetch_bm25_corpus_error hts=%s error=%s", hts_code, e)
        return []
    finally:
        cur.close()
        conn.close()


# ── TOOL 10 — verify_fr_doc ───────────────────────────────────────────────────

def verify_fr_doc(doc_number: str) -> bool:
    """Check USTR + CBP Federal Register tables."""
    conn = _sf()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT 1 FROM TARIFFIQ.RAW.FEDERAL_REGISTER_NOTICES "
            "WHERE document_number=%s LIMIT 1",
            (doc_number,),
        )
        if cur.fetchone():
            return True
        cur.execute(
            "SELECT 1 FROM TARIFFIQ.RAW.CBP_FEDERAL_REGISTER_NOTICES "
            "WHERE document_number=%s LIMIT 1",
            (doc_number,),
        )
        return cur.fetchone() is not None
    except Exception as e:
        logger.error("verify_fr_doc_error doc=%s error=%s", doc_number, e)
        return True  # fail open
    finally:
        cur.close()
        conn.close()


def verify_itc_doc(doc_number: str) -> bool:
    """Check ITC_DOCUMENTS table for USITC notices."""
    conn = _sf()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT 1 FROM TARIFFIQ.RAW.ITC_DOCUMENTS "
            "WHERE document_number=%s LIMIT 1",
            (doc_number,),
        )
        return cur.fetchone() is not None
    except Exception as e:
        # Table may not exist — fail open
        logger.debug("verify_itc_doc_error doc=%s error=%s", doc_number, e)
        return False
    finally:
        cur.close()
        conn.close()


# ── TOOL 11 — write_hitl_record ───────────────────────────────────────────────

def write_hitl_record(
    query_text: str,
    trigger_reason: str,
    classifier_hts: Optional[str] = None,
    classifier_conf: Optional[float] = None,
) -> Optional[str]:
    hitl_id = str(uuid.uuid4())
    conn = _sf()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO TARIFFIQ.RAW.HITL_RECORDS
              (hitl_id, query_text, trigger_reason, classifier_hts,
               classifier_conf, human_decision, adjudicated_at, created_at)
            VALUES (%s,%s,%s,%s,%s,NULL,NULL,CURRENT_TIMESTAMP())
            """,
            (hitl_id, query_text, trigger_reason, classifier_hts, classifier_conf),
        )
        logger.info("write_hitl_record id=%s reason=%s", hitl_id, trigger_reason)
        return hitl_id
    except Exception as e:
        logger.error("write_hitl_record_error error=%s", e)
        return None
    finally:
        cur.close()
        conn.close()


# ── TOOL 12 — census_trade_flow ───────────────────────────────────────────────

def census_trade_flow(hts_code: str) -> Dict[str, Any]:
    """Live Census Bureau API — fetches most recent available month."""
    from ingestion.census_client import get_trade_flow
    try:
        return get_trade_flow(hts_code)
    except Exception as e:
        logger.error("census_trade_flow_error hts=%s error=%s", hts_code, e)
        return {}


def census_trade_flow_timed(hts_code: str, time: str) -> Dict[str, Any]:
    """
    Live Census Bureau API with explicit time (e.g. '2025-02').
    Used by trade_agent YoY comparison to fetch prior year same month.
    """
    from ingestion.census_client import get_trade_flow
    try:
        return get_trade_flow(hts_code, time=time)
    except Exception as e:
        logger.error("census_trade_flow_timed_error hts=%s time=%s error=%s", hts_code, time, e)
        return {}