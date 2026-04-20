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
  13. fetch_top_importer_countries(hts_code, months=24, top_n=8)
  14. fetch_all_hts_linked_policy_chunks(hts_code) — all chunks for all docs referencing HTS

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

                # Skip header rows with no rate — fall through to child lookup
                if not general_rate_str or not general_rate_str.strip():
                    continue
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

        # Last resort: find first child code with a rate under this parent
        cur.execute(
            """
            SELECT hts_code, description, general_rate, special_rate, footnotes
            FROM TARIFFIQ.RAW.HTS_CODES
            WHERE hts_code LIKE %s
              AND is_header_row = FALSE
              AND general_rate IS NOT NULL
              AND chapter NOT IN ('98','99')
            ORDER BY hts_code ASC
            LIMIT 1
            """,
            (hts_code.replace(".", "") + "%",) if "." not in hts_code else (hts_code + "%",),
        )
        row = cur.fetchone()
        if row:
            logger.info("hts_base_rate_child_fallback original=%s resolved=%s", hts_code, row[0])
            db_hts, description, general_rate_str, special_rate, footnotes_raw = row
            mfn_rate = _parse_rate_string(general_rate_str or "")
            footnotes = _parse_footnotes(footnotes_raw)
            fta_result = _parse_fta_rate(special_rate or "", country)
            if fta_result:
                fta_rate, fta_program = fta_result
                effective_rate = fta_rate
                fta_applied = True
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
        # Accept if exact code has a rate, OR if it has children with rates
        cur.execute(
            "SELECT 1 FROM TARIFFIQ.RAW.HTS_CODES WHERE hts_code = %s "
            "AND is_header_row = FALSE AND general_rate IS NOT NULL LIMIT 1",
            (hts_code,),
        )
        if cur.fetchone():
            return True
        # Check if children exist with rates (handles 4-digit parent codes like 6114)
        prefix = hts_code + "%" if "." in hts_code else hts_code + "%"
        cur.execute(
            "SELECT 1 FROM TARIFFIQ.RAW.HTS_CODES WHERE hts_code LIKE %s "
            "AND is_header_row = FALSE AND general_rate IS NOT NULL LIMIT 1",
            (prefix,),
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
                    "UPDATE TARIFFIQ.RAW.PRODUCT_ALIASES SET confidence=%s "
                    "WHERE LOWER(alias)=LOWER(%s)",
                    (stored, product.strip()),
                )
                logger.info("alias_write_updated product=%s hts=%s", product, hts_code)
            elif existing[0] != hts_code:
                logger.warning("alias_write_conflict product=%s existing=%s new=%s",
                               product, existing[0], hts_code)
        else:
            cur.execute(
                "INSERT INTO TARIFFIQ.RAW.PRODUCT_ALIASES "
                "(alias, hts_code, confidence, created_at) "
                "VALUES (%s,%s,%s,CURRENT_TIMESTAMP())",
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


def verify_eop_doc(doc_number: str) -> bool:
    """Check EOP_DOCUMENTS for Executive Office notices."""
    conn = _sf()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT 1 FROM TARIFFIQ.RAW.EOP_DOCUMENTS "
            "WHERE document_number=%s LIMIT 1",
            (doc_number,),
        )
        return cur.fetchone() is not None
    except Exception as e:
        logger.debug("verify_eop_doc_error doc=%s error=%s", doc_number, e)
        return False
    finally:
        cur.close()
        conn.close()


def verify_ita_doc(doc_number: str) -> bool:
    """Check ITA_FEDERAL_REGISTER_NOTICES."""
    conn = _sf()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT 1 FROM TARIFFIQ.RAW.ITA_FEDERAL_REGISTER_NOTICES "
            "WHERE document_number=%s LIMIT 1",
            (doc_number,),
        )
        return cur.fetchone() is not None
    except Exception as e:
        logger.debug("verify_ita_doc_error doc=%s error=%s", doc_number, e)
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

# ── TOOL 13 — chapter99_lookup ───────────────────────────────────────────────

def chapter99_lookup(chapter99_codes: List[str], country: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Look up Chapter 99 surcharge codes directly in HTS_CODES.
    These codes (e.g. 9903.88.01, 9903.91.06) contain the actual adder rate
    in their general_rate field as "The duty provided in the applicable subheading + X%".

    Returns the highest applicable adder rate found, with its source code.
    Filters China-specific codes (9903.88.xx, 9903.91.xx) by country.
    """
    if not chapter99_codes:
        return None

    country_lower = (country or "").lower().strip()
    is_china = country_lower in ("china", "prc", "people's republic of china")

    conn = _sf()
    cur = conn.cursor()
    total_rate = 0.0
    applied_codes = []
    primary_code = None
    primary_desc = None

    try:
        for code in chapter99_codes:
            code = code.strip()
            if not code.startswith("99"):
                continue

            # China-specific codes — only apply if country is China
            if code.startswith("9903.88") or code.startswith("9903.91"):
                if not is_china:
                    continue

            cur.execute(
                "SELECT hts_code, general_rate, description FROM TARIFFIQ.RAW.HTS_CODES "
                "WHERE hts_code = %s LIMIT 1",
                (code,),
            )
            row = cur.fetchone()
            if not row:
                continue

            hts_code, general_rate_str, description = row
            if not general_rate_str:
                continue

            # Parse rate from strings like:
            # "The duty provided in the applicable subheading + 25%"
            # "The duty provided in the applicable subheading plus 7.5%"
            m = re.search(r"\+\s*(\d+(?:\.\d+)?)\s*%", general_rate_str)
            if not m:
                m = re.search(r"plus\s+(\d+(?:\.\d+)?)\s*%", general_rate_str, re.IGNORECASE)
            if m:
                rate = float(m.group(1))
                # IEEPA stacking: sum all applicable Chapter 99 rates
                # e.g. 9903.88.15 (7.5% Section 301) + 9903.91.06 (25% IEEPA) = 32.5%
                total_rate += rate
                applied_codes.append(hts_code)
                if primary_code is None:
                    primary_code = hts_code
                    primary_desc = description
                logger.info("chapter99_lookup found code=%s rate=%.1f total_so_far=%.1f country=%s",
                            code, rate, total_rate, country)

        if applied_codes:
            return {
                "adder_rate": total_rate,
                "chapter99_code": applied_codes[-1],  # most recently added (most recent notice)
                "chapter99_codes": applied_codes,     # all applied codes
                "description": primary_desc,
            }
        return None

    except Exception as e:
        logger.error("chapter99_lookup_error error=%s", e)
        return None
    finally:
        cur.close()
        conn.close()


# ── TOOL 14 — fetch_chapter99_from_notices ───────────────────────────────────

def fetch_chapter99_from_notices(hts_code: str) -> List[str]:
    """
    Scan NOTICE_HTS_CODES and CBP_NOTICE_HTS_CODES context_snippets
    for Chapter 99 code references (9903.xx.xx pattern).
    Returns list of unique Chapter 99 codes found.
    Used when hts_footnotes is empty but notices contain Chapter 99 references.
    """
    codes = set()
    conn = _sf()
    cur = conn.cursor()
    try:
        codes_to_try = [hts_code]
        parts = hts_code.split(".")
        while len(parts) > 2:
            parts = parts[:-1]
            codes_to_try.append(".".join(parts))

        for table in ["NOTICE_HTS_CODES", "CBP_NOTICE_HTS_CODES"]:
            for code in codes_to_try:
                try:
                    cur.execute(
                        f"SELECT context_snippet FROM TARIFFIQ.RAW.{table} "
                        f"WHERE hts_code = %s AND context_snippet IS NOT NULL LIMIT 10",
                        (code,),
                    )
                    for (snippet,) in cur.fetchall():
                        if not snippet:
                            continue
                        matches = re.findall(r"9903\.\d{2}\.\d{2}", snippet)
                        codes.update(matches)
                except Exception:
                    continue

        if codes:
            logger.info("fetch_chapter99_from_notices hts=%s found=%s", hts_code, codes)
        return list(codes)
    except Exception as e:
        logger.error("fetch_chapter99_from_notices_error hts=%s error=%s", hts_code, e)
        return []
    finally:
        cur.close()
        conn.close()


# ── TOOL 15 — fetch_rate_change_history ─────────────────────────────────────

def fetch_rate_change_history(hts_code: str, country: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch Federal Register notices that reference this HTS code,
    ordered by publication date descending.
    Used for "has the tariff changed?" queries.
    Returns list of {document_number, title, publication_date, source}
    """
    if not hts_code:
        return []

    conn = _sf()
    cur = conn.cursor()
    history = []
    seen = set()

    try:
        codes_to_try = [hts_code]
        parts = hts_code.split(".")
        while len(parts) > 2:
            parts = parts[:-1]
            codes_to_try.append(".".join(parts))

        for code in codes_to_try:
            # USTR notices
            cur.execute(
                """
                SELECT f.document_number, f.title, f.publication_date::VARCHAR, 'USTR' as source
                FROM TARIFFIQ.RAW.FEDERAL_REGISTER_NOTICES f
                INNER JOIN TARIFFIQ.RAW.NOTICE_HTS_CODES n ON f.document_number = n.document_number
                WHERE n.hts_code = %s
                ORDER BY f.publication_date DESC NULLS LAST
                LIMIT 8
                """,
                (code,),
            )
            for doc_num, title, pub_date, source in cur.fetchall():
                if doc_num and doc_num not in seen:
                    seen.add(doc_num)
                    history.append({
                        "document_number": doc_num,
                        "title": title or "",
                        "publication_date": pub_date or "",
                        "source": source,
                    })

            # CBP notices
            cur.execute(
                """
                SELECT f.document_number, f.title, f.publication_date::VARCHAR, 'CBP' as source
                FROM TARIFFIQ.RAW.CBP_FEDERAL_REGISTER_NOTICES f
                INNER JOIN TARIFFIQ.RAW.CBP_NOTICE_HTS_CODES n ON f.document_number = n.document_number
                WHERE n.hts_code = %s
                ORDER BY f.publication_date DESC NULLS LAST
                LIMIT 8
                """,
                (code,),
            )
            for doc_num, title, pub_date, source in cur.fetchall():
                if doc_num and doc_num not in seen:
                    seen.add(doc_num)
                    history.append({
                        "document_number": doc_num,
                        "title": title or "",
                        "publication_date": pub_date or "",
                        "source": source,
                    })

            if history:
                break

        # Sort by date descending
        history.sort(key=lambda x: x.get("publication_date", ""), reverse=True)
        logger.info("fetch_rate_change_history hts=%s found=%d", hts_code, len(history))
        return history[:14]

    except Exception as e:
        logger.error("fetch_rate_change_history_error hts=%s error=%s", hts_code, e)
        return []
    finally:
        cur.close()
        conn.close()


# ── TOOL 16 — hitl_feedback_write ────────────────────────────────────────────

def hitl_feedback_write(hitl_id: str, correct_hts: str, human_notes: str = "") -> bool:
    """
    Write human decision back to HITL_RECORDS and update PRODUCT_ALIASES.
    Called when a human reviewer adjudicates a HITL escalation.
    This closes the feedback loop — correct HTS gets learned for future queries.
    """
    conn = _sf()
    cur = conn.cursor()
    try:
        # Update HITL record with human decision
        cur.execute(
            """
            UPDATE TARIFFIQ.RAW.HITL_RECORDS
            SET human_decision = %s,
                adjudicated_at = CURRENT_TIMESTAMP()
            WHERE hitl_id = %s
            """,
            (correct_hts, hitl_id),
        )

        # Fetch the original query to write alias
        cur.execute(
            "SELECT query_text FROM TARIFFIQ.RAW.HITL_RECORDS WHERE hitl_id = %s",
            (hitl_id,),
        )
        row = cur.fetchone()
        if row:
            query_text = row[0]
            # Extract product (first few words before "from")
            product = re.split(r"\s+from\s+", query_text.lower())[0]
            product = re.sub(r"^(what is (the )?tariff on |tariff on |import duty on )", "", product).strip()
            if product and correct_hts:
                alias_write(product, correct_hts, 0.90)
                logger.info("hitl_feedback_alias_written product=%s hts=%s", product, correct_hts)

        logger.info("hitl_feedback_write hitl_id=%s hts=%s", hitl_id, correct_hts)
        return True

    except Exception as e:
        logger.error("hitl_feedback_write_error hitl_id=%s error=%s", hitl_id, e)
        return False
    finally:
        cur.close()
        conn.close()


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


def _census_cty_code_to_lookup_country(cty_code: str, cty_name: str) -> str:
    """Map Census Schedule C country code / name to a country string for hts_base_rate_lookup."""
    from agents.trade_agent import COUNTRY_CODE_MAP

    inv: Dict[str, str] = getattr(_census_cty_code_to_lookup_country, "_inv", None)
    if inv is None:
        inv = {}
        for name, code in COUNTRY_CODE_MAP.items():
            if code not in inv:
                inv[code] = name
        _census_cty_code_to_lookup_country._inv = inv  # type: ignore[attr-defined]

    cc = str(cty_code or "").strip().zfill(4) if str(cty_code or "").strip().isdigit() else str(cty_code or "").strip()
    if cc in inv:
        return inv[cc]

    base = str(cty_name or "").split(",")[0].strip().lower()
    aliases = {
        "viet nam": "vietnam",
        "russian federation": "russia",
        "korea": "south korea",
        "north korea": "north korea",
        "hong kong": "hong kong",
        "macao": "macao",
        "türkiye": "turkey",
        "turkiye": "turkey",
        "people's republic of china": "china",
        "peoples republic of china": "china",
    }
    if base in aliases:
        return aliases[base]
    return base


def fetch_top_importer_countries(
    hts_code: str,
    *,
    months: int = 24,
    top_n: int = 8,
) -> List[Dict[str, Any]]:
    """
    Rank partner countries by summed US import value (GEN_VAL_MO) over trailing ``months``
    via Census HS import API. Attach indicative MFN/FTA baseline rate per country from HTS_CODES.

    Section 301 / 232 adders are not recomputed here — same caveats as hts_base_rate_lookup.
    """
    from ingestion.census_client import get_trade_trend

    if not hts_code:
        return []

    months = max(1, min(int(months), 36))
    trend = get_trade_trend(hts_code, months=months)

    def _monthly_val(row: Dict[str, Any]) -> float:
        raw = row.get("GEN_VAL_MO") if "GEN_VAL_MO" in row else row.get("gen_val_mo")
        try:
            if raw in (None, "", "(D)", "-"):
                return 0.0
            return float(str(raw).replace(",", ""))
        except (ValueError, TypeError):
            return 0.0

    totals: Dict[str, Dict[str, Any]] = {}
    for snap in trend:
        for row in snap.get("rows") or []:
            code = str(row.get("CTY_CODE", "") or "").strip()
            name = str(row.get("CTY_NAME", "") or "").strip()
            nu = name.upper()
            if not code or code == "-" or "TOTAL FOR ALL COUNTRIES" in nu:
                continue
            val = _monthly_val(row)
            if code not in totals:
                totals[code] = {"cty_code": code, "cty_name": name, "imports_usd": 0.0}
            totals[code]["imports_usd"] += val

    ranked = sorted(totals.values(), key=lambda x: x["imports_usd"], reverse=True)[:top_n]

    out: List[Dict[str, Any]] = []
    for item in ranked:
        lookup_name = _census_cty_code_to_lookup_country(item["cty_code"], item["cty_name"])
        rr = hts_base_rate_lookup(hts_code, country=lookup_name)
        row_out: Dict[str, Any] = {
            "census_country_name": item["cty_name"],
            "cty_code": item["cty_code"],
            "imports_usd_trailing": round(item["imports_usd"], 2),
            "months_in_sample": months,
            "lookup_country": lookup_name,
            "base_rate": None,
            "mfn_rate": None,
            "fta_program": None,
            "fta_applied": None,
        }
        if rr:
            row_out["base_rate"] = rr.get("base_rate")
            row_out["mfn_rate"] = rr.get("mfn_rate")
            row_out["fta_program"] = rr.get("fta_program")
            row_out["fta_applied"] = rr.get("fta_applied")
        out.append(row_out)

    logger.info(
        "fetch_top_importer_countries hts=%s months=%d rows=%d",
        hts_code,
        months,
        len(out),
    )
    return out


def _hts_parent_codes(hts_code: str) -> List[str]:
    """8421.30.01.00 → [8421.30.01.00, 8421.30.01, 8421.30, 8421]."""
    if not (hts_code or "").strip():
        return []
    codes = [hts_code.strip()]
    parts = hts_code.split(".")
    while len(parts) > 2:
        parts = parts[:-1]
        codes.append(".".join(parts))
    return codes


def fetch_all_hts_linked_policy_chunks(hts_code: str) -> List[Dict[str, Any]]:
    """
    Load chunks for **every** document that references this HTS (or a parent code), but at most
    ``HTS_LINKED_CHUNKS_PER_DOCUMENT`` chunks per document (ordered by chunk_index).

    Sources: USTR, CBP, USITC, EOP (notice joins); ITA_CHUNKS by hts_code column.
    """
    if not (hts_code or "").strip():
        return []

    per_doc = max(1, int(os.environ.get("HTS_LINKED_CHUNKS_PER_DOCUMENT", "5")))

    codes = _hts_parent_codes(hts_code)
    seen: Set[str] = set()
    rows_out: List[Dict[str, Any]] = []

    conn = _sf()
    cur = conn.cursor()
    try:
        for code in codes:

            def _add_row(
                chunk_id: Any,
                chunk_text: Any,
                doc_num: Any,
                chunk_index: Any,
                section: Any,
                title: Any,
                pub_date: Any,
                src: str,
                method: str,
            ) -> None:
                if not chunk_id or not chunk_text:
                    return
                cid = str(chunk_id)
                if cid in seen:
                    return
                seen.add(cid)
                rows_out.append({
                    "chunk_id": cid,
                    "chunk_text": str(chunk_text),
                    "document_number": str(doc_num or ""),
                    "chunk_index": int(chunk_index) if chunk_index is not None else 0,
                    "section": str(section or ""),
                    "title": str(title or ""),
                    "publication_date": str(pub_date or ""),
                    "source": src,
                    "hts_code": code,
                    "retrieval_method": method,
                })

            cur.execute(
                """
                WITH ranked AS (
                    SELECT c.chunk_id, c.chunk_text, c.document_number, c.chunk_index, c.section,
                           f.title, f.publication_date::VARCHAR AS publication_date,
                           ROW_NUMBER() OVER (
                               PARTITION BY c.document_number
                               ORDER BY c.chunk_index ASC NULLS LAST
                           ) AS rn
                    FROM TARIFFIQ.RAW.CHUNKS c
                    INNER JOIN TARIFFIQ.RAW.NOTICE_HTS_CODES n
                        ON c.document_number = n.document_number
                    LEFT JOIN TARIFFIQ.RAW.FEDERAL_REGISTER_NOTICES f
                        ON c.document_number = f.document_number
                    WHERE n.hts_code = %s AND c.chunk_text IS NOT NULL
                )
                SELECT chunk_id, chunk_text, document_number, chunk_index, section, title, publication_date
                FROM ranked
                WHERE rn <= %s
                ORDER BY publication_date ASC NULLS LAST, document_number, chunk_index
                """,
                (code, per_doc),
            )
            for r in cur.fetchall():
                _add_row(
                    r[0], r[1], r[2], r[3], r[4], r[5], r[6], "USTR", "hts_notice_full",
                )

            cur.execute(
                """
                WITH ranked AS (
                    SELECT c.chunk_id, c.chunk_text, c.document_number, c.chunk_index, c.section,
                           f.title, f.publication_date::VARCHAR AS publication_date,
                           ROW_NUMBER() OVER (
                               PARTITION BY c.document_number
                               ORDER BY c.chunk_index ASC NULLS LAST
                           ) AS rn
                    FROM TARIFFIQ.RAW.CBP_CHUNKS c
                    INNER JOIN TARIFFIQ.RAW.CBP_NOTICE_HTS_CODES n
                        ON c.document_number = n.document_number
                    LEFT JOIN TARIFFIQ.RAW.CBP_FEDERAL_REGISTER_NOTICES f
                        ON c.document_number = f.document_number
                    WHERE n.hts_code = %s AND c.chunk_text IS NOT NULL
                )
                SELECT chunk_id, chunk_text, document_number, chunk_index, section, title, publication_date
                FROM ranked
                WHERE rn <= %s
                ORDER BY publication_date ASC NULLS LAST, document_number, chunk_index
                """,
                (code, per_doc),
            )
            for r in cur.fetchall():
                _add_row(
                    r[0], r[1], r[2], r[3], r[4], r[5], r[6], "CBP", "hts_notice_full",
                )

            try:
                cur.execute(
                    """
                    WITH ranked AS (
                        SELECT c.chunk_id, c.chunk_text, c.document_number, c.chunk_index, c.section,
                               f.title, f.publication_date::VARCHAR AS publication_date,
                               ROW_NUMBER() OVER (
                                   PARTITION BY c.document_number
                                   ORDER BY c.chunk_index ASC NULLS LAST
                               ) AS rn
                        FROM TARIFFIQ.RAW.ITC_CHUNKS c
                        INNER JOIN TARIFFIQ.RAW.NOTICE_HTS_CODES_ITC n
                            ON c.document_number = n.document_number
                        LEFT JOIN TARIFFIQ.RAW.ITC_DOCUMENTS f
                            ON c.document_number = f.document_number
                        WHERE n.hts_code = %s AND c.chunk_text IS NOT NULL
                    )
                    SELECT chunk_id, chunk_text, document_number, chunk_index, section, title, publication_date
                    FROM ranked
                    WHERE rn <= %s
                    ORDER BY publication_date ASC NULLS LAST, document_number, chunk_index
                    """,
                    (code, per_doc),
                )
                for r in cur.fetchall():
                    _add_row(
                        r[0], r[1], r[2], r[3], r[4], r[5], r[6], "USITC", "hts_notice_full",
                    )
            except Exception as e:
                logger.debug("fetch_all_hts_itc code=%s err=%s", code, e)

            try:
                cur.execute(
                    """
                    WITH ranked AS (
                        SELECT c.chunk_id, c.chunk_text, c.document_number, c.chunk_index, c.section,
                               f.title, f.publication_date::VARCHAR AS publication_date,
                               ROW_NUMBER() OVER (
                                   PARTITION BY c.document_number
                                   ORDER BY c.chunk_index ASC NULLS LAST
                               ) AS rn
                        FROM TARIFFIQ.RAW.EOP_CHUNKS c
                        INNER JOIN TARIFFIQ.RAW.NOTICE_HTS_CODES_EOP n
                            ON c.document_number = n.document_number
                        LEFT JOIN TARIFFIQ.RAW.EOP_DOCUMENTS f
                            ON c.document_number = f.document_number
                        WHERE n.hts_code = %s AND c.chunk_text IS NOT NULL
                    )
                    SELECT chunk_id, chunk_text, document_number, chunk_index, section, title, publication_date
                    FROM ranked
                    WHERE rn <= %s
                    ORDER BY publication_date ASC NULLS LAST, document_number, chunk_index
                    """,
                    (code, per_doc),
                )
                for r in cur.fetchall():
                    _add_row(
                        r[0], r[1], r[2], r[3], r[4], r[5], r[6], "EOP", "hts_notice_full",
                    )
            except Exception as e:
                logger.debug("fetch_all_hts_eop code=%s err=%s", code, e)

            try:
                cur.execute(
                    """
                    WITH ranked AS (
                        SELECT c.chunk_id, c.chunk_text, c.document_number, c.chunk_index, c.section,
                               CAST(NULL AS VARCHAR) AS title,
                               CAST(NULL AS VARCHAR) AS publication_date,
                               ROW_NUMBER() OVER (
                                   PARTITION BY c.document_number
                                   ORDER BY c.chunk_index ASC NULLS LAST
                               ) AS rn
                        FROM TARIFFIQ.RAW.ITA_CHUNKS c
                        WHERE c.hts_code = %s AND c.chunk_text IS NOT NULL
                    )
                    SELECT chunk_id, chunk_text, document_number, chunk_index, section, title, publication_date
                    FROM ranked
                    WHERE rn <= %s
                    ORDER BY document_number, chunk_index
                    """,
                    (code, per_doc),
                )
                for r in cur.fetchall():
                    _add_row(
                        r[0], r[1], r[2], r[3], r[4], r[5], r[6], "ITA", "hts_chunk_hts_code",
                    )
            except Exception as e:
                logger.debug("fetch_all_hts_ita code=%s err=%s", code, e)

        rows_out.sort(
            key=lambda x: (
                x.get("publication_date") or "",
                x.get("document_number") or "",
                x.get("chunk_index") or 0,
            )
        )
        logger.info(
            "fetch_all_hts_linked_policy_chunks hts=%s codes_tried=%d chunks=%d per_doc_cap=%d",
            hts_code,
            len(codes),
            len(rows_out),
            per_doc,
        )
        return rows_out
    except Exception as e:
        logger.error("fetch_all_hts_linked_policy_chunks_error hts=%s error=%s", hts_code, e)
        return []
    finally:
        cur.close()
        conn.close()