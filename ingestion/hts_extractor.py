import json
import logging
import re
from typing import Any, List

from ingestion.snowflake_writer import write_notice_hts_codes

logger = logging.getLogger(__name__)

# Patterns in priority order — longest/most specific first.
_PATTERNS = [
    (re.compile(r"\b(\d{4}\.\d{2}\.\d{4})\b"), "HTS_CODE"),
    (re.compile(r"\b(\d{4}\.\d{2}\.\d{2})\b"), "HTS_CODE"),
    (re.compile(r"\b(\d{4}\.\d{2}\s+through\s+\d{4}\.\d{2})\b", re.IGNORECASE), "HTS_RANGE"),
    (re.compile(r"\b(\d{4}\.\d{2})\b"), "HTS_CODE"),
    (re.compile(r"\b(chapter\s+\d{1,2})\b", re.IGNORECASE), "HTS_CHAPTER"),
    (re.compile(r"\b(heading\s+\d{4})\b", re.IGNORECASE), "HTS_HEADING"),
]

_EXTERNAL_REF_PATTERNS: list[tuple[str, str, str]] = [
    ("see the Preliminary Decision Memorandum", "DECISION_MEMO", "DOCKET_LOOKUP"),
    ("see the Decision Memorandum", "DECISION_MEMO", "DOCKET_LOOKUP"),
    ("see the Final Decision Memorandum", "DECISION_MEMO", "DOCKET_LOOKUP"),
    ("see the Issues and Decision Memorandum", "DECISION_MEMO", "DOCKET_LOOKUP"),
    ("for a complete description of the scope", "DECISION_MEMO", "DOCKET_LOOKUP"),
    ("for a full description of the scope", "DECISION_MEMO", "DOCKET_LOOKUP"),
    ("see Annex A", "ANNEX", "SAME_DOC_ANNEX"),
    ("see Annex B", "ANNEX", "SAME_DOC_ANNEX"),
    ("see the Annex to this", "ANNEX", "SAME_DOC_ANNEX"),
    ("see the annex attached", "ANNEX", "SAME_DOC_ANNEX"),
    ("as set forth in the annex", "ANNEX", "SAME_DOC_ANNEX"),
    ("the annex to this order", "ANNEX", "SAME_DOC_ANNEX"),
    ("see HQ ruling", "CBP_RULING", "EXTERNAL_DB"),
    ("see NY ruling", "CBP_RULING", "EXTERNAL_DB"),
    ("pursuant to ruling", "CBP_RULING", "EXTERNAL_DB"),
    ("CBP ruling", "CBP_RULING", "EXTERNAL_DB"),
    ("see the Commission's report", "COMMISSION_REPORT", "DOCKET_LOOKUP"),
    ("see the Commission's determination", "COMMISSION_REPORT", "DOCKET_LOOKUP"),
    ("in accordance with the Commission's", "COMMISSION_REPORT", "DOCKET_LOOKUP"),
    ("see the schedule attached hereto", "SCHEDULE", "SAME_DOC_ANNEX"),
    ("see the annexed schedule", "SCHEDULE", "SAME_DOC_ANNEX"),
    ("as set forth in the schedule", "SCHEDULE", "SAME_DOC_ANNEX"),
]

_AD_CVD_DOCKET_RE = re.compile(r"\b[AC]-\d{3}-\d{3}\b")
_HTS_DOTTED_PATTERN = re.compile(r"\b\d{4}\.\d{2}(?:\.\d{2,4})?\b")


def _digits_only(s: str) -> str:
    return "".join(c for c in (s or "") if c.isdigit())


def _hts_chapter_from_code(hts_code: str) -> str | None:
    d = _digits_only(hts_code)
    return d[:2] if len(d) >= 2 else None


def _sentence_for_match(text: str, start: int, end: int) -> str:
    """Return fixed-width context: 500 chars before and after match."""
    left = max(0, start - 500)
    right = min(len(text), end + 500)
    return text[left:right].strip()[:2000]


def extract_hts_codes_precise(document_number: str, text: str) -> list[dict[str, Any]]:
    """PoC-style direct extraction with first-hit sentence snippets."""
    if not text:
        return []
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for m in _HTS_DOTTED_PATTERN.finditer(text):
        code = m.group(0)
        if code in seen:
            continue
        seen.add(code)
        records.append(
            {
                "document_number": document_number,
                "hts_code": code,
                "hts_chapter": _hts_chapter_from_code(code),
                "context_snippet": _sentence_for_match(text, m.start(), m.end()),
                "match_status": "UNVERIFIED",
                "hs_level": None,
                "raw_match": code,
            }
        )
    return records


def extract_hts_entities(text: str) -> List[dict]:
    seen_spans: set[tuple[int, int]] = set()
    entities: List[dict] = []

    for pattern, label in _PATTERNS:
        for match in pattern.finditer(text or ""):
            start, end = match.start(), match.end()
            if any(s <= start < e or s < end <= e for (s, e) in seen_spans):
                continue
            seen_spans.add((start, end))
            entities.append(
                {
                    "entity_text": match.group(1) if match.lastindex else match.group(),
                    "label": label,
                    "start_char": start,
                    "end_char": end,
                }
            )

    entities.sort(key=lambda x: x["start_char"])
    return entities


def detect_external_reference(text: str, title: str, agency: str) -> dict[str, Any]:
    del title, agency
    hay = (text or "").lower()
    found: list[str] = []
    ref_type = None
    strategy = "NONE"
    for phrase, rtype, rstrategy in _EXTERNAL_REF_PATTERNS:
        if phrase.lower() in hay:
            found.append(phrase)
            if ref_type is None:
                ref_type = rtype
                strategy = rstrategy
    return {
        "has_external_ref": bool(found),
        "ref_type": ref_type,
        "ref_patterns_found": found,
        "resolution_strategy": strategy,
    }


def _normalize_country(raw: str) -> str:
    s = (raw or "").strip(" .;:")
    s = re.sub(r"^the\s+", "", s, flags=re.IGNORECASE)
    if "people's republic of china" in s.lower():
        return "China"
    return s


def extract_product_and_country(title: str) -> dict[str, Any]:
    t = (title or "").strip()
    patterns = [
        re.compile(r"^Certain\s+(.+?)\s+from\s+(.+?)(?::|$)", re.IGNORECASE),
        re.compile(r"^(.+?)\s+from\s+(.+?)(?::|$)", re.IGNORECASE),
        re.compile(r"^(.+?)\s+from\s+the\s+(.+?)(?::|$)", re.IGNORECASE),
    ]
    product = None
    countries: list[str] = []
    for pat in patterns:
        m = pat.search(t)
        if not m:
            continue
        product = m.group(1).strip()
        raw_countries = m.group(2).strip()
        parts = re.split(r"\s+and\s+|,", raw_countries)
        countries = [_normalize_country(p) for p in parts if p.strip()]
        break
    return {
        "product_name": product,
        "countries": countries,
        "raw_title": t,
    }


def _extract_annex_section(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"\bANNEX\b[\s\S]*$", text, flags=re.IGNORECASE)
    return m.group(0).strip() if m else ""


def _extract_docket_number(text: str, docket_number: str | None) -> str | None:
    if docket_number and _AD_CVD_DOCKET_RE.search(docket_number):
        return _AD_CVD_DOCKET_RE.search(docket_number).group(0)
    m = _AD_CVD_DOCKET_RE.search(text or "")
    return m.group(0) if m else None


def resolve_via_docket(document_number: str, docket_number: str, conn) -> list[dict]:
    if not docket_number:
        logger.info("DOCKET_NOT_RESOLVED doc=%s reason=no_docket", document_number)
        return []
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT document_number
            FROM FEDERAL_REGISTER_NOTICES
            WHERE raw_json::text ILIKE %s
              AND processing_status IN ('hts_extracted', 'chunked')
            ORDER BY publication_date ASC
            LIMIT 1
            """,
            (f"%{docket_number}%",),
        )
        row = cur.fetchone()
        if not row:
            logger.info("DOCKET_NOT_RESOLVED doc=%s docket=%s", document_number, docket_number)
            return []
        source_doc = row[0]
        cur.execute(
            """
            SELECT hts_code, hts_chapter, context_snippet
            FROM NOTICE_HTS_CODES
            WHERE document_number = %s
            """,
            (source_doc,),
        )
        out = []
        for hts_code, hts_chapter, context_snippet in cur.fetchall() or []:
            if not hts_code:
                continue
            out.append(
                {
                    "document_number": document_number,
                    "hts_code": hts_code,
                    "hts_chapter": hts_chapter,
                    "context_snippet": context_snippet or f"Resolved via docket {docket_number}",
                    "match_status": "DOCKET_RESOLVED",
                    "hs_level": None,
                    "raw_match": docket_number,
                }
            )
        return out
    except Exception as e:
        logger.error("resolve_via_docket_failed doc=%s docket=%s err=%s", document_number, docket_number, e)
        return []
    finally:
        cur.close()


def _to_code_records(document_number: str, text: str, entities: list[dict]) -> list[dict]:
    records: list[dict] = []
    for e in entities:
        if e.get("label") not in ("HTS_CODE", "HTS_RANGE"):
            continue
        code = (e.get("entity_text") or "").strip()
        if not code:
            continue
        level = None
        if re.search(r"\bthrough\b", code, re.IGNORECASE):
            level = "RANGE"
        else:
            n = len(_digits_only(code))
            if n in (2, 4, 6, 8, 10):
                level = f"HS{n}"
        # Capture 500 chars before and 500 chars after the match
        s = max(0, int(e.get("start_char", 0)) - 500)
        ee = min(len(text), int(e.get("end_char", 0)) + 500)
        ctx = (text[s:ee] if text else "").replace("\n", " ").strip()
        records.append(
            {
                "document_number": document_number,
                "hts_code": code,
                "hts_chapter": _hts_chapter_from_code(code),
                "context_snippet": ctx,
                "match_status": "UNVERIFIED",
                "hs_level": level,
                "raw_match": code,
            }
        )
    return records


def validate_hts_codes(codes_or_conn, maybe_conn=None):
    """
    Backward compatible:
      - validate_hts_codes(conn, ["8471.30", ...]) -> [{hts_code, match_status}, ...]
      - validate_hts_codes([record_dict, ...], conn) -> validated record dicts
    """
    if maybe_conn is None:
        conn = codes_or_conn
        codes = []
    elif isinstance(codes_or_conn, list):
        conn = maybe_conn
        records: list[dict] = [dict(r) for r in codes_or_conn]
        if not records:
            return []
        cur = conn.cursor()
        try:
            for rec in records:
                status = rec.get("match_status")
                code = (rec.get("hts_code") or "").strip() if rec.get("hts_code") else ""
                if not code:
                    continue
                if status in (
                    "DOCKET_RESOLVED",
                    "NEEDS_DECISION_MEMO",
                    "NEEDS_ANNEX",
                    "NEEDS_CBP_RULING",
                    "NEEDS_COMMISSION_REPORT",
                    "NEEDS_EXTERNAL_REF",
                    "NO_HTS_FOUND",
                    "PRODUCT_NAME_ONLY",
                ):
                    continue
                if code.startswith("99"):
                    rec["match_status"] = "CHAPTER_99"
                    continue
                norm = _digits_only(code)
                if not norm:
                    rec["match_status"] = "UNVERIFIED"
                    continue
                try:
                    if len(norm) >= 6:
                        cur.execute(
                            """
                            SELECT 1
                            FROM HTS_CODES
                            WHERE hts_code = %s
                               OR REPLACE(hts_code, '.', '') = %s
                               OR REPLACE(hts_code, '.', '') LIKE CONCAT(%s, '%%')
                            LIMIT 1
                            """,
                            (code, norm, norm),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT 1
                            FROM HTS_CODES
                            WHERE hts_code = %s
                               OR REPLACE(hts_code, '.', '') = %s
                            LIMIT 1
                            """,
                            (code, norm),
                        )
                    rec["match_status"] = "VERIFIED" if cur.fetchone() else "UNVERIFIED"
                except Exception as e:
                    logger.warning("validate_hts_codes failed code=%s err=%s", code, e)
                    rec["match_status"] = "UNVERIFIED"
        finally:
            cur.close()
        return records
    else:
        conn = codes_or_conn
        codes = maybe_conn or []

    # Legacy mode with list[str]
    if not codes:
        return []
    unique: list[str] = []
    seen: set[str] = set()
    for c in codes:
        key = (c or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(key)
    results: list[dict] = []
    cur = conn.cursor()
    try:
        for code in unique:
            if re.search(r"\bthrough\b", code, re.IGNORECASE):
                results.append({"hts_code": code, "match_status": "UNVERIFIED"})
                continue
            norm = _digits_only(code)
            if not norm:
                results.append({"hts_code": code, "match_status": "UNVERIFIED"})
                continue
            verified = False
            try:
                if len(norm) >= 6:
                    cur.execute(
                        """
                        SELECT 1
                        FROM HTS_CODES
                        WHERE hts_code = %s
                           OR REPLACE(hts_code, '.', '') = %s
                           OR REPLACE(hts_code, '.', '') LIKE CONCAT(%s, '%%')
                        LIMIT 1
                        """,
                        (code, norm, norm),
                    )
                else:
                    cur.execute(
                        """
                        SELECT 1
                        FROM HTS_CODES
                        WHERE hts_code = %s
                           OR REPLACE(hts_code, '.', '') = %s
                        LIMIT 1
                        """,
                        (code, norm),
                    )
                verified = cur.fetchone() is not None
            except Exception as e:
                logger.warning("validate_hts_codes failed code=%s err=%s", code, e)
            results.append({"hts_code": code, "match_status": "VERIFIED" if verified else "UNVERIFIED"})
    finally:
        cur.close()
    return results


def _write_notice_records(conn, records: list[dict]) -> int:
    if not records:
        return 0
    cur = conn.cursor()
    written = 0
    try:
        for rec in records:
            status = rec.get("match_status") or "UNVERIFIED"
            code = (rec.get("hts_code") or "").strip()
            if not code:
                # NOTICE_HTS_CODES requires non-null hts_code; store a status marker row.
                code = f"__{status}__"
            chapter = rec.get("hts_chapter") or _hts_chapter_from_code(code) or ""
            context = (rec.get("context_snippet") or "")[:2000]
            cur.execute(
                """
                MERGE INTO NOTICE_HTS_CODES AS t
                USING (SELECT %s AS document_number, %s AS hts_code) AS s
                ON t.document_number = s.document_number AND t.hts_code = s.hts_code
                WHEN MATCHED THEN UPDATE SET
                    t.hts_chapter = %s,
                    t.context_snippet = %s,
                    t.match_status = %s
                WHEN NOT MATCHED THEN INSERT
                    (document_number, hts_code, hts_chapter, context_snippet, match_status)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    rec.get("document_number"),
                    code,
                    chapter,
                    context,
                    status,
                    rec.get("document_number"),
                    code,
                    chapter,
                    context,
                    status,
                ),
            )
            written += 1
    finally:
        cur.close()
    return written


def update_chunks_with_hts(chunks: List[dict[str, Any]], full_text: str) -> List[dict[str, Any]]:
    _ = full_text
    out: List[dict[str, Any]] = []
    for ch in chunks:
        row = dict(ch)
        text = str(row.get("chunk_text") or "")
        entities = extract_hts_entities(text)
        code_ents = [e for e in entities if e["label"] == "HTS_CODE"]
        if code_ents:
            best = max(code_ents, key=lambda e: len(e["entity_text"]))
            raw = best["entity_text"].strip()
            row["hts_code"] = raw
            row["hts_chapter"] = _hts_chapter_from_code(raw)
        out.append(row)
    return out


def _lookup_title_and_agency(document_number: str, conn) -> tuple[str, str, str | None]:
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT title, agency_names, raw_json
            FROM FEDERAL_REGISTER_NOTICES
            WHERE document_number = %s
            LIMIT 1
            """,
            (document_number,),
        )
        row = cur.fetchone()
        if not row:
            return "", "", None
        title, agency_names, raw_json = row
        agency = ""
        if isinstance(agency_names, str):
            try:
                parsed = json.loads(agency_names)
                if isinstance(parsed, list) and parsed:
                    agency = str(parsed[0])
            except Exception:
                agency = agency_names
        elif isinstance(agency_names, list) and agency_names:
            agency = str(agency_names[0])
        raw_text = json.dumps(raw_json) if isinstance(raw_json, (dict, list)) else str(raw_json or "")
        docket = _extract_docket_number(raw_text, None)
        return str(title or ""), agency, docket
    except Exception as e:
        logger.warning("lookup_title_and_agency_failed doc=%s err=%s", document_number, e)
        return "", "", None
    finally:
        cur.close()


def run_extraction_pipeline(
    document_number: str,
    text: str,
    conn,
    title: str = "",
    agency: str = "",
    docket_number: str | None = None,
) -> dict[str, Any]:
    """Direct HTS extraction with cross-reference fallback and docket resolution."""
    try:
        if not title or not agency or not docket_number:
            t, a, d = _lookup_title_and_agency(document_number, conn)
            title = title or t
            agency = agency or a
            docket_number = docket_number or d

        records = extract_hts_codes_precise(document_number, text or "")
        product_info = extract_product_and_country(title)

        if not records:
            ref_info = detect_external_reference(text or "", title, agency)
            if ref_info["has_external_ref"]:
                if ref_info["resolution_strategy"] == "DOCKET_LOOKUP" and docket_number:
                    records = resolve_via_docket(document_number, docket_number, conn)
                elif ref_info["resolution_strategy"] == "SAME_DOC_ANNEX":
                    annex_text = _extract_annex_section(text or "")
                    if annex_text:
                        annex_entities = extract_hts_entities(annex_text)
                        records = _to_code_records(document_number, annex_text, annex_entities)

                if not records:
                    needs_status = {
                        "DECISION_MEMO": "NEEDS_DECISION_MEMO",
                        "ANNEX": "NEEDS_ANNEX",
                        "CBP_RULING": "NEEDS_CBP_RULING",
                        "COMMISSION_REPORT": "NEEDS_COMMISSION_REPORT",
                    }.get(ref_info.get("ref_type"), "NEEDS_EXTERNAL_REF")
                    records = [
                        {
                            "document_number": document_number,
                            "hts_code": None,
                            "hts_chapter": None,
                            "context_snippet": (
                                f"Product: {product_info.get('product_name')} | "
                                f"Countries: {product_info.get('countries')} | "
                                f"Ref type: {ref_info.get('ref_type')}"
                            ),
                            "match_status": needs_status,
                            "hs_level": None,
                            "raw_match": None,
                        }
                    ]
            else:
                status = "PRODUCT_NAME_ONLY" if product_info.get("product_name") else "NO_HTS_FOUND"
                records = [
                    {
                        "document_number": document_number,
                        "hts_code": None,
                        "hts_chapter": None,
                        "context_snippet": (
                            f"Product: {product_info.get('product_name')} | "
                            f"Countries: {product_info.get('countries')}"
                        ),
                        "match_status": status,
                        "hs_level": None,
                        "raw_match": None,
                    }
                ]

        validated = validate_hts_codes(records, conn)
        written = _write_notice_records(conn, validated)

        return {
            "document_number": document_number,
            "total_extracted": len([c for c in validated if c.get("hts_code") and not str(c.get("hts_code")).startswith("__")]),
            "verified": len([c for c in validated if c.get("match_status") == "VERIFIED"]),
            "unverified": len([c for c in validated if c.get("match_status") == "UNVERIFIED"]),
            "chapter99": len([c for c in validated if c.get("match_status") == "CHAPTER_99"]),
            "docket_resolved": len([c for c in validated if c.get("match_status") == "DOCKET_RESOLVED"]),
            "needs_external": len([c for c in validated if str(c.get("match_status", "")).startswith("NEEDS_")]),
            "product_name": product_info.get("product_name"),
            "countries": product_info.get("countries"),
            "notice_rows_written": written,
        }
    except Exception as e:
        logger.error("run_extraction_pipeline_failed doc=%s err=%s", document_number, e)
        return {
            "document_number": document_number,
            "total_extracted": 0,
            "verified": 0,
            "unverified": 0,
            "chapter99": 0,
            "docket_resolved": 0,
            "needs_external": 0,
            "product_name": None,
            "countries": [],
            "notice_rows_written": 0,
            "error": str(e),
        }


def run_cbp_extraction_pipeline(
    document_number: str,
    text: str,
    conn,
    title: str = "",
) -> dict[str, Any]:
    """Direct HTS extraction for CBP notices (simpler than Federal Register — no docket resolution)."""
    try:
        # Extract HTS codes with context snippets
        records = extract_hts_codes_precise(document_number, text or "")
        product_info = extract_product_and_country(title)

        if not records:
            # No HTS codes found — still log product info if available
            status = "PRODUCT_NAME_ONLY" if product_info.get("product_name") else "NO_HTS_FOUND"
            records = [
                {
                    "document_number": document_number,
                    "hts_code": None,
                    "hts_chapter": None,
                    "context_snippet": (
                        f"Product: {product_info.get('product_name')} | "
                        f"Countries: {product_info.get('countries')}"
                    ),
                    "match_status": status,
                    "hs_level": None,
                    "raw_match": None,
                }
            ]

        # Validate HTS codes against HTS_CODES table
        validated = validate_hts_codes(records, conn)

        # Write to CBP_NOTICE_HTS_CODES
        written = _write_cbp_notice_records(conn, validated)

        return {
            "document_number": document_number,
            "total_extracted": len([c for c in validated if c.get("hts_code") and not str(c.get("hts_code")).startswith("__")]),
            "verified": len([c for c in validated if c.get("match_status") == "VERIFIED"]),
            "unverified": len([c for c in validated if c.get("match_status") == "UNVERIFIED"]),
            "chapter99": len([c for c in validated if c.get("match_status") == "CHAPTER_99"]),
            "product_name": product_info.get("product_name"),
            "countries": product_info.get("countries"),
            "notice_rows_written": written,
        }
    except Exception as e:
        logger.error("run_cbp_extraction_pipeline_failed doc=%s err=%s", document_number, e)
        return {
            "document_number": document_number,
            "total_extracted": 0,
            "verified": 0,
            "unverified": 0,
            "chapter99": 0,
            "product_name": None,
            "countries": [],
            "notice_rows_written": 0,
            "error": str(e),
        }


def _write_cbp_notice_records(conn, records: list[dict]) -> int:
    """Write HTS code records to CBP_NOTICE_HTS_CODES table."""
    if not records:
        return 0
    cur = conn.cursor()
    written = 0
    try:
        for rec in records:
            status = rec.get("match_status") or "UNVERIFIED"
            code = (rec.get("hts_code") or "").strip()
            if not code:
                # CBP_NOTICE_HTS_CODES requires non-null hts_code; store a status marker row.
                code = f"__{status}__"
            chapter = rec.get("hts_chapter") or _hts_chapter_from_code(code) or ""
            context = (rec.get("context_snippet") or "")[:2000]

            # MERGE into CBP_NOTICE_HTS_CODES with context_snippet
            cur.execute(
                """
                MERGE INTO CBP_NOTICE_HTS_CODES AS t
                USING (SELECT %s AS document_number, %s AS hts_code) AS s
                ON t.document_number = s.document_number AND t.hts_code = s.hts_code
                WHEN MATCHED THEN UPDATE SET
                    t.hts_chapter = %s,
                    t.context_snippet = %s,
                    t.match_status = %s
                WHEN NOT MATCHED THEN INSERT
                    (document_number, hts_code, hts_chapter, context_snippet, match_status)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    rec.get("document_number"),
                    code,
                    chapter,
                    context,
                    status,
                    rec.get("document_number"),
                    code,
                    chapter,
                    context,
                    status,
                ),
            )
            written += 1
        conn.commit()
    finally:
        cur.close()
    return written


__all__ = [
    "extract_hts_entities",
    "detect_external_reference",
    "extract_product_and_country",
    "resolve_via_docket",
    "validate_hts_codes",
    "update_chunks_with_hts",
    "run_extraction_pipeline",
    "run_cbp_extraction_pipeline",
    "write_notice_hts_codes",
]
