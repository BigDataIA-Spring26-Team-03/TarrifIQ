import json
import logging
import re
from typing import Any

from ingestion import hts_extractor as base

logger = logging.getLogger(__name__)

extract_hts_entities = base.extract_hts_entities
detect_external_reference = base.detect_external_reference
extract_product_and_country = base.extract_product_and_country
validate_hts_codes = base.validate_hts_codes
update_chunks_with_hts = base.update_chunks_with_hts
_to_code_records = base._to_code_records
_extract_annex_section = base._extract_annex_section
_extract_docket_number = base._extract_docket_number
_hts_chapter_from_code = base._hts_chapter_from_code


# Loose HTS/US-style dotted codes (captures Annex / list / legal boilerplate misses)
_LOOSE_HTS_RE = re.compile(r"(?<!\d)\d{4}\.\d{2}(?:\.\d{2}){0,2}(?!\d)")

# At least HS subheading-level (XXXX.XX.XX)
_HAS_SUBHEADING_RE = re.compile(r"\d{4}\.\d{2}\.\d{2}")

_CHAPTER_PAIR_RE = re.compile(
    r"\bChapters?\s+(\d{1,2})\s+and\s+(\d{1,2})\b",
    re.IGNORECASE,
)
_CHAPTER_SINGLE_RE = re.compile(r"\bChapter\s+(\d{1,2})\b", re.IGNORECASE)


def _snippet(text: str, start: int, end: int, width: int = 500) -> str:
    if not text:
        return ""
    lo = max(0, start - width)
    hi = min(len(text), end + width)
    return text[lo:hi].strip()[:2000]


def _has_subheading_codes(records: list[dict]) -> bool:
    for r in records:
        c = (r.get("hts_code") or "").strip()
        if not c or str(c).startswith("__"):
            continue
        if _HAS_SUBHEADING_RE.search(c):
            return True
    return False


def _annex_focused_text(text: str) -> tuple[str, bool]:
    """
    If the body references an Annex or Chapter 99 subchapter III, prioritize extraction
    from that portion of the document (usually where HTS lists live).
    """
    if not text:
        return "", False
    ch99 = re.search(r"subchapter\s+III\s+of\s+chapter\s+99", text, re.IGNORECASE)
    if ch99:
        return text[ch99.start() :], True
    annex_m = re.search(r"\bAnnex\b", text, re.IGNORECASE)
    if annex_m:
        return text[annex_m.start() :], True
    return text, False


def _extract_loose_hts(
    document_number: str,
    text: str,
    *,
    from_annex: bool = False,
) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for m in _LOOSE_HTS_RE.finditer(text or ""):
        code = m.group(0)
        if code in seen:
            continue
        seen.add(code)
        rec = {
            "document_number": document_number,
            "hts_code": code,
            "hts_chapter": _hts_chapter_from_code(code),
            "context_snippet": _snippet(text, m.start(), m.end()),
            "match_status": "LOOSE_REGEX",
            "hs_level": None,
            "raw_match": code,
        }
        if from_annex:
            rec["_from_annex"] = True
        out.append(rec)
    return out


def _extract_chapter_level_records(document_number: str, text: str) -> list[dict]:
    """Chapter-only references when no subheading-level codes exist."""
    rows: list[dict] = []
    seen_ch: set[str] = set()
    hay = text or ""

    for m in _CHAPTER_PAIR_RE.finditer(hay):
        for g in (m.group(1), m.group(2)):
            ch = f"{int(g):02d}"
            if ch in seen_ch:
                continue
            seen_ch.add(ch)
            rows.append(
                {
                    "document_number": document_number,
                    "hts_code": f"Chapter {ch}",
                    "hts_chapter": ch,
                    "context_snippet": _snippet(hay, m.start(), m.end()),
                    "match_status": "CHAPTER_LEVEL_ONLY",
                    "hs_level": None,
                    "raw_match": m.group(0),
                }
            )

    for m in _CHAPTER_SINGLE_RE.finditer(hay):
        ch = f"{int(m.group(1)):02d}"
        if ch in seen_ch:
            continue
        seen_ch.add(ch)
        rows.append(
            {
                "document_number": document_number,
                "hts_code": f"Chapter {ch}",
                "hts_chapter": ch,
                "context_snippet": _snippet(hay, m.start(), m.end()),
                "match_status": "CHAPTER_LEVEL_ONLY",
                "hs_level": None,
                "raw_match": m.group(0),
            }
        )

    return rows


def _merge_code_records(primary: list[dict], extra: list[dict]) -> list[dict]:
    """Dedupe by hts_code; prefer earlier / higher-priority rows."""
    seen: set[str] = set()
    out: list[dict] = []
    for bucket in (primary, extra):
        for r in bucket:
            code = (r.get("hts_code") or "").strip()
            if not code:
                continue
            if code in seen:
                continue
            seen.add(code)
            out.append(r)
    return out


def resolve_via_docket(document_number: str, docket_number: str, conn) -> list[dict]:
    if not docket_number:
        logger.info("DOCKET_NOT_RESOLVED doc=%s reason=no_docket", document_number)
        return []
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT document_number
            FROM EOP_DOCUMENTS
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
            FROM NOTICE_HTS_CODES_EOP
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


def resolve_via_executive_citations(
    document_number: str,
    text: str,
    title: str,
    conn,
) -> list[dict]:
    """
    Resolve HTS rows from another EOP document cited as Proclamation N or Executive Order N.
    """
    hay = f"{title or ''}\n{text or ''}"
    queries: list[tuple[str, str]] = []

    for m in re.finditer(r"Proclamation\s+(?:No\.?\s*)?(\d+)", hay, re.IGNORECASE):
        queries.append(("proclamation", m.group(1)))
    for m in re.finditer(r"Executive\s+Order\s+(?:No\.?\s*)?(\d+)", hay, re.IGNORECASE):
        queries.append(("eo", m.group(1)))
    for m in re.finditer(r"E\.O\.\s*(?:No\.?\s*)?(\d+)\b", hay, re.IGNORECASE):
        queries.append(("eo", m.group(1)))

    seen_pairs: set[tuple[str, str]] = set()
    unique_queries: list[tuple[str, str]] = []
    for kind, num in queries:
        key = (kind, num)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        unique_queries.append((kind, num))

    out: list[dict] = []
    seen_codes: set[str] = set()
    cur = conn.cursor()
    try:
        for kind, num in unique_queries:
            if kind == "proclamation":
                needle = f"%Proclamation {num}%"
                status_ok = "PROCLAMATION_RESOLVED"
                label = f"Proclamation {num}"
            else:
                needle = f"%Executive Order {num}%"
                status_ok = "EXEC_ORDER_RESOLVED"
                label = f"Executive Order {num}"

            cur.execute(
                """
                SELECT document_number
                FROM EOP_DOCUMENTS
                WHERE document_number <> %s
                  AND processing_status IN ('hts_extracted', 'chunked')
                  AND (
                        title ILIKE %s
                     OR raw_json::text ILIKE %s
                  )
                ORDER BY publication_date ASC
                LIMIT 5
                """,
                (document_number, needle, needle),
            )
            candidates = [r[0] for r in (cur.fetchall() or []) if r and r[0]]

            if not candidates and kind == "eo":
                needle2 = f"%E.O.%{num}%"
                cur.execute(
                    """
                    SELECT document_number
                    FROM EOP_DOCUMENTS
                    WHERE document_number <> %s
                      AND processing_status IN ('hts_extracted', 'chunked')
                      AND (title ILIKE %s OR raw_json::text ILIKE %s)
                    ORDER BY publication_date ASC
                    LIMIT 5
                    """,
                    (document_number, needle2, needle2),
                )
                candidates = [r[0] for r in (cur.fetchall() or []) if r and r[0]]

            resolved_this_ref = False
            for src in candidates:
                cur.execute(
                    """
                    SELECT hts_code, hts_chapter, context_snippet
                    FROM NOTICE_HTS_CODES_EOP
                    WHERE document_number = %s
                    """,
                    (src,),
                )
                rows = cur.fetchall() or []
                if not rows:
                    continue
                for hts_code, hts_chapter, context_snippet in rows:
                    if not hts_code:
                        continue
                    hc = str(hts_code).strip()
                    if hc in seen_codes:
                        continue
                    seen_codes.add(hc)
                    out.append(
                        {
                            "document_number": document_number,
                            "hts_code": hc,
                            "hts_chapter": hts_chapter,
                            "context_snippet": (
                                (context_snippet or "")[:1500]
                                + f" | Resolved via {label} (source doc {src})"
                            )[:2000],
                            "match_status": status_ok,
                            "hs_level": None,
                            "raw_match": label,
                        }
                    )
                    resolved_this_ref = True
                if resolved_this_ref:
                    break
    except Exception as e:
        logger.error("resolve_via_executive_citations_failed doc=%s err=%s", document_number, e)
    finally:
        cur.close()

    return out


def _lookup_title_and_agency(document_number: str, conn) -> tuple[str, str, str | None]:
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT title, agency_names, raw_json
            FROM EOP_DOCUMENTS
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


def _write_notice_records(conn, records: list[dict]) -> int:
    if not records:
        return 0
    cur = conn.cursor()
    written = 0
    try:
        for rec in records:
            rec = {k: v for k, v in rec.items() if not str(k).startswith("_")}
            status = rec.get("match_status") or "UNVERIFIED"
            code = (rec.get("hts_code") or "").strip()
            if not code:
                code = f"__{status}__"
            chapter = rec.get("hts_chapter") or _hts_chapter_from_code(code) or ""
            context = (rec.get("context_snippet") or "")[:2000]
            cur.execute(
                """
                MERGE INTO NOTICE_HTS_CODES_EOP AS t
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


def _strip_internal_keys(records: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if not str(k).startswith("_")} for r in records]


def run_extraction_pipeline(
    document_number: str,
    text: str,
    conn,
    title: str = "",
    agency: str = "",
    docket_number: str | None = None,
) -> dict[str, Any]:
    try:
        if not title or not agency or not docket_number:
            t, a, d = _lookup_title_and_agency(document_number, conn)
            title = title or t
            agency = agency or a
            docket_number = docket_number or d

        full_text = text or ""
        focus_text, annex_priority = _annex_focused_text(full_text)

        # 1) Precise + loose on annex/ch99-focused text first
        records = base.extract_hts_codes_precise(document_number, focus_text)
        loose_annex = _extract_loose_hts(document_number, focus_text, from_annex=annex_priority)
        records = _merge_code_records(records, loose_annex)

        # 2) If focus is short/empty or still sparse, scan full document (codes often in body + annex)
        if annex_priority:
            records = _merge_code_records(
                records,
                base.extract_hts_codes_precise(document_number, full_text),
            )
            records = _merge_code_records(
                records,
                _extract_loose_hts(document_number, full_text, from_annex=False),
            )
        elif not records:
            records = _merge_code_records(
                records,
                base.extract_hts_codes_precise(document_number, full_text),
            )
            records = _merge_code_records(
                records,
                _extract_loose_hts(document_number, full_text, from_annex=False),
            )

        product_info = extract_product_and_country(title)

        # 3) Executive cross-references (Proclamation / EO) when subheading list still missing
        if not _has_subheading_codes(records):
            records = _merge_code_records(
                records,
                resolve_via_executive_citations(document_number, full_text, title, conn),
            )

        # 4) Chapter-level intelligence if still no subheading-level codes
        if not _has_subheading_codes(records):
            records = _merge_code_records(records, _extract_chapter_level_records(document_number, full_text))

        # 5) Legacy external-reference path (docket, same-doc annex, NEEDS_* placeholders)
        if not _has_subheading_codes(records):
            ref_info = detect_external_reference(full_text, title, agency)
            if ref_info["has_external_ref"]:
                if ref_info["resolution_strategy"] == "DOCKET_LOOKUP" and docket_number:
                    records = _merge_code_records(records, resolve_via_docket(document_number, docket_number, conn))
                elif ref_info["resolution_strategy"] == "SAME_DOC_ANNEX":
                    annex_text = _extract_annex_section(full_text)
                    if annex_text:
                        annex_entities = extract_hts_entities(annex_text)
                        records = _merge_code_records(
                            records,
                            _to_code_records(document_number, annex_text, annex_entities),
                        )
                        records = _merge_code_records(
                            records,
                            _extract_loose_hts(document_number, annex_text, from_annex=True),
                        )

                if not _has_subheading_codes(records):
                    needs_status = {
                        "DECISION_MEMO": "NEEDS_DECISION_MEMO",
                        "ANNEX": "NEEDS_ANNEX",
                        "CBP_RULING": "NEEDS_CBP_RULING",
                        "COMMISSION_REPORT": "NEEDS_COMMISSION_REPORT",
                    }.get(ref_info.get("ref_type"), "NEEDS_EXTERNAL_REF")
                    if not any((r.get("hts_code") or "").strip() for r in records):
                        records.append(
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
                        )
            elif not records:
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

        # Validate Snowflake HTS for dotted codes; passthrough chapter / exec-resolve copies
        passthrough_statuses = {
            "CHAPTER_LEVEL_ONLY",
            "PROCLAMATION_RESOLVED",
            "EXEC_ORDER_RESOLVED",
        }
        to_validate = [r for r in records if r.get("match_status") not in passthrough_statuses]
        validated = validate_hts_codes(to_validate, conn)
        validated.extend([r for r in records if r.get("match_status") in passthrough_statuses])

        validated = _strip_internal_keys(validated)
        written = _write_notice_records(conn, validated)

        annex_extracted = sum(
            1
            for r in records
            if r.get("_from_annex") and (r.get("hts_code") or "").strip()
        )
        chapter_level_only = sum(1 for r in validated if r.get("match_status") == "CHAPTER_LEVEL_ONLY")

        return {
            "document_number": document_number,
            "total_extracted": len(
                [c for c in validated if c.get("hts_code") and not str(c.get("hts_code")).startswith("__")]
            ),
            "verified": len([c for c in validated if c.get("match_status") == "VERIFIED"]),
            "unverified": len([c for c in validated if c.get("match_status") == "UNVERIFIED"]),
            "chapter99": len([c for c in validated if c.get("match_status") == "CHAPTER_99"]),
            "docket_resolved": len([c for c in validated if c.get("match_status") == "DOCKET_RESOLVED"]),
            "needs_external": len([c for c in validated if str(c.get("match_status", "")).startswith("NEEDS_")]),
            "chapter_level_only": chapter_level_only,
            "annex_extracted": annex_extracted,
            "loose_regex": len([c for c in validated if c.get("match_status") == "LOOSE_REGEX"]),
            "proclamation_resolved": len([c for c in validated if c.get("match_status") == "PROCLAMATION_RESOLVED"]),
            "exec_order_resolved": len([c for c in validated if c.get("match_status") == "EXEC_ORDER_RESOLVED"]),
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
            "chapter_level_only": 0,
            "annex_extracted": 0,
            "loose_regex": 0,
            "proclamation_resolved": 0,
            "exec_order_resolved": 0,
            "product_name": None,
            "countries": [],
            "notice_rows_written": 0,
            "error": str(e),
        }


__all__ = [
    "extract_hts_entities",
    "detect_external_reference",
    "extract_product_and_country",
    "resolve_via_docket",
    "resolve_via_executive_citations",
    "validate_hts_codes",
    "update_chunks_with_hts",
    "run_extraction_pipeline",
]
