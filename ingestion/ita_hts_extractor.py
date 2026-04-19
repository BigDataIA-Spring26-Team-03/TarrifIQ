import json
import logging
from typing import Any, List

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


def resolve_via_docket(document_number: str, docket_number: str, conn) -> list[dict]:
    if not docket_number:
        logger.info("DOCKET_NOT_RESOLVED doc=%s reason=no_docket", document_number)
        return []
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT document_number
            FROM ITA_FEDERAL_REGISTER_NOTICES
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
            FROM ITA_NOTICE_HTS_CODES
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


def _lookup_title_and_agency(document_number: str, conn) -> tuple[str, str, str | None]:
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT title, agency_names, raw_json
            FROM ITA_FEDERAL_REGISTER_NOTICES
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
            status = rec.get("match_status") or "UNVERIFIED"
            code = (rec.get("hts_code") or "").strip()
            if not code:
                code = f"__{status}__"
            chapter = rec.get("hts_chapter") or _hts_chapter_from_code(code) or ""
            context = (rec.get("context_snippet") or "")[:2000]
            cur.execute(
                """
                MERGE INTO ITA_NOTICE_HTS_CODES AS t
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

        records = base.extract_hts_codes_precise(document_number, text or "")
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


__all__ = [
    "extract_hts_entities",
    "detect_external_reference",
    "extract_product_and_country",
    "resolve_via_docket",
    "validate_hts_codes",
    "update_chunks_with_hts",
    "run_extraction_pipeline",
]
