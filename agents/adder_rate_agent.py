"""
Adder Rate Agent — Pipeline Steps 4, 5, and 7

Determines the Section 301 / IEEPA / Section 232 adder rate through a 3-step process:

STEP 4 — FOOTNOTE → CHAPTER 99 LOOKUP (most authoritative)
  - Parse hts_footnotes for pattern: 9903\.\d{2}\.\d{2}
  - Query HTS_CODES for matching chapter 99 codes
  - Use highest general_rate as chapter99_adder (primary authority)
  - If no footnotes, search NOTICE_HTS_CODES tables for chapter 99 references

STEP 5 — NOTICE_HTS_CODES LOOKUP (LLM extraction from snippets)
  - Search all 5 notice tables (USTR, CBP, USITC, EOP, ITA)
  - Try progressively shorter HTS prefixes
  - Pass context_snippets to LLM with specific prompt
  - LLM returns notice_adder + document_number

STEP 7 — RATE STACKING (priority-based selection)
  - Priority order (highest to lowest):
    1. chapter99_adder (direct from HTS schedule, most reliable)
    2. notice_adder (LLM reading of FR notice snippets)
    3. regex_fallback (last resort on policy chunk text)
    4. 0.0 if nothing found
  - Special case: if FTA applied, adder=0.0 (FTA exempt)

Redis cache: 1-hour TTL keyed on (hts_code + country).
"""

import asyncio
import ast
import json
import logging
import os
import re
from typing import Dict, Any, List, Optional, Set

from agents.state import TariffState
from agents import tools

logger = logging.getLogger(__name__)

CACHE_TTL = 3_600  # 1 h
CHAP99_RE = re.compile(r"(9903\.\d{2}(?:\.\d{2}(?:\.\d{2})?)?)")

ADDER_PROMPT = """Given the Federal Register excerpts below, determine the CURRENT effective
Section 301, Section 232, or IEEPA additional duty rate for:

Product HTS code: {hts_code}
Country of origin: {country}

Return ONLY valid JSON:
{{"adder_rate": <number or null>, "document_number": "<FR doc or null>", "basis": "<one sentence>"}}

Rules:
- adder_rate is a percentage NUMBER (e.g. 25.0 for 25%) — not the base MFN rate
- If multiple rates exist across documents, return the rate from the most recently dated excerpt (dates shown in brackets like [2025-06-09])
- If a recent excerpt says rates were increased from X% to Y%, return Y%
- Section 232 duties on "all steel articles" or "all aluminum articles" apply universally
  to ALL HTS codes within that product category, regardless of specific subheading.
  If excerpts mention a duty rate on "steel articles" generally, that rate applies to
  the queried steel HTS code.
- Section 232 duties apply to ALL countries unless the excerpt specifically exempts a country.
  If no country exemption is mentioned, assume the rate applies to the queried country.
- If this country is not subject to additional duties, return adder_rate: 0
- If no excerpt contains a specific percentage rate, return adder_rate: null
- document_number must come from the excerpts — never fabricate one

Federal Register excerpts (ordered newest first):
{context}"""


def _redis():
    try:
        import redis
        c = redis.Redis(
            host=os.environ.get("REDIS_HOST", "redis"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            socket_connect_timeout=2, socket_timeout=2,
        )
        c.ping()
        return c
    except Exception:
        return None


def _cache_key(hts_code: str, country: Optional[str]) -> str:
    return f"tariffiq:adder_rate:{hts_code}:{(country or 'ALL').lower().replace(' ','_')}"


def _cache_get(hts_code: str, country: Optional[str]) -> Optional[Dict]:
    r = _redis()
    if not r:
        return None
    try:
        raw = r.get(_cache_key(hts_code, country))
        if raw:
            logger.info("adder_rate_cache_hit hts=%s country=%s", hts_code, country)
            return json.loads(raw)
    except Exception:
        pass
    return None


def _cache_set(hts_code: str, country: Optional[str], result: Dict) -> None:
    r = _redis()
    if not r:
        return
    try:
        r.setex(_cache_key(hts_code, country), CACHE_TTL, json.dumps(result))
    except Exception:
        pass


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    lines = []
    for i, chunk in enumerate(chunks[:8], start=1):
        doc = chunk.get("document_number", "UNKNOWN")
        src = chunk.get("source", "USTR").upper()
        pub = chunk.get("publication_date", "") or ""
        text = chunk.get("chunk_text", "")
        lines.append(f"[{i}] {src} | {doc} | {pub}\n{text[:500]}")
    return "\n\n".join(lines)


def _regex_fallback(chunks: List[Dict[str, Any]], hts_code: str) -> float:
    """Last-resort regex on chunk text. Returns 0.0 if nothing plausible found."""
    for chunk in chunks:
        text = chunk.get("chunk_text", "")
        escaped = re.escape(hts_code.strip())
        m = re.search(escaped + r".{0,200}?(\d{1,3}(?:\.\d+)?)\s*%", text, re.DOTALL)
        if m:
            rate = float(m.group(1))
            if 0 < rate <= 200:
                logger.info("adder_rate_regex_hts_match hts=%s rate=%.1f", hts_code, rate)
                return rate
        m2 = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", text)
        if m2:
            rate = float(m2.group(1))
            if 0 < rate <= 200:
                logger.info("adder_rate_regex_broad hts=%s rate=%.1f", hts_code, rate)
                return rate
    return 0.0


def _fetch_notice_snippets(hts_code: str, country: Optional[str]) -> List[Dict[str, Any]]:
    """
    Fetch context snippets directly from NOTICE_HTS_CODES and
    CBP_NOTICE_HTS_CODES tables for this HTS code.
    These tables store pre-extracted snippets from FR documents
    that explicitly mention this HTS code — more targeted than
    ChromaDB vector search for rate extraction.
    """
    snippets = []
    country_lower = (country or "").lower().strip()
    is_china = country_lower in ("china", "prc", "people's republic of china")

    try:
        conn = tools._sf()
        cur = conn.cursor()

        # Try full code first, then progressively shorter prefixes
        codes_to_try = [hts_code]
        parts = hts_code.split(".")
        while len(parts) > 2:
            parts = parts[:-1]
            codes_to_try.append(".".join(parts))

        for table, fr_table in [
            ("NOTICE_HTS_CODES",     "FEDERAL_REGISTER_NOTICES"),
            ("CBP_NOTICE_HTS_CODES", "CBP_FEDERAL_REGISTER_NOTICES"),
            ("NOTICE_HTS_CODES_ITC", "ITC_DOCUMENTS"),
            ("NOTICE_HTS_CODES_EOP", "EOP_DOCUMENTS"),
            ("ITA_NOTICE_HTS_CODES", "ITA_FEDERAL_REGISTER_NOTICES"),
        ]:
            for code in codes_to_try:
                try:
                    cur.execute(
                        f"""
                        SELECT n.document_number, n.context_snippet,
                               f.title, f.publication_date
                        FROM TARIFFIQ.RAW.{table} n
                        LEFT JOIN TARIFFIQ.RAW.{fr_table} f
                            ON n.document_number = f.document_number
                        WHERE n.hts_code = %s
                        ORDER BY f.publication_date DESC NULLS LAST
                        LIMIT 5
                        """,
                        (code,),
                    )
                    rows = cur.fetchall()
                    for doc_num, snippet, title, pub_date in rows:
                        if not snippet:
                            continue
                        if title and any(
                            kw in title.lower()
                            for kw in ["china", "chinese"]
                        ):
                            if not is_china:
                                continue

                        if "CBP" in table:
                            source_label = "CBP"
                        elif "ITC" in table:
                            source_label = "USITC"
                        elif "EOP" in table:
                            source_label = "EOP"
                        elif "ITA" in table:
                            source_label = "ITA"
                        else:
                            source_label = "USTR"

                        snippets.append({
                            "document_number": doc_num,
                            "chunk_text": snippet,
                            "source": source_label,
                            "publication_date": str(pub_date) if pub_date else "",
                        })
                    if snippets:
                        break
                except Exception as e:
                    logger.debug(
                        "adder_fetch_snippets_table_skip "
                        "table=%s code=%s error=%s",
                        table, code, e
                    )
                    continue
            if snippets:
                break

        cur.close()
        conn.close()
    except Exception as e:
        logger.debug("adder_fetch_snippets_error hts=%s error=%s", hts_code, e)

    return snippets


def _step4_chapter99_lookup(hts_code: str, hts_footnotes: Optional[List[str]], country: Optional[str] = None) -> tuple[Optional[float], Optional[str]]:
    """
    STEP 4: Footnote → Chapter 99 Lookup

    Sub-step 4a: Parse hts_footnotes for 9903\.\d{2}\.\d{2} pattern
    Sub-step 4b: If found, query HTS_CODES for general_rate and use highest
    Sub-step 4c: If no footnotes, search NOTICE_HTS_CODES tables for chapter 99 refs

    Returns: (chapter99_adder, chapter99_doc) or (None, None)
    """
    if not hts_code:
        return None, None

    chapter99_codes = []

    # Sub-step 4a: Parse hts_footnotes
    if hts_footnotes:
        for fn in hts_footnotes:
            # Footnotes may be str(dict) or dict or raw string
            if isinstance(fn, dict):
                value = fn.get("value", "") or ""
            elif isinstance(fn, str):
                try:
                    parsed = ast.literal_eval(fn)
                    value = parsed.get("value", "") if isinstance(parsed, dict) else fn
                except (ValueError, SyntaxError):
                    value = fn
            else:
                continue
            matches = CHAP99_RE.findall(value)
            chapter99_codes.extend(matches)
            if matches:
                logger.debug("step4_chap99_from_footnote codes=%s", matches)

    # Sub-step 4c: If no footnotes, scan NOTICE_HTS_CODES snippets for chapter 99 codes
    # IMPORTANT: filter China-specific 9903.88.xx and 9903.91.xx codes here
    # based on country — don't collect them and rely on description matching later
    if not chapter99_codes:
        country_lower_4c = (country or "").lower().strip()
        is_china_4c = country_lower_4c in ("china", "prc", "people's republic of china")
        try:
            conn = tools._sf()
            cur = conn.cursor()
            codes_to_try = [hts_code]
            parts = hts_code.split(".")
            while len(parts) > 2:
                parts = parts[:-1]
                codes_to_try.append(".".join(parts))

            for code in codes_to_try:
                for table in ["NOTICE_HTS_CODES", "CBP_NOTICE_HTS_CODES", "NOTICE_HTS_CODES_ITC", "NOTICE_HTS_CODES_EOP", "ITA_NOTICE_HTS_CODES"]:
                    try:
                        cur.execute(
                            f"SELECT context_snippet FROM TARIFFIQ.RAW.{table} WHERE hts_code = %s LIMIT 10",
                            (code,),
                        )
                        rows = cur.fetchall()
                        for row in rows:
                            snippet = row[0] if row else ""
                            matches = CHAP99_RE.findall(snippet)
                            for ch99 in matches:
                                # Skip China-specific Section 301 codes for non-China countries
                                # 9903.88.xx = Section 301 China, 9903.91.xx = IEEPA China
                                if not is_china_4c and (
                                    ch99.startswith("9903.88") or ch99.startswith("9903.91")
                                ):
                                    logger.debug(
                                        "step4c_skip_china_ch99 code=%s country=%s",
                                        ch99, country,
                                    )
                                    continue
                                chapter99_codes.append(ch99)
                        if chapter99_codes:
                            break
                    except Exception:
                        continue
                if chapter99_codes:
                    break

            cur.close()
            conn.close()
            if chapter99_codes:
                logger.debug("step4_chap99_from_notices codes=%s", chapter99_codes)
        except Exception as e:
            logger.debug("step4_notice_scan_error error=%s", e)

    # Step 4d — Search Section 301 chunks directly
    # Runs only if footnotes were empty AND notice scan found no chapter 99 codes
    if not chapter99_codes:
        chunk_result = tools.find_section301_rate_from_chunks(hts_code, country)
        if chunk_result:
            chapter99_adder = chunk_result["adder_rate"]
            chapter99_doc = chunk_result["document_number"]
            logger.info(
                "chapter99_from_chunks hts=%s rate=%.1f list=%s",
                hts_code,
                chapter99_adder,
                chunk_result["list_name"],
            )
            return chapter99_adder, chapter99_doc

    if not chapter99_codes:
        return None, None

    # Sub-step 4b: Query HTS_CODES for matching chapter 99 codes
    try:
        conn = tools._sf()
        cur = conn.cursor()

        best_rate = None
        best_code = None

        for ch99_code in chapter99_codes:
            try:
                cur.execute(
                    "SELECT general_rate, description FROM TARIFFIQ.RAW.HTS_CODES WHERE hts_code = %s LIMIT 1",
                    (ch99_code,),
                )
                row = cur.fetchone()
                if row and row[0]:
                    try:
                        rate_str = str(row[0]).strip()
                        desc_str = str(row[1] or "").lower() if len(row) > 1 else ""
                        # Skip China-specific codes for non-China countries
                        country_lower = (country or "").lower().strip()
                        is_china = country_lower in ("china", "prc", "people's republic of china")
                        if not is_china and "product of china" in desc_str:
                            logger.debug("step4_skip_china_code ch99=%s country=%s", ch99_code, country)
                            continue
                        # Handle "The duty provided in the applicable subheading + X%"
                        m = re.search(r"\+\s*(\d+(?:\.\d+)?)\s*%", rate_str)
                        if m:
                            rate = float(m.group(1))
                        else:
                            rate = float(rate_str.rstrip('%'))
                        if best_rate is None or rate > best_rate:
                            best_rate = rate
                            best_code = ch99_code
                        logger.debug("step4_hts_lookup ch99=%s rate=%.2f", ch99_code, rate)
                    except (ValueError, TypeError):
                        pass
            except Exception as e:
                logger.debug("step4_hts_lookup_error code=%s error=%s", ch99_code, e)

        cur.close()
        conn.close()

        if best_rate is not None and best_code:
            logger.info("step4_chapter99_found code=%s rate=%.2f", best_code, best_rate)
            return best_rate, best_code

    except Exception as e:
        logger.debug("step4_lookup_error error=%s", e)

    return None, None


def _step5_notice_lookup(hts_code: str, country: Optional[str]) -> tuple[Optional[float], Optional[str], Optional[str]]:
    """
    STEP 5: NOTICE_HTS_CODES Lookup

    Search all 5 notice tables, try shorter prefixes, collect snippets,
    pass to LLM for extraction.

    Returns: (notice_adder, notice_doc, notice_basis) or (None, None, None)
    """
    if not hts_code:
        return None, None, None

    snippets = []

    try:
        conn = tools._sf()
        cur = conn.cursor()

        # Try progressively shorter HTS prefixes
        codes_to_try = [hts_code]
        parts = hts_code.split(".")
        while len(parts) > 2:
            parts = parts[:-1]
            codes_to_try.append(".".join(parts))

        # Search all 5 notice tables
        for table, fr_table in [
            ("NOTICE_HTS_CODES",     "FEDERAL_REGISTER_NOTICES"),
            ("CBP_NOTICE_HTS_CODES", "CBP_FEDERAL_REGISTER_NOTICES"),
            ("NOTICE_HTS_CODES_ITC", "ITC_DOCUMENTS"),
            ("NOTICE_HTS_CODES_EOP", "EOP_DOCUMENTS"),
            ("ITA_NOTICE_HTS_CODES", "ITA_FEDERAL_REGISTER_NOTICES"),
        ]:
            for code in codes_to_try:
                try:
                    cur.execute(
                        f"""
                        SELECT n.document_number, n.context_snippet,
                               f.title, f.publication_date
                        FROM TARIFFIQ.RAW.{table} n
                        LEFT JOIN TARIFFIQ.RAW.{fr_table} f
                            ON n.document_number = f.document_number
                        WHERE n.hts_code = %s
                        ORDER BY f.publication_date DESC NULLS LAST
                        LIMIT 5
                        """,
                        (code,),
                    )
                    rows = cur.fetchall()
                    country_lower_s5 = (country or "").lower().strip()
                    is_china_s5 = country_lower_s5 in ("china", "prc", "people's republic of china")
                    added_from_this_table = 0
                    for doc_num, snippet, title, pub_date in rows:
                        if not snippet:
                            continue
                        # Skip China-specific docs for non-China countries
                        if not is_china_s5 and title and any(
                            kw in (title or "").lower() for kw in ["china", "chinese", "people's republic"]
                        ):
                            continue
                        snippets.append({
                            "document_number": doc_num,
                            "chunk_text": snippet,
                            "source": (
                                "CBP" if "CBP" in table else
                                "USITC" if "ITC" in table else
                                "EOP" if "EOP" in table else
                                "ITA" if "ITA" in table else
                                "USTR"
                            ),
                            "publication_date": str(pub_date) if pub_date else "",
                        })
                        added_from_this_table += 1
                    # Only break to next table if we actually added non-filtered snippets
                    if added_from_this_table > 0:
                        break
                except Exception as e:
                    logger.debug("step5_table_error table=%s code=%s error=%s", table, code, e)
                    continue
            # Continue to next table regardless — accumulate from all sources
            # Stop only when we have enough snippets (8+) for good LLM context
            if len(snippets) >= 8:
                break

        cur.close()
        conn.close()
    except Exception as e:
        logger.debug("step5_query_error error=%s", e)

    if not snippets:
        logger.info("step5_no_snippets hts=%s", hts_code)
        return None, None, None

    logger.info("step5_snippets_found hts=%s count=%d", hts_code, len(snippets))

    # Build context for LLM
    context = _build_context(snippets)

    # LLM call
    try:
        from services.llm.router import get_router, TaskType
        router = get_router()
        loop = asyncio.new_event_loop()
        try:
            resp = loop.run_until_complete(
                router.complete(
                    task=TaskType.POLICY_ANALYSIS,
                    messages=[{
                        "role": "user",
                        "content": ADDER_PROMPT.format(
                            hts_code=hts_code,
                            country=country or "unspecified",
                            context=context,
                        ),
                    }],
                )
            )
        finally:
            loop.close()

        raw = re.sub(r"```(?:json)?", "", resp.choices[0].message.content.strip()).strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            logger.warning("step5_llm_no_json hts=%s", hts_code)
            return None, None, None

        parsed = json.loads(raw[start:end+1])

        raw_rate = parsed.get("adder_rate")
        raw_doc = parsed.get("document_number")
        basis = parsed.get("basis", "").strip()

        if raw_rate is not None:
            try:
                rate_val = float(raw_rate)
                if raw_doc:
                    for prefix in ("FR: ", "FR:", "FR "):
                        if raw_doc.startswith(prefix):
                            raw_doc = raw_doc[len(prefix):].strip()
                if 0 <= rate_val <= 200 and raw_doc:
                    logger.info("step5_notice_found hts=%s rate=%.2f doc=%s", hts_code, rate_val, raw_doc)
                    return rate_val, raw_doc, basis
            except (ValueError, TypeError):
                pass

    except Exception as e:
        logger.debug("step5_llm_error hts=%s error=%s", hts_code, e)

    return None, None, None


def _step4b_global_adder_lookup(
    hts_code: str,
    country: Optional[str],
) -> tuple[Optional[float], Optional[str]]:
    """
    Global adder lookup for non-China countries via ChromaDB + LLM.
    No SQL, no hardcoded document numbers.
    Searches ChromaDB for relevant policy chunks, passes to LLM for rate extraction.
    """
    if not hts_code or not country:
        return None, None

    country_lower = (country or "").lower().strip()
    if country_lower in ("china", "prc", "people's republic of china"):
        return None, None

    hts_2digit = hts_code.replace(".", "")[:2]

    CHAPTER_PRODUCT = {
        "72": "steel articles",
        "73": "steel articles",
        "76": "aluminum articles",
        "74": "copper articles",
        "44": "timber lumber wood products",
        "87": "motor vehicles automobiles",
        "84": "machinery computers electronics",
        "85": "electrical equipment electronics",
        "61": "apparel clothing",
        "62": "apparel clothing woven",
        "94": "furniture",
        "95": "toys games",
    }
    product_label = CHAPTER_PRODUCT.get(hts_2digit, "imported goods")

    chunks = tools.fetch_adder_chunks_from_all_agencies(
        hts_code, country, product_label
    )

    if not chunks:
        logger.info("step4b_no_chunks hts=%s country=%s", hts_code, country)
        return None, None

    logger.info(
        "step4b_chunks_found hts=%s country=%s chunks=%d",
        hts_code,
        country,
        len(chunks),
    )

    from services.llm.router import get_router, TaskType

    context = _build_context(chunks)
    router = get_router()

    try:
        loop = asyncio.new_event_loop()
        try:
            resp = loop.run_until_complete(
                router.complete(
                    task=TaskType.POLICY_ANALYSIS,
                    messages=[
                        {
                            "role": "user",
                            "content": ADDER_PROMPT.format(
                                hts_code=hts_code,
                                country=country or "unspecified",
                                context=context,
                            ),
                        }
                    ],
                )
            )
        finally:
            loop.close()

        raw = re.sub(
            r"```(?:json)?", "", resp.choices[0].message.content.strip()
        ).strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            parsed = json.loads(raw[start : end + 1])
            raw_rate = parsed.get("adder_rate")
            raw_doc = parsed.get("document_number")
            # Strip agency prefix from document number if present
            if raw_doc:
                raw_doc = str(raw_doc)
                for prefix in (
                    "FR: ",
                    "FR:",
                    "FR ",
                    "EOP ",
                    "USTR ",
                    "CBP ",
                    "ITA ",
                    "USITC ",
                    "EOP: ",
                    "USTR: ",
                ):
                    if raw_doc.startswith(prefix):
                        raw_doc = raw_doc[len(prefix):].strip()
                        break
            if raw_rate is not None:
                try:
                    rate_val = float(raw_rate)
                    if 0 < rate_val <= 200:
                        doc_str = str(raw_doc) if raw_doc else None
                        logger.info(
                            "step4b_llm_found hts=%s country=%s rate=%.1f doc=%s",
                            hts_code,
                            country,
                            rate_val,
                            raw_doc,
                        )
                        return rate_val, doc_str
                except (ValueError, TypeError):
                    pass

    except Exception as e:
        logger.debug("step4b_llm_error hts=%s error=%s", hts_code, e)

    return None, None


def _step4c_universal_surcharge_lookup(
    hts_code: str,
    country: Optional[str],
    fta_applied: bool,
) -> tuple[Optional[float], Optional[str]]:
    """
    Universal country surcharge lookup (e.g., balance-of-payments style actions).
    Uses ChromaDB + LLM extraction with the same JSON contract as other adder lookups.
    """
    if not hts_code or not country:
        return None, None

    country_lower = (country or "").lower().strip()
    hts_2digit = hts_code.replace(".", "")[:2]

    # Skip USMCA countries when FTA is applied.
    if fta_applied and country_lower in ("canada", "mexico"):
        logger.info("step4c_skip_usmca hts=%s country=%s", hts_code, country)
        return None, None

    # Skip Section 232 product chapters.
    if hts_2digit in ("72", "73", "74", "76"):
        logger.info("step4c_skip_section232_product hts=%s chapter=%s", hts_code, hts_2digit)
        return None, None

    try:
        from services.retrieval.hybrid import get_retriever
        from services.llm.router import get_router, TaskType

        retriever = get_retriever()
        query = (
            f"import surcharge balance of payments {country} "
            "all products ad valorem temporary"
        )
        chunks = retriever.search_policy(
            query=query,
            hts_chapter=None,
            source=None,
            top_k=15,
        )
        if not chunks:
            logger.info("step4c_no_chunks hts=%s country=%s", hts_code, country)
            return None, None

        logger.info(
            "step4c_chunks_found hts=%s country=%s chunks=%d",
            hts_code,
            country,
            len(chunks),
        )

        context = _build_context(chunks)
        router = get_router()
        loop = asyncio.new_event_loop()
        try:
            resp = loop.run_until_complete(
                router.complete(
                    task=TaskType.POLICY_ANALYSIS,
                    messages=[
                        {
                            "role": "user",
                            "content": ADDER_PROMPT.format(
                                hts_code=hts_code,
                                country=country or "unspecified",
                                context=context,
                            ),
                        }
                    ],
                )
            )
        finally:
            loop.close()

        raw = re.sub(
            r"```(?:json)?", "", resp.choices[0].message.content.strip()
        ).strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            parsed = json.loads(raw[start : end + 1])
            raw_rate = parsed.get("adder_rate")
            raw_doc = parsed.get("document_number")
            if raw_doc:
                raw_doc = str(raw_doc)
                for prefix in (
                    "FR: ",
                    "FR:",
                    "FR ",
                    "EOP ",
                    "USTR ",
                    "CBP ",
                    "ITA ",
                    "USITC ",
                    "EOP: ",
                    "USTR: ",
                ):
                    if raw_doc.startswith(prefix):
                        raw_doc = raw_doc[len(prefix):].strip()
                        break
            if raw_rate is not None:
                try:
                    rate_val = float(raw_rate)
                    if 0 < rate_val <= 200:
                        doc_str = str(raw_doc) if raw_doc else None
                        logger.info(
                            "step4c_universal_found hts=%s country=%s rate=%.1f doc=%s",
                            hts_code,
                            country,
                            rate_val,
                            raw_doc,
                        )
                        return rate_val, doc_str
                except (ValueError, TypeError):
                    pass
    except Exception as e:
        logger.debug("step4c_universal_error hts=%s country=%s error=%s", hts_code, country, e)

    return None, None


def run_adder_rate_agent(state: TariffState) -> Dict[str, Any]:
    hts_code = (state.get("hts_code") or "").strip()
    country = state.get("country")
    base_rate = state.get("base_rate") or 0.0
    fta_applied = state.get("fta_applied", False)
    fta_rate = state.get("fta_rate")
    mfn_rate = state.get("mfn_rate") or base_rate
    hts_footnotes = state.get("hts_footnotes") or []
    policy_chunks = state.get("policy_chunks") or []

    if not hts_code:
        return {
            "chapter99_adder": None, "chapter99_doc": None,
            "notice_adder": None, "notice_doc": None, "notice_basis": None,
            "adder_rate": 0.0, "adder_doc": None, "adder_basis": "none",
            "total_duty": base_rate,
        }

    logger.info("adder_rate_agent_start hts=%s country=%s fta=%s", hts_code, country, fta_applied)

    # Cache check (recompute total_duty with current base_rate)
    cached = _cache_get(hts_code, country)
    if cached:
        adder = cached.get("adder_rate") or 0.0
        section122 = cached.get("section122_adder") or 0.0
        cached["total_duty"] = round(base_rate + adder + section122, 4)
        return cached

    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 4: CHAPTER 99 FOOTNOTE LOOKUP (highest priority, most authoritative)
    # ─────────────────────────────────────────────────────────────────────────────
    chapter99_adder, chapter99_doc = _step4_chapter99_lookup(hts_code, hts_footnotes, country)

    # ── STEP 4b: DATABASE-DRIVEN GLOBAL ADDER LOOKUP ─────────────────────────────
    # Fires when footnote lookup found nothing.
    # Catches Section 232 steel/aluminum and IEEPA reciprocal tariffs
    # that apply globally and don't appear in product-level footnotes.
    if chapter99_adder is None:
        chapter99_adder, chapter99_doc = _step4b_global_adder_lookup(hts_code, country)

    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 5: NOTICE_HTS_CODES LOOKUP (LLM-extracted from FR snippets)
    # ─────────────────────────────────────────────────────────────────────────────
    notice_adder, notice_doc, notice_basis = _step5_notice_lookup(hts_code, country)

    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 7: RATE STACKING (priority-based selection)
    # ─────────────────────────────────────────────────────────────────────────────

    # Special case: if FTA applied, adder is 0 (FTA exempt)
    if fta_applied:
        final_adder = 0.0
        final_doc = None
        final_basis = "fta_exempt"
        effective_base = fta_rate or 0.0
        logger.info("step7_fta_exempt hts=%s fta_rate=%.2f", hts_code, effective_base)
    else:
        # Priority order: chapter99 > notice > regex > 0.0
        if chapter99_adder is not None:
            final_adder = chapter99_adder
            final_doc = chapter99_doc
            final_basis = "chapter99"
            logger.info("step7_chapter99_selected hts=%s rate=%.2f", hts_code, final_adder)
        elif notice_adder is not None:
            final_adder = notice_adder
            final_doc = notice_doc
            final_basis = "notice_llm"
            logger.info("step7_notice_selected hts=%s rate=%.2f doc=%s", hts_code, final_adder, final_doc)
        else:
            final_adder = 0.0
            final_doc = None
            final_basis = "none"
            logger.info("step7_no_adder hts=%s country=%s", hts_code, country)

        effective_base = mfn_rate

    # Only check Section 122 if we found a primary adder
    # (Section 122 stacks on top of existing duties, not standalone)
    if final_adder > 0 and not fta_applied:
        section122_adder, section122_doc = _step4c_universal_surcharge_lookup(
            hts_code=hts_code,
            country=country,
            fta_applied=bool(fta_applied),
        )
    else:
        section122_adder, section122_doc = None, None

    stacked_surcharge = section122_adder or 0.0
    total_duty = round(effective_base + final_adder + stacked_surcharge, 4)

    logger.info(
        "adder_rate_agent_done hts=%s country=%s ch99=%s notice=%s section122=%s final=%.4f basis=%s total=%.4f",
        hts_code,
        country,
        chapter99_adder,
        notice_adder,
        section122_adder,
        final_adder,
        final_basis,
        total_duty,
    )

    result = {
        "chapter99_adder": chapter99_adder,
        "chapter99_doc": chapter99_doc,
        "notice_adder": notice_adder,
        "notice_doc": notice_doc,
        "notice_basis": notice_basis,
        "adder_rate": round(final_adder, 4),
        "section122_adder": section122_adder,
        "section122_doc": section122_doc,
        "adder_doc": final_doc,
        "adder_basis": final_basis,
        "adder_method": final_basis,  # Backwards compat
        "total_duty": total_duty,
    }
    _cache_set(hts_code, country, result)
    return result