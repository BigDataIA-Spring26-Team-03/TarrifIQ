"""
Adder Rate Agent — Pipeline Step 5

Determines the Section 301 / IEEPA / Section 232 adder rate by having
the LLM read the actual policy chunks retrieved by policy_agent.

Uses ModelRouter(TaskType.POLICY_ANALYSIS) — the same model (claude-haiku)
and system prompt that policy_agent uses, which is correct since this is
also a policy document reading task.

WHY THIS AGENT EXISTS
──────────────────────
The old rate_agent computed the adder by running a regex on a short
context_snippet stored in NOTICE_HTS_CODES. That snippet often contains
multiple percentages (2018 rate, 2019 escalation, 2025 IEEPA rate) and
the regex blindly returned the first one.

This agent reads the full policy chunks with an LLM and asks specifically:
"what is the CURRENT effective adder for HTS X from country Y?"
Falls back to regex on chunk text if the LLM fails or returns null.

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
- If multiple rates exist, return the MOST RECENT currently effective one
- If this country is not subject to additional duties, return adder_rate: 0
- If context is insufficient, return adder_rate: null
- document_number must come from the excerpts — never fabricate one

Federal Register excerpts:
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
        if result.get("adder_method") in ("llm_policy", "none"):
            r.setex(_cache_key(hts_code, country), CACHE_TTL, json.dumps(result))
    except Exception:
        pass


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    lines = []
    for i, chunk in enumerate(chunks[:5], start=1):
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


def run_adder_rate_agent(state: TariffState) -> Dict[str, Any]:
    hts_code = (state.get("hts_code") or "").strip()
    country = state.get("country")
    base_rate = state.get("base_rate") or 0.0
    policy_chunks = state.get("policy_chunks") or []

    if not hts_code:
        return {
            "adder_rate": 0.0, "adder_doc": None,
            "adder_method": "none",
            "total_duty": base_rate,
        }

    logger.info("adder_rate_agent_start hts=%s country=%s chunks=%d",
                hts_code, country, len(policy_chunks))

    # Cache check (recompute total_duty with current base_rate)
    cached = _cache_get(hts_code, country)
    if cached:
        adder = cached.get("adder_rate") or 0.0
        cached["total_duty"] = round(base_rate + adder, 4)
        return cached

    # ── Step 0: Chapter 99 footnote lookup (most reliable, pure SQL) ────────
    # HTS_CODES footnotes contain Chapter 99 references like "See 9903.88.15"
    # These map directly to the adder rate without any LLM or regex needed.
    hts_footnotes = state.get("hts_footnotes") or []
    chapter99_codes = []
    for fn in hts_footnotes:
        # Footnotes are stored as str(dict) with single quotes — use ast.literal_eval
        if isinstance(fn, dict):
            value = fn.get("value", "") or ""
        elif isinstance(fn, str):
            try:
                parsed = ast.literal_eval(fn)
                value = parsed.get("value", "") if isinstance(parsed, dict) else fn
            except (ValueError, SyntaxError):
                value = fn  # fallback: run regex on raw string
        else:
            continue
        matches = CHAP99_RE.findall(value)
        chapter99_codes.extend(matches)
        if matches:
            logger.info("chap99_footnote_parsed value=%s codes=%s", value.strip(), matches)

    if chapter99_codes:
        ch99_result = tools.chapter99_lookup(chapter99_codes, country=country)
        if ch99_result and ch99_result.get("adder_rate") == -1.0:
            result = {
                "adder_rate": 0.0,
                "adder_specific_duty": ch99_result.get("adder_specific_duty", ""),
                "adder_method": "chapter99_specific_duty",
                "adder_doc": ch99_result.get("chapter99_code", ""),
                "total_duty": round(base_rate, 4),
            }
            logger.info("chap99_specific_duty hts=%s duty=%s", hts_code, result["adder_specific_duty"])
            _cache_set(hts_code, country, result)
            return result
        if ch99_result and ch99_result.get("adder_rate", 0) > 0:
            adder_val = ch99_result["adder_rate"]
            ch99_code = ch99_result["chapter99_code"]
            total = round(base_rate + adder_val, 4)
            logger.info("adder_rate_chapter99 hts=%s country=%s code=%s rate=%.1f",
                        hts_code, country, ch99_code, adder_val)
            result = {
                "adder_rate": adder_val,
                "adder_doc": ch99_code,
                "adder_method": "chapter99_lookup",
                "total_duty": total,
            }
            _cache_set(hts_code, country, result)
            return result

    # ── Step 1: Extract Chapter 99 codes from NOTICE_HTS_CODES snippets ────
    # Even if hts_footnotes is empty, NOTICE_HTS_CODES context_snippets
    # often contain Chapter 99 code references like "9903.88.03" or "9903.90.07"
    # Extract those and look them up directly in HTS_CODES for the rate.
    if not chapter99_codes:
        notice_ch99 = tools.fetch_chapter99_from_notices(hts_code)
        if notice_ch99:
            chapter99_codes.extend(notice_ch99)
            ch99_result = tools.chapter99_lookup(chapter99_codes, country=country)
            if ch99_result and ch99_result.get("adder_rate") == -1.0:
                result = {
                    "adder_rate": 0.0,
                    "adder_specific_duty": ch99_result.get("adder_specific_duty", ""),
                    "adder_method": "chapter99_specific_duty",
                    "adder_doc": ch99_result.get("chapter99_code", ""),
                    "total_duty": round(base_rate, 4),
                }
                logger.info("chap99_specific_duty hts=%s duty=%s", hts_code, result["adder_specific_duty"])
                _cache_set(hts_code, country, result)
                return result
            if ch99_result and ch99_result.get("adder_rate", 0) > 0:
                adder_val = ch99_result["adder_rate"]
                ch99_code = ch99_result["chapter99_code"]
                total = round(base_rate + adder_val, 4)
                logger.info("adder_rate_notice_ch99 hts=%s country=%s code=%s rate=%.1f",
                            hts_code, country, ch99_code, adder_val)
                result = {
                    "adder_rate": adder_val,
                    "adder_doc": ch99_code,
                    "adder_method": "chapter99_lookup",
                    "total_duty": total,
                }
                _cache_set(hts_code, country, result)
                return result

    # ── Step 2: Fetch targeted snippets from NOTICE_HTS_CODES + CBP tables ──
    notice_snippets = _fetch_notice_snippets(hts_code, country)
    if notice_snippets:
        logger.info("adder_rate_notice_snippets hts=%s count=%d", hts_code, len(notice_snippets))

    # Combine: notice snippets first (more targeted), then ChromaDB chunks
    all_chunks = notice_snippets + policy_chunks

    if not all_chunks:
        result = {"adder_rate": 0.0, "adder_doc": None, "adder_method": "none",
                  "total_duty": round(base_rate, 4)}
        _cache_set(hts_code, country, result)
        return result

    valid_docs: Set[str] = {
        c.get("document_number", "") for c in all_chunks
        if c.get("document_number")
    }
    # Use all_chunks for context, not just policy_chunks
    policy_chunks = all_chunks
    context = _build_context(all_chunks)

    # LLM call via ModelRouter — POLICY_ANALYSIS task (claude-haiku)
    from services.llm.router import get_router, TaskType
    router = get_router()

    adder_rate: Optional[float] = None
    adder_doc: Optional[str] = None
    method = "none"

    try:
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
        # Find the first { and last } to extract the JSON object
        # Handles trailing text and nested structures from the LLM
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON object found in response: {raw[:100]}")
        parsed = json.loads(raw[start:end+1])

        raw_rate = parsed.get("adder_rate")
        raw_doc = parsed.get("document_number")

        if raw_rate is not None:
            try:
                rate_val = float(raw_rate)
                if 0 <= rate_val <= 200:
                    adder_rate = rate_val
                    method = "llm_policy"
            except (ValueError, TypeError):
                pass

        # Validate doc number against retrieved chunks only
        if raw_doc and raw_doc in valid_docs:
            adder_doc = raw_doc
        elif raw_doc:
            logger.warning("adder_rate_hallucinated_doc doc=%s — rejecting", raw_doc)

        logger.info("adder_rate_llm hts=%s country=%s rate=%s doc=%s basis=%s",
                    hts_code, country, adder_rate, adder_doc,
                    str(parsed.get("basis", ""))[:80])

    except Exception as e:
        logger.warning("adder_rate_llm_failed hts=%s error=%s", hts_code, e)

    # Regex fallback if LLM failed or returned null
    if adder_rate is None:
        adder_rate = _regex_fallback(policy_chunks, hts_code)
        method = "regex_fallback" if adder_rate > 0 else "none"

    total_duty = round(base_rate + (adder_rate or 0.0), 4)

    logger.info("adder_rate_agent_done hts=%s country=%s base=%.4f adder=%.4f total=%.4f method=%s",
                hts_code, country, base_rate, adder_rate or 0.0, total_duty, method)

    result = {
        "adder_rate": adder_rate or 0.0,
        "adder_doc": adder_doc,
        "adder_method": method,
        "total_duty": total_duty,
    }
    _cache_set(hts_code, country, result)
    return result