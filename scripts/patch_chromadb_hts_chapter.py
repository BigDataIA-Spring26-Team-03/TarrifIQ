#!/usr/bin/env python3
# Run: python scripts/patch_chromadb_hts_chapter.py
# Requires: Docker running (ChromaDB at localhost:9001), .env loaded
# Safe to re-run — updates are idempotent

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import chromadb

# Ensure project root is importable when running as:
#   python scripts/patch_chromadb_hts_chapter.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.connection import get_snowflake_conn


BATCH_SIZE = 500
COLLECTIONS = ("policy_notices", "hts_descriptions")


def _get_chroma_client() -> chromadb.HttpClient:
    host = os.environ.get("CHROMADB_HOST", "127.0.0.1")
    port = int(os.environ.get("CHROMADB_PORT", "9001"))
    return chromadb.HttpClient(host=host, port=port)


def _extract_chapters(hts_codes: List[str]) -> Tuple[str, str, Dict[str, int]]:
    chapters: Set[str] = set()
    chapter_counter: Dict[str, int] = defaultdict(int)
    normalized_codes: Set[str] = set()

    for hts in hts_codes:
        raw = str(hts or "").strip()
        if not raw:
            continue
        digits = "".join(ch for ch in raw if ch.isdigit())
        if not digits:
            continue

        chapter = digits[:2]
        if chapter.isdigit() and 1 <= int(chapter) <= 97:
            chapters.add(chapter)
            chapter_counter[chapter] += 1

        # Most specific code up to 6 digits
        normalized_codes.add(digits[:6])

    chapter_str = "|".join(sorted(chapters))
    hts_code_str = "|".join(sorted(c for c in normalized_codes if c))
    return chapter_str, hts_code_str, chapter_counter


def _lookup_doc_hts_codes(
    doc_number: str,
    source: str,
    conn,
    cache: Dict[Tuple[str, str], List[str]],
) -> List[str]:
    key = (doc_number, source)
    if key in cache:
        return cache[key]

    codes: Set[str] = set()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT DISTINCT HTS_CODE
            FROM TARIFFIQ.RAW.NOTICE_HTS_CODES
            WHERE DOCUMENT_NUMBER = %s
            """,
            (doc_number,),
        )
        for (hts_code,) in cur.fetchall():
            if hts_code:
                codes.add(str(hts_code))

        if source.upper() == "CBP":
            cur.execute(
                """
                SELECT DISTINCT HTS_CODE
                FROM TARIFFIQ.RAW.CBP_NOTICE_HTS_CODES
                WHERE DOCUMENT_NUMBER = %s
                """,
                (doc_number,),
            )
            for (hts_code,) in cur.fetchall():
                if hts_code:
                    codes.add(str(hts_code))
    finally:
        cur.close()

    out = sorted(codes)
    cache[key] = out
    return out


def _patch_collection(
    collection_name: str,
    chroma_client: chromadb.HttpClient,
    sf_conn,
    chapter_totals: Dict[str, int],
) -> Tuple[int, int, int]:
    collection = chroma_client.get_collection(collection_name)
    total = collection.count()
    patched = 0
    skipped = 0
    processed = 0
    cache: Dict[Tuple[str, str], List[str]] = {}

    offset = 0
    while offset < total:
        batch = collection.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["metadatas"],
        )
        ids = batch.get("ids", []) or []
        metadatas = batch.get("metadatas", []) or []
        if not ids:
            break

        for idx, chunk_id in enumerate(ids):
            processed += 1
            existing_metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
            if not isinstance(existing_metadata, dict):
                existing_metadata = {}

            doc_number = str(existing_metadata.get("document_number", "") or "").strip()
            if not doc_number:
                skipped += 1
                continue

            source = str(existing_metadata.get("source", "") or "").strip()
            hts_codes = _lookup_doc_hts_codes(doc_number, source, sf_conn, cache)
            if not hts_codes:
                skipped += 1
                continue

            chapter_str, hts_code_str, chapter_counter = _extract_chapters(hts_codes)
            if not chapter_str and not hts_code_str:
                skipped += 1
                continue

            for chapter, cnt in chapter_counter.items():
                chapter_totals[chapter] = chapter_totals.get(chapter, 0) + cnt

            collection.update(
                ids=[chunk_id],
                metadatas=[{**existing_metadata, "hts_chapter": chapter_str, "hts_code": hts_code_str}],
            )
            patched += 1

            if processed % BATCH_SIZE == 0:
                print(
                    f"[{collection_name}] patched={patched} total={processed} skipped={skipped}"
                )

        offset += BATCH_SIZE

    return processed, patched, skipped


def main() -> None:
    chroma_client = _get_chroma_client()
    sf_conn = get_snowflake_conn()
    chapter_totals: Dict[str, int] = {}

    total_processed = 0
    total_patched = 0
    total_skipped = 0

    try:
        for name in COLLECTIONS:
            processed, patched, skipped = _patch_collection(
                collection_name=name,
                chroma_client=chroma_client,
                sf_conn=sf_conn,
                chapter_totals=chapter_totals,
            )
            total_processed += processed
            total_patched += patched
            total_skipped += skipped
            print(
                f"[{name}] done: processed={processed} updated={patched} skipped={skipped}"
            )
    finally:
        sf_conn.close()

    print(
        f"Patch complete: {total_processed} chunks processed, "
        f"{total_patched} updated, {total_skipped} skipped (no document_number)"
    )
    print(f"Chapters found: {dict(sorted(chapter_totals.items(), key=lambda x: x[0]))}")


if __name__ == "__main__":
    main()
