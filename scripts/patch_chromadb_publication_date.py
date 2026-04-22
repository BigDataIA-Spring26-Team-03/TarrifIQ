#!/usr/bin/env python3
"""
One-time patch: add publication_date metadata to all policy_notices chunks in ChromaDB.
Fetches dates from all 5 Snowflake notice tables and updates ChromaDB metadata in batches.
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv(override=True)

import chromadb
import snowflake.connector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _sf():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
    )


def build_date_map():
    """Fetch publication_date for all documents from all 5 notice tables."""
    conn = _sf()
    cur = conn.cursor()
    date_map = {}
    tables = [
        "FEDERAL_REGISTER_NOTICES",
        "CBP_FEDERAL_REGISTER_NOTICES",
        "EOP_DOCUMENTS",
        "ITC_DOCUMENTS",
        "ITA_FEDERAL_REGISTER_NOTICES",
    ]
    for table in tables:
        try:
            cur.execute(
                f"""
                SELECT document_number, publication_date::VARCHAR
                FROM TARIFFIQ.RAW.{table}
                WHERE publication_date IS NOT NULL
                """
            )
            for doc_num, pub_date in cur.fetchall():
                if doc_num and pub_date:
                    date_map[str(doc_num)] = str(pub_date)
            logger.info("date_map loaded from %s: %d total", table, len(date_map))
        except Exception as e:
            logger.warning("date_map_error table=%s error=%s", table, e)
    cur.close()
    conn.close()
    logger.info("date_map total: %d documents", len(date_map))
    return date_map


def patch_chromadb():
    host = os.environ.get("CHROMADB_HOST", "127.0.0.1")
    port = int(os.environ.get("CHROMADB_PORT", "9001"))
    client = chromadb.HttpClient(host=host, port=port)
    col = client.get_collection("policy_notices")

    date_map = build_date_map()
    total_chunks = col.count()
    logger.info("Total chunks in ChromaDB: %d", total_chunks)

    batch_size = 500
    offset = 0
    updated = 0
    skipped = 0

    while offset < total_chunks:
        results = col.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"],
        )
        ids = results["ids"]
        if not ids:
            break

        ids_to_update = []
        metadatas_to_update = []

        for i, chunk_id in enumerate(ids):
            meta = results["metadatas"][i]
            doc_num = meta.get("document_number", "")
            pub_date = date_map.get(doc_num)
            if pub_date and meta.get("publication_date") != pub_date:
                new_meta = {**meta, "publication_date": pub_date}
                ids_to_update.append(chunk_id)
                metadatas_to_update.append(new_meta)
            else:
                skipped += 1

        if ids_to_update:
            try:
                col.update(ids=ids_to_update, metadatas=metadatas_to_update)
                updated += len(ids_to_update)
            except Exception as e:
                logger.error("update_error offset=%d error=%s", offset, e)

        offset += batch_size
        logger.info(
            "Progress: %d/%d chunks processed, %d updated",
            offset,
            total_chunks,
            updated,
        )

    logger.info("Patch complete. Updated=%d Skipped=%d", updated, skipped)


if __name__ == "__main__":
    patch_chromadb()
