"""
Local test of the complete FR ingestion pipeline WITHOUT Airflow.
Mimics: fetch → parse → chunk → embed → index
"""

from ingestion.connection import get_snowflake_conn
from ingestion.html_parser import parse_fr_document, extract_fr_sections, ParsedFRDocument
from ingestion.chunker import SemanticFRChunker
from ingestion.embedder import Embedder
from ingestion.hts_extractor import extract_hts_entities
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def test_parse_and_chunk():
    """Test parsing and chunking on real Snowflake data."""
    conn = get_snowflake_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT document_number, document_type, title, agency_names, publication_date, full_text
        FROM FEDERAL_REGISTER_NOTICES
        LIMIT 20
    """)

    rows = cur.fetchall()
    logger.info(f"Fetched {len(rows)} documents from Snowflake")

    embedder = Embedder()
    chunker = SemanticFRChunker(embedder)

    total_chunks = 0
    failed = 0

    for row in rows:
        document_number, document_type, title, agency_names_json, publication_date, full_text = row

        try:
            # Parse
            agency_names = json.loads(agency_names_json) if isinstance(agency_names_json, str) else agency_names_json or []

            doc = parse_fr_document(
                document_number=document_number,
                document_type=document_type,
                title=title,
                agency_names=agency_names,
                publication_date=publication_date,
                full_text=full_text
            )

            # Extract HTS entities
            hts_entities = extract_hts_entities(full_text)
            hts_annotations = {}
            for entity in hts_entities:
                entity_text = full_text[entity["start_char"] : entity["end_char"]]
                hts_annotations[entity_text] = {
                    "hts_code": entity["entity_text"] if entity["label"] == "HTS_CODE" else None,
                    "hts_chapter": entity["entity_text"][:2] if entity["label"] == "HTS_CODE" else (
                        entity["entity_text"].split()[-1] if entity["label"] == "HTS_CHAPTER" else None
                    ),
                }

            # Chunk
            chunks = chunker.chunk_document(doc, hts_annotations)
            total_chunks += len(chunks)

            logger.info(f"  {document_number}: {len(chunks)} chunks, {doc.word_count} words, hash={doc.content_hash[:8]}...")

        except Exception as e:
            logger.error(f"  {document_number}: FAILED - {e}")
            failed += 1

    logger.info(f"\nParse + Chunk complete:")
    logger.info(f"  Documents processed: {len(rows) - failed}/{len(rows)}")
    logger.info(f"  Total chunks created: {total_chunks}")
    logger.info(f"  Failed: {failed}")

    cur.close()
    conn.close()

    return len(rows) - failed, total_chunks

if __name__ == "__main__":
    processed, chunks = test_parse_and_chunk()
    if chunks > 0:
        print(f"\nSUCCESS: {processed} docs -> {chunks} chunks created")
        print("Pipeline ready for Airflow DAG or ChromaDB embedding test")
    else:
        print("\nFAILED: No chunks created")
