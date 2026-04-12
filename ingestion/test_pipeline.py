
from ingestion.connection import get_snowflake_conn
from ingestion.html_parser import parse_fr_document
from ingestion.chunker import SemanticFRChunker
from ingestion.embedder import Embedder
import json

conn = get_snowflake_conn()
cur = conn.cursor()

cur.execute("""
    SELECT document_number, document_type, title, agency_names, publication_date, full_text
    FROM FEDERAL_REGISTER_NOTICES
    LIMIT 20
""")

rows = cur.fetchall()
print(f"Fetched {len(rows)} documents from Snowflake\n")

embedder = Embedder()
chunker = SemanticFRChunker(embedder)

for row in rows:
    document_number, document_type, title, agency_names_json, publication_date, full_text = row
    
    try:
        agency_names = json.loads(agency_names_json) if isinstance(agency_names_json, str) else agency_names_json or []
        
        doc = parse_fr_document(
            document_number=document_number,
            document_type=document_type,
            title=title,
            agency_names=agency_names,
            publication_date=publication_date,
            full_text=full_text
        )
        
        chunks = chunker.chunk_document(doc)
        
        print(f"{document_number}: {len(chunks)} chunks created")
        
    except Exception as e:
        print(f"{document_number}: ERROR - {e}")

cur.close()
conn.close()
