-- Create CHUNKS table to store parsed and chunked Federal Register document segments
CREATE TABLE IF NOT EXISTS CHUNKS (
    chunk_id VARCHAR(100) NOT NULL,  -- Format: document_number_chunk_index
    document_number VARCHAR(20) NOT NULL,  -- Match FEDERAL_REGISTER_NOTICES type
    chunk_index INT NOT NULL,
    chunk_text VARCHAR(16777216) NOT NULL,  -- The actual chunk content
    section VARCHAR(100),  -- e.g., "SUMMARY", "SUPPLEMENTARY INFORMATION"
    hts_code VARCHAR(20),  -- Extracted HTS code if present
    hts_chapter VARCHAR(20),  -- Extracted HTS chapter if present
    word_count INT,  -- Number of words in chunk

    -- Metadata
    ingested_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),

    -- Constraints
    PRIMARY KEY (chunk_id),
    FOREIGN KEY (document_number) REFERENCES FEDERAL_REGISTER_NOTICES(document_number)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_number ON CHUNKS(document_number);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON CHUNKS(section);
CREATE INDEX IF NOT EXISTS idx_chunks_hts_code ON CHUNKS(hts_code);
