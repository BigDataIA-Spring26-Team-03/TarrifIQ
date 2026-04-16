USE DATABASE TARIFFIQ;
USE SCHEMA RAW;

CREATE TABLE IF NOT EXISTS ITC_DOCUMENTS (
    document_number     VARCHAR(20)     NOT NULL,
    title               TEXT,
    publication_date    DATE,
    document_type       VARCHAR(50),
    agency_names        VARIANT,
    abstract            TEXT,
    full_text           TEXT,
    html_url            VARCHAR(500),
    body_html_url       VARCHAR(500),
    char_count          INTEGER,
    chunk_count         INTEGER,
    raw_json            VARIANT,
    s3_key              VARCHAR(300),
    content_hash        VARCHAR(64),
    processing_status   VARCHAR(50)     DEFAULT 'pending',
    processing_error    TEXT,
    word_count          INTEGER,
    ingested_at         TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (document_number)
);

CREATE TABLE IF NOT EXISTS NOTICE_HTS_CODES_ITC (
    document_number     VARCHAR(20)     NOT NULL,
    hts_code            VARCHAR(32)     NOT NULL,
    hts_chapter         VARCHAR(2),
    context_snippet     TEXT,
    match_status        VARCHAR(50)     DEFAULT 'UNVERIFIED',
    PRIMARY KEY (document_number, hts_code)
);

CREATE TABLE IF NOT EXISTS ITC_CHUNKS (
    chunk_id            VARCHAR(100)    NOT NULL,
    document_number     VARCHAR(20)     NOT NULL,
    chunk_index         INT             NOT NULL,
    chunk_text          VARCHAR(16777216) NOT NULL,
    section             VARCHAR(100),
    hts_code            VARCHAR(20),
    hts_chapter         VARCHAR(20),
    word_count          INT,
    ingested_at         TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (chunk_id),
    FOREIGN KEY (document_number) REFERENCES ITC_DOCUMENTS(document_number)
);

ALTER TABLE ITC_CHUNKS CLUSTER BY (document_number, section, hts_code);

CREATE TABLE IF NOT EXISTS ITC_HTS_CODES LIKE HTS_CODES;
