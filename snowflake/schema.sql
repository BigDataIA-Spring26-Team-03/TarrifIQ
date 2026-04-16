CREATE DATABASE IF NOT EXISTS TARIFFIQ;
CREATE SCHEMA IF NOT EXISTS TARIFFIQ.RAW;
USE TARIFFIQ.RAW;

CREATE TABLE IF NOT EXISTS HTS_CODES (
    hts_id          VARCHAR(32)     NOT NULL,
    hts_code        VARCHAR(32)     NOT NULL,
    stat_suffix     VARCHAR(2),
    chapter         VARCHAR(2),
    level           VARCHAR(20),
    indent_level    INTEGER,
    description     TEXT            NOT NULL,
    general_rate    VARCHAR(100),
    special_rate    VARCHAR(500),
    other_rate      VARCHAR(100),
    units           VARCHAR(50),
    footnotes       VARIANT,
    is_chapter99    BOOLEAN         DEFAULT FALSE,
    is_header_row   BOOLEAN         DEFAULT FALSE,
    raw_json        VARIANT,
    loaded_at       TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (hts_code)
);

CREATE TABLE IF NOT EXISTS FEDERAL_REGISTER_NOTICES (
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
    ingested_at         TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (document_number)
);

CREATE TABLE IF NOT EXISTS NOTICE_HTS_CODES (
    document_number     VARCHAR(20)     NOT NULL,
    hts_code            VARCHAR(32)     NOT NULL,
    hts_chapter         VARCHAR(2),
    context_snippet     TEXT,
    match_status        VARCHAR(20)     DEFAULT 'UNVERIFIED',
    PRIMARY KEY (document_number, hts_code)
);

CREATE TABLE IF NOT EXISTS PRODUCT_ALIASES (
    alias               VARCHAR(200)    NOT NULL,
    hts_code            VARCHAR(32)     NOT NULL,
    confidence          DECIMAL(4,2),
    created_at          TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (alias)
);

CREATE TABLE IF NOT EXISTS HITL_RECORDS (
    hitl_id             VARCHAR(50)     NOT NULL,
    query_text          TEXT,
    trigger_reason      VARCHAR(100),
    classifier_hts      VARCHAR(32),
    classifier_conf     DECIMAL(4,2),
    human_decision      VARCHAR(200),
    adjudicated_at      TIMESTAMP_NTZ,
    created_at          TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (hitl_id)
);

CREATE TABLE IF NOT EXISTS CBP_FEDERAL_REGISTER_NOTICES (
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
    s3_key              VARCHAR(300),
    processing_status   VARCHAR(20),
    processing_error    TEXT,
    content_hash        VARCHAR(64),
    word_count          INTEGER,
    raw_json            VARIANT,
    ingested_at         TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (document_number)
);

CREATE TABLE IF NOT EXISTS CBP_NOTICE_HTS_CODES (
    document_number     VARCHAR(20)     NOT NULL,
    hts_code            VARCHAR(32)     NOT NULL,
    hts_chapter         VARCHAR(2),
    context_snippet     TEXT,
    match_status        VARCHAR(20)     DEFAULT 'UNVERIFIED',
    general_rate        VARCHAR(100),
    PRIMARY KEY (document_number, hts_code)
);

CREATE TABLE IF NOT EXISTS CBP_CHUNKS (
    chunk_id            VARCHAR(100)        NOT NULL,
    document_number     VARCHAR(20)         NOT NULL,
    chunk_index         INT                 NOT NULL,
    chunk_text          VARCHAR(16777216)   NOT NULL,
    section             VARCHAR(100),
    hts_code            VARCHAR(20),
    hts_chapter                   VARCHAR(20),
    general_rate        VARCHAR(100),
    word_count          INT,
    PRIMARY KEY (chunk_id)
);