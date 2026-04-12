-- Remove full_text, abstract, and char_count from FEDERAL_REGISTER_NOTICES
-- full_text and abstract are now stored as chunks in the CHUNKS table
-- char_count was based on full_text length, now measured per-chunk in CHUNKS table
ALTER TABLE FEDERAL_REGISTER_NOTICES DROP COLUMN IF EXISTS abstract;
ALTER TABLE FEDERAL_REGISTER_NOTICES DROP COLUMN IF EXISTS full_text;
ALTER TABLE FEDERAL_REGISTER_NOTICES DROP COLUMN IF EXISTS char_count;
