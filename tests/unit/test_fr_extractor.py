"""
Unit tests for HTS entity extraction and cross-source validation.

Tests cover:
- extract_hts_entities: regex pattern matching for HTS codes, chapters, headings, ranges
- validate_hts_codes: Snowflake lookup with graceful mocking for all DB interactions
"""

import pytest
from unittest.mock import MagicMock

from ingestion.hts_extractor import extract_hts_entities
from ingestion.cross_source_validator import validate_hts_codes


# ===== Tests for extract_hts_entities (pure regex, no mocking) =====


class TestExtractHtsEntities:
    """Test HTS entity extraction from text."""

    def test_6digit_hts_code(self):
        """Test extraction of 6-digit HTS code."""
        result = extract_hts_entities("tariff on 8471.30 applies")
        assert len(result) == 1
        assert result[0]["entity_text"] == "8471.30"
        assert result[0]["label"] == "HTS_CODE"

    def test_10digit_hts_code(self):
        """Test extraction of 10-digit HTS code (most specific match)."""
        result = extract_hts_entities("rate 8471.30.0100 is duty-free")
        assert len(result) == 1
        assert result[0]["entity_text"] == "8471.30.0100"
        assert result[0]["label"] == "HTS_CODE"

    def test_8digit_hts_code(self):
        """Test extraction of 8-digit HTS code."""
        result = extract_hts_entities("subheading 8471.30.01 covers")
        assert len(result) == 1
        assert result[0]["entity_text"] == "8471.30.01"
        assert result[0]["label"] == "HTS_CODE"

    def test_chapter_label(self):
        """Test extraction of chapter reference."""
        result = extract_hts_entities("See chapter 84 for details")
        assert len(result) == 1
        assert result[0]["entity_text"] == "chapter 84"
        assert result[0]["label"] == "HTS_CHAPTER"

    def test_chapter_label_case_insensitive(self):
        """Test that chapter pattern is case-insensitive."""
        result = extract_hts_entities("See CHAPTER 84 for details")
        assert len(result) == 1
        assert result[0]["entity_text"] == "CHAPTER 84"
        assert result[0]["label"] == "HTS_CHAPTER"

    def test_heading_label(self):
        """Test extraction of heading reference."""
        result = extract_hts_entities("under heading 8471")
        assert len(result) == 1
        assert result[0]["entity_text"] == "heading 8471"
        assert result[0]["label"] == "HTS_HEADING"

    def test_range_pattern(self):
        """Test extraction of HTS code range (e.g., 8471.30 through 8471.49)."""
        result = extract_hts_entities("applies to 8471.30 through 8471.49")
        assert len(result) == 1
        assert result[0]["entity_text"] == "8471.30 through 8471.49"
        assert result[0]["label"] == "HTS_RANGE"

    def test_no_double_match(self):
        """Test that higher-priority 10-digit pattern prevents 6-digit match at same position."""
        result = extract_hts_entities("8471.30.0100")
        # Should only have 1 match: the 10-digit code, NOT also a 6-digit code
        assert len(result) == 1
        assert result[0]["entity_text"] == "8471.30.0100"

    def test_empty_text(self):
        """Test extraction from empty string."""
        result = extract_hts_entities("")
        assert result == []

    def test_sorted_by_start_char(self):
        """Test that results are sorted by start_char."""
        text = "See 8471.30 and chapter 84 and heading 8471"
        result = extract_hts_entities(text)
        # Multiple matches, should be sorted by position
        assert len(result) == 3
        start_chars = [e["start_char"] for e in result]
        assert start_chars == sorted(start_chars)

    def test_multiple_codes_no_overlap(self):
        """Test extraction of multiple non-overlapping codes."""
        text = "Apply 8471.30.0100 duty for 8507.60 products"
        result = extract_hts_entities(text)
        assert len(result) == 2
        assert result[0]["entity_text"] == "8471.30.0100"
        assert result[1]["entity_text"] == "8507.60"


# ===== Tests for validate_hts_codes (with mocking) =====


class TestValidateHtsCodes:
    """Test HTS code validation against Snowflake HTS_CODES table."""

    def test_verified_code(self):
        """Test that a code found in HTS_CODES returns VERIFIED."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchall.return_value = [("8507.60",)]

        result = validate_hts_codes(["8507.60"], mock_conn)

        assert len(result) == 1
        assert result[0]["hts_code"] == "8507.60"
        assert result[0]["match_status"] == "VERIFIED"

    def test_unmatched_code(self):
        """Test that a code NOT in HTS_CODES returns UNMATCHED."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchall.return_value = []  # No matches

        result = validate_hts_codes(["FAKE-9999"], mock_conn)

        assert len(result) == 1
        assert result[0]["hts_code"] == "FAKE-9999"
        assert result[0]["match_status"] == "UNMATCHED"

    def test_mixed_verified_and_unmatched(self):
        """Test a mix of verified and unmatched codes."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchall.return_value = [("8507.60",)]  # Only one found

        result = validate_hts_codes(["8507.60", "FAKE-9999"], mock_conn)

        assert len(result) == 2
        verified = [r for r in result if r["match_status"] == "VERIFIED"]
        unmatched = [r for r in result if r["match_status"] == "UNMATCHED"]
        assert len(verified) == 1
        assert len(unmatched) == 1
        assert verified[0]["hts_code"] == "8507.60"
        assert unmatched[0]["hts_code"] == "FAKE-9999"

    def test_empty_input(self):
        """Test that empty input list returns empty result without DB call."""
        mock_conn = MagicMock()
        result = validate_hts_codes([], mock_conn)

        assert result == []
        # Verify no cursor was created (no DB call)
        mock_conn.cursor.assert_not_called()

    def test_deduplication(self):
        """Test that duplicate codes are deduplicated before DB query."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchall.return_value = [("8507.60",)]

        result = validate_hts_codes(["8507.60", "8507.60", "8507.60"], mock_conn)

        # Should only have 1 result (deduplicated)
        assert len(result) == 1
        assert result[0]["hts_code"] == "8507.60"

        # Verify the SQL query was called with only 1 parameter
        call_args = mock_cur.execute.call_args
        # The parameter list should have only 1 item
        assert len(call_args[0][1]) == 1

    def test_db_failure_graceful(self):
        """Test that DB failure marks all codes as UNMATCHED without crashing."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_cur.execute.side_effect = Exception("Connection lost")

        result = validate_hts_codes(["8507.60", "8471.30"], mock_conn)

        # Should still return results, all UNMATCHED
        assert len(result) == 2
        for r in result:
            assert r["match_status"] == "UNMATCHED"

    def test_cursor_cleanup(self):
        """Test that cursor is properly closed even on exception."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_cur.execute.side_effect = Exception("DB error")

        # Should not raise, gracefully handle error
        result = validate_hts_codes(["8507.60"], mock_conn)

        # Verify cursor.close() was called despite exception
        mock_cur.close.assert_called_once()

    def test_maintains_order(self):
        """Test that result order matches input order (before deduplication)."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchall.return_value = [("8507.60",), ("8471.30",)]

        result = validate_hts_codes(["8507.60", "FAKE-1", "8471.30", "FAKE-2"], mock_conn)

        # Check order is preserved
        assert result[0]["hts_code"] == "8507.60"
        assert result[1]["hts_code"] == "FAKE-1"
        assert result[2]["hts_code"] == "8471.30"
        assert result[3]["hts_code"] == "FAKE-2"
