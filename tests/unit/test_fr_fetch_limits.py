"""Unit tests for Federal Register incremental fetch limits."""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

# Avoid importing heavyweight NLP deps in this focused fetch-limit test.
if "spacy" not in sys.modules:
    fake_spacy = types.SimpleNamespace(load=lambda *_args, **_kwargs: None)
    sys.modules["spacy"] = fake_spacy

from ingestion.federal_register_client import fetch_and_load_incrementally


def _doc(i: int) -> dict:
    return {
        "document_number": f"2026-{i:04d}",
        "title": f"Doc {i}",
        "publication_date": "2026-01-15",
        "html_url": f"https://example.com/{i}",
        "full_text_xml_url": f"https://example.com/{i}.xml",
        "type": "Notice",
        "agencies": [{"name": "Test Agency"}],
    }


@patch("ingestion.federal_register_client.load_to_snowflake")
@patch("ingestion.federal_register_client._fetch_full_text")
@patch("ingestion.federal_register_client._iter_pages")
def test_fetch_incremental_no_cap_loads_all_docs(mock_iter_pages, mock_fetch_full_text, mock_load):
    docs = [_doc(i) for i in range(1, 13)]
    mock_iter_pages.return_value = [docs]  # one page, 12 docs
    mock_fetch_full_text.return_value = ("full text", "raw/federal-register/2026/01/test.xml")

    def _load_side_effect(rows):
        return len(rows)

    mock_load.side_effect = _load_side_effect

    loaded = fetch_and_load_incrementally(
        test_mode=True,
        cutoff_year=2016,
        batch_size=50,
        max_documents=None,
    )

    assert loaded == 12
    assert mock_fetch_full_text.call_count == 12
    assert mock_load.call_count == 1


@patch("ingestion.federal_register_client.load_to_snowflake")
@patch("ingestion.federal_register_client._fetch_full_text")
@patch("ingestion.federal_register_client._iter_pages")
def test_fetch_incremental_cap_10_stops_at_10(mock_iter_pages, mock_fetch_full_text, mock_load):
    docs = [_doc(i) for i in range(1, 25)]
    mock_iter_pages.return_value = [docs]  # one page, 24 docs available
    mock_fetch_full_text.return_value = ("full text", "raw/federal-register/2026/01/test.xml")

    def _load_side_effect(rows):
        return len(rows)

    mock_load.side_effect = _load_side_effect

    loaded = fetch_and_load_incrementally(
        test_mode=True,
        cutoff_year=2016,
        batch_size=50,
        max_documents=10,
    )

    assert loaded == 10
    assert mock_fetch_full_text.call_count == 10
    assert mock_load.call_count == 1
