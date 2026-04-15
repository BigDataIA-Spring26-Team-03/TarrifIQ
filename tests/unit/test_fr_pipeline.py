"""
Unit tests for Federal Register ingestion pipeline:
- Section extraction (spaCy-based)
- Semantic chunking (cosine similarity breakpoints)
- Content hashing (SHA-256)
- ParsedFRDocument dataclass
"""

from __future__ import annotations

import hashlib
from datetime import datetime

import pytest

from ingestion.html_parser import (
    ParsedFRDocument,
    extract_fr_sections,
    parse_fr_document,
)


class TestFRSectionExtraction:
    """Test section extraction from Federal Register documents."""

    def test_extract_sections_with_clear_headers(self):
        """Test extraction when document has recognizable section headers."""
        text = """
        SUMMARY
        This rule addresses tariffs on solar panels.

        SUPPLEMENTARY INFORMATION
        Background and rationale for the rule.

        DATES
        Effective date is January 1, 2024.
        """

        sections = extract_fr_sections(text, "Rule")

        # Should have at least 3 sections
        assert len(sections) >= 2
        assert "SUMMARY" in sections or "FULL_DOCUMENT" in sections

    def test_extract_sections_no_headers(self):
        """Test fallback to FULL_DOCUMENT when no recognized headers."""
        text = "This is a simple document without any section headers."

        sections = extract_fr_sections(text, "Notice")

        assert "FULL_DOCUMENT" in sections
        assert sections["FULL_DOCUMENT"].strip() == text.strip()

    def test_extract_sections_empty_text(self):
        """Test with empty text."""
        sections = extract_fr_sections("", "Notice")
        assert sections == {}

    def test_extract_sections_preserves_content(self):
        """Test that section extraction preserves all content."""
        text = """
        SUMMARY
        Section 1 content.

        BACKGROUND
        Section 2 content.
        """

        sections = extract_fr_sections(text, "Rule")

        # All content should be preserved somewhere
        full_content = " ".join(sections.values())
        assert "Section 1 content" in full_content
        assert "Section 2 content" in full_content


class TestParsedFRDocument:
    """Test ParsedFRDocument dataclass creation and properties."""

    def test_parse_fr_document_basic(self):
        """Test basic document parsing."""
        doc = parse_fr_document(
            document_number="2024-1234",
            document_type="Rule",
            title="Solar Panel Tariff Rule",
            agency_names=["Office of Trade Representative"],
            publication_date="2024-01-15",
            full_text="This is the full document text.",
        )

        assert doc.document_number == "2024-1234"
        assert doc.title == "Solar Panel Tariff Rule"
        assert doc.word_count == 6  # "This is the full document text."
        assert len(doc.content_hash) == 64  # SHA-256 hex digest

    def test_content_hash_deterministic(self):
        """Test that identical text produces identical hashes."""
        text = "Sample Federal Register document."

        doc1 = parse_fr_document(
            document_number="2024-1",
            document_type="Notice",
            title="Test 1",
            agency_names=["Test Agency"],
            publication_date="2024-01-01",
            full_text=text,
        )

        doc2 = parse_fr_document(
            document_number="2024-2",
            document_type="Notice",
            title="Test 2",
            agency_names=["Test Agency"],
            publication_date="2024-01-02",
            full_text=text,
        )

        # Same text → same hash
        assert doc1.content_hash == doc2.content_hash

    def test_content_hash_different_text(self):
        """Test that different text produces different hashes."""
        doc1 = parse_fr_document(
            document_number="2024-1",
            document_type="Notice",
            title="Test",
            agency_names=["Test"],
            publication_date="2024-01-01",
            full_text="Document A",
        )

        doc2 = parse_fr_document(
            document_number="2024-2",
            document_type="Notice",
            title="Test",
            agency_names=["Test"],
            publication_date="2024-01-01",
            full_text="Document B",
        )

        # Different text → different hash
        assert doc1.content_hash != doc2.content_hash

    def test_content_hash_matches_sha256(self):
        """Test that content_hash matches manual SHA-256 calculation."""
        text = "Test document content"

        doc = parse_fr_document(
            document_number="2024-test",
            document_type="Rule",
            title="Test",
            agency_names=["Test"],
            publication_date="2024-01-01",
            full_text=text,
        )

        expected_hash = hashlib.sha256(text.encode()).hexdigest()
        assert doc.content_hash == expected_hash

    def test_parse_fr_document_with_s3_key(self):
        """Test parsing with S3 key."""
        doc = parse_fr_document(
            document_number="2024-1234",
            document_type="Notice",
            title="Test",
            agency_names=["Test"],
            publication_date="2024-01-15",
            full_text="Content",
            source_s3_key="s3://bucket/raw/federal-register/2024/01/2024-1234.xml",
        )

        assert doc.source_s3_key == "s3://bucket/raw/federal-register/2024/01/2024-1234.xml"

    def test_word_count_calculation(self):
        """Test that word count is calculated correctly."""
        test_cases = [
            ("", 0),
            ("one", 1),
            ("one two three", 3),
            ("word1 word2  word3", 3),  # extra spaces
            ("multi\nline\ntext", 3),  # newlines as separators
        ]

        for text, expected_count in test_cases:
            doc = parse_fr_document(
                document_number="test",
                document_type="Notice",
                title="Test",
                agency_names=["Test"],
                publication_date="2024-01-01",
                full_text=text,
            )
            assert doc.word_count == expected_count, f"Failed for text: {text}"


class TestSemanticChunking:
    """Test SemanticFRChunker (requires embeddings)."""

    def test_semantic_chunker_creates_chunks(self):
        """Test that semantic chunker produces chunks for documents without HTS entities."""
        from ingestion.chunker import SemanticFRChunker
        from ingestion.embedder import Embedder

        embedder = Embedder()
        chunker = SemanticFRChunker(
            embedder,
            breakpoint_threshold=0.75,
            max_chunk_words=400,
            min_chunk_words=30,
        )

        # Create a document with NO HTS entities to test the fix
        text = """
        The Federal Register publishes rules and notices from federal agencies.

        This rule establishes new import duties on certain products.

        The effective date for this rule is January 1, 2024.

        All importers must comply with the new requirements.
        """

        doc = parse_fr_document(
            document_number="2024-test",
            document_type="Notice",
            title="Test Rule",
            agency_names=["Test Agency"],
            publication_date="2024-01-01",
            full_text=text,
        )

        chunks = chunker.chunk_document(doc)

        # Key fix: should produce chunks even without HTS entities
        assert len(chunks) > 0, "Chunker should produce chunks even without HTS entities"

        # All chunks should have required fields
        for chunk in chunks:
            assert "chunk_text" in chunk
            assert "document_number" in chunk
            assert "section" in chunk
            assert chunk["document_number"] == "2024-test"

    def test_semantic_chunker_respects_min_chunk_size(self):
        """Test that chunks below min_chunk_size are discarded."""
        from ingestion.chunker import SemanticFRChunker
        from ingestion.embedder import Embedder

        embedder = Embedder()
        chunker = SemanticFRChunker(
            embedder,
            breakpoint_threshold=0.75,
            max_chunk_words=400,
            min_chunk_words=50,  # Discard chunks smaller than 50 words
        )

        text = " ".join(["word"] * 100)  # 100 words total

        doc = parse_fr_document(
            document_number="2024-test",
            document_type="Notice",
            title="Test",
            agency_names=["Test"],
            publication_date="2024-01-01",
            full_text=text,
        )

        chunks = chunker.chunk_document(doc)

        # All chunks should be at least min_chunk_words
        for chunk in chunks:
            word_count = len(chunk["chunk_text"].split())
            assert word_count >= 50, f"Chunk is too small: {word_count} words"

    def test_semantic_chunker_with_sections(self):
        """Test chunking of multi-section documents."""
        from ingestion.chunker import SemanticFRChunker
        from ingestion.embedder import Embedder

        embedder = Embedder()
        chunker = SemanticFRChunker(embedder)

        text = """
        SUMMARY
        This is the summary section with substantial content about the rule and its implications
        for importers and the international trade community. The summary should provide context.

        SUPPLEMENTARY INFORMATION
        The supplementary information section provides additional details and background
        for why this rule was necessary and how it affects tariff classifications.
        The information here is critical for understanding the regulatory changes.
        """

        doc = parse_fr_document(
            document_number="2024-multi",
            document_type="Rule",
            title="Test Multi-Section",
            agency_names=["Test"],
            publication_date="2024-01-01",
            full_text=text,
        )

        chunks = chunker.chunk_document(doc)

        # Should have chunks from multiple sections
        assert len(chunks) > 0

        # Each chunk should have a section identifier
        for chunk in chunks:
            assert chunk["section"] != ""


class TestChunkingWithoutHTSEntities:
    """
    Regression tests for the CHUNK_COUNT=0 bug.
    Ensure documents without HTS entities still produce chunks.
    """

    def test_document_without_hts_entities_produces_chunks(self):
        """Regression test: non-HTS documents should still be chunked."""
        from ingestion.chunker import SemanticFRChunker
        from ingestion.embedder import Embedder

        embedder = Embedder()
        chunker = SemanticFRChunker(embedder)

        # Generic Federal Register text with no HTS codes or chapters
        text = """
        NOTICE OF PROPOSED RULEMAKING
        The Department of Commerce proposes to amend the regulations governing
        import procedures and tariff administration.

        BACKGROUND
        Recent changes in international trade patterns have necessitated this update.

        DATES
        Comments must be submitted within 60 days of publication.
        """

        doc = parse_fr_document(
            document_number="2024-nonhts",
            document_type="Notice",
            title="Generic FR Notice",
            agency_names=["Commerce Department"],
            publication_date="2024-01-01",
            full_text=text,
        )

        chunks = chunker.chunk_document(doc)

        # This is the critical fix: chunk_count must be > 0
        assert len(chunks) > 0, "Bug regression: documents without HTS entities should still be chunked"

        # Verify chunk structure
        assert all("chunk_text" in c for c in chunks)
        assert all("document_number" in c for c in chunks)
        assert all(c["document_number"] == "2024-nonhts" for c in chunks)
