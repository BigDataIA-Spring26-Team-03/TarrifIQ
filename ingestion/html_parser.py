import hashlib
import re
from dataclasses import dataclass
from typing import Optional

import spacy
from bs4 import BeautifulSoup


def strip_html(raw_html: str) -> str:
    """
    Strips HTML tags from a Federal Register document body.
    Preserves paragraph structure as double newlines.
    Returns clean plain text.
    """
    if not raw_html:
        return ""

    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove non-content tags entirely
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    # Insert double newline after block-level elements to preserve structure
    for tag in soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br"]):
        tag.append("\n\n")

    text = soup.get_text(separator=" ")

    # Normalize whitespace — collapse runs of spaces/tabs, preserve paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


@dataclass
class ParsedFRDocument:
    """Structured representation of a parsed Federal Register document."""
    document_number: str
    document_type: str  # "Notice", "Rule", "Proposed Rule"
    title: str
    agency_names: list[str]
    publication_date: str
    full_text: str  # cleaned plain text
    sections: dict[str, str]  # {"SUPPLEMENTARY INFORMATION": "...", "SUMMARY": "..."}
    content_hash: str  # SHA-256 of full_text
    word_count: int
    source_s3_key: Optional[str] = None


# spaCy patterns for Federal Register section headers
FR_SECTION_PATTERNS = [
    {"label": "FR_SECTION", "pattern": [{"ORTH": "SUPPLEMENTARY"}, {"ORTH": "INFORMATION"}]},
    {"label": "FR_SECTION", "pattern": [{"ORTH": "SUMMARY"}]},
    {"label": "FR_SECTION", "pattern": [{"ORTH": "BACKGROUND"}]},
    {"label": "FR_SECTION", "pattern": [{"ORTH": "DATES"}]},
    {"label": "FR_SECTION", "pattern": [{"ORTH": "EFFECTIVE"}, {"ORTH": "DATE"}]},
    {"label": "FR_SECTION", "pattern": [{"ORTH": "PREAMBLE"}]},
    {"label": "FR_SECTION", "pattern": [{"ORTH": "FINDINGS"}]},
    {"label": "FR_SECTION", "pattern": [{"ORTH": "DETERMINATION"}]},
    {"label": "FR_SECTION", "pattern": [{"ORTH": "DISCUSSION"}]},
    {"label": "FR_SECTION", "pattern": [{"ORTH": "CONCLUSION"}]},
    {"label": "FR_SECTION", "pattern": [{"ORTH": "APPENDIX"}]},
    {"label": "FR_SECTION", "pattern": [{"ORTH": "FOR"}, {"ORTH": "FURTHER"}, {"ORTH": "INFORMATION"}, {"ORTH": "CONTACT"}]},
]


def _get_spacy_nlp():
    """Load spaCy model with FR section entity ruler (lazy load)."""
    nlp = spacy.load("en_core_web_sm")
    if "fr_section_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(FR_SECTION_PATTERNS)
    return nlp


def extract_fr_sections(text: str, document_type: str) -> dict[str, str]:
    """
    Extract sections from Federal Register document using spaCy EntityRuler.

    Args:
        text: Full document text (clean plain text)
        document_type: "Notice", "Rule", "Proposed Rule"

    Returns:
        Dictionary of {section_name: section_text}
        Falls back to {"FULL_DOCUMENT": text} if no sections detected.
    """
    if not text or not text.strip():
        return {}

    nlp = _get_spacy_nlp()
    doc = nlp(text)

    # Find all FR_SECTION entities
    section_entities = [ent for ent in doc.ents if ent.label_ == "FR_SECTION"]

    if not section_entities:
        # No recognized sections — treat entire document as one section
        return {"FULL_DOCUMENT": text}

    sections = {}
    for i, entity in enumerate(section_entities):
        section_name = entity.text.upper()

        # Section text starts after the header
        start_char = entity.end_char

        # Section text ends at the next section header (or end of document)
        if i + 1 < len(section_entities):
            end_char = section_entities[i + 1].start_char
        else:
            end_char = len(text)

        section_text = text[start_char:end_char].strip()
        if section_text:
            sections[section_name] = section_text

    # If no valid sections were extracted, fall back to full document
    if not sections:
        return {"FULL_DOCUMENT": text}

    return sections


def parse_fr_document(
    document_number: str,
    document_type: str,
    title: str,
    agency_names: list[str],
    publication_date: str,
    full_text: str,
    source_s3_key: Optional[str] = None,
) -> ParsedFRDocument:
    """
    Parse a Federal Register document into structured format.

    Args:
        document_number: Federal Register document number (e.g., "2024-1234")
        document_type: Type of document ("Notice", "Rule", "Proposed Rule")
        title: Document title
        agency_names: List of agency names responsible for this document
        publication_date: Publication date as string (e.g., "2024-01-15")
        full_text: Full plain text of document
        source_s3_key: Optional S3 key where raw XML was stored

    Returns:
        ParsedFRDocument with sections extracted and content hashed
    """
    # Clean up text
    full_text = full_text.strip() if full_text else ""

    # Compute content hash (SHA-256 of full text for deduplication)
    content_hash = hashlib.sha256(full_text.encode()).hexdigest()

    # Extract sections
    sections = extract_fr_sections(full_text, document_type)

    # Count words
    word_count = len(full_text.split()) if full_text else 0

    return ParsedFRDocument(
        document_number=document_number,
        document_type=document_type,
        title=title,
        agency_names=agency_names,
        publication_date=publication_date,
        full_text=full_text,
        sections=sections,
        content_hash=content_hash,
        word_count=word_count,
        source_s3_key=source_s3_key,
    )
