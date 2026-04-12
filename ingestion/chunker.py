import logging
from datetime import datetime, timezone
from typing import List, Optional

import spacy
from sklearn.metrics.pairwise import cosine_similarity

from ingestion.embedder import Embedder
from ingestion.html_parser import ParsedFRDocument

logger = logging.getLogger(__name__)


class SemanticFRChunker:
    """
    Semantic chunking for Federal Register documents using cosine similarity breakpoints.
    Chunks are created by finding semantic shifts between adjacent sentences.
    """

    def __init__(
        self,
        embedder: Embedder,
        breakpoint_threshold: float = 0.75,
        max_chunk_words: int = 400,
        min_chunk_words: int = 30,
    ):
        """
        Args:
            embedder: Embedder instance (uses all-MiniLM-L6-v2)
            breakpoint_threshold: Cosine similarity below this creates new chunk
            max_chunk_words: Hard cap on chunk size to prevent runaway merges
            min_chunk_words: Discard chunks smaller than this
        """
        self.embedder = embedder
        self.breakpoint_threshold = breakpoint_threshold
        self.max_chunk_words = max_chunk_words
        self.min_chunk_words = min_chunk_words

        # Load spaCy model for sentence segmentation
        self.nlp = spacy.load("en_core_web_sm")

    def chunk_document(
        self,
        doc: ParsedFRDocument,
        hts_annotations: Optional[dict] = None,
    ) -> List[dict]:
        """
        Chunk a ParsedFRDocument into semantic chunks using cosine similarity.

        Args:
            doc: ParsedFRDocument with sections already extracted
            hts_annotations: Optional dict of {chunk_text: {"hts_code": "...", "hts_chapter": "..."}}
                            from entity extraction step

        Returns:
            List of chunk dicts:
            {
                "chunk_text": str,
                "document_number": str,
                "section": str,
                "chunk_index": int,
                "hts_code": Optional[str],
                "hts_chapter": Optional[str],
                "ingested_at": str,
            }
        """
        if not doc.sections:
            return []

        hts_annotations = hts_annotations or {}
        chunks = []

        for section_name, section_text in doc.sections.items():
            if not section_text or not section_text.strip():
                continue

            section_chunks = self._semantic_split(
                section_text,
                doc.document_number,
                section_name,
                hts_annotations,
            )
            chunks.extend(section_chunks)

        logger.info(
            "chunk_document done doc=%s sections=%d total_chunks=%d",
            doc.document_number,
            len(doc.sections),
            len(chunks),
        )
        return chunks

    def _semantic_split(
        self,
        text: str,
        document_number: str,
        section: str,
        hts_annotations: dict,
    ) -> List[dict]:
        """
        Split a section into semantic chunks using sentence-level embeddings.

        Args:
            text: Section text
            document_number: Document number
            section: Section name
            hts_annotations: HTS code/chapter annotations from entity extraction

        Returns:
            List of chunk dicts for this section
        """
        # Segment into sentences
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        if not sentences:
            return []

        # Single sentence → single chunk
        if len(sentences) == 1:
            return [
                self._make_chunk(
                    sentences[0],
                    document_number,
                    section,
                    chunk_index=0,
                    hts_annotations=hts_annotations,
                )
            ]

        # Embed all sentences in batch
        embeddings = self.embedder.embed_batch(sentences)

        # Find semantic breakpoints using cosine similarity
        groups = []
        current_group = [sentences[0]]
        current_words = len(sentences[0].split())

        for i in range(1, len(sentences)):
            sim = cosine_similarity(
                [embeddings[i - 1]], [embeddings[i]]
            )[0][0]
            next_words = len(sentences[i].split())

            # Break if: semantic shift OR chunk would exceed max size
            if (
                sim < self.breakpoint_threshold
                or (current_words + next_words) > self.max_chunk_words
            ):
                groups.append(current_group)
                current_group = [sentences[i]]
                current_words = next_words
            else:
                current_group.append(sentences[i])
                current_words += next_words

        groups.append(current_group)

        # Convert groups to chunks, discard noise
        results = []
        for idx, group in enumerate(groups):
            text_joined = " ".join(group)
            if len(text_joined.split()) >= self.min_chunk_words:
                results.append(
                    self._make_chunk(
                        text_joined,
                        document_number,
                        section,
                        chunk_index=idx,
                        hts_annotations=hts_annotations,
                    )
                )

        return results

    def _make_chunk(
        self,
        chunk_text: str,
        document_number: str,
        section: str,
        chunk_index: int,
        hts_annotations: dict,
    ) -> dict:
        """Create a chunk dict with optional HTS annotation."""
        hts_code = None
        hts_chapter = None

        # Check if this chunk text was annotated with HTS code/chapter
        if chunk_text in hts_annotations:
            hts_code = hts_annotations[chunk_text].get("hts_code")
            hts_chapter = hts_annotations[chunk_text].get("hts_chapter")

        return {
            "chunk_text": chunk_text,
            "document_number": document_number,
            "section": section,
            "chunk_index": chunk_index,
            "hts_code": hts_code,
            "hts_chapter": hts_chapter,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }


# Legacy function for backward compatibility
def chunk_document(text: str, entities: List[dict], doc_meta: dict) -> List[dict]:
    """
    Legacy HTS-anchor-only chunking. Kept for backward compatibility.
    New code should use SemanticFRChunker directly.

    Creates ~200-word context windows anchored around HTS entities.
    Returns empty list if no entities found.
    """
    if not entities or not text:
        return []

    words, word_starts = _build_word_index(text)
    if not words:
        return []

    # Build windows — one per entity
    WINDOW_HALF = 100
    raw_windows = []
    for entity in entities:
        center_start = _char_to_word_idx(entity["start_char"], word_starts)
        center_end = _char_to_word_idx(entity["end_char"], word_starts)

        win_start = max(0, center_start - WINDOW_HALF)
        win_end = min(len(words), center_end + WINDOW_HALF + 1)

        raw_windows.append(
            {
                "win_start": win_start,
                "win_end": win_end,
                "hts_code": (
                    entity["entity_text"]
                    if entity["label"] == "HTS_CODE"
                    else ""
                ),
                "hts_chapter": (
                    entity["entity_text"].split()[-1]  # e.g. "chapter 84" → "84"
                    if entity["label"] == "HTS_CHAPTER"
                    else entity["entity_text"][:2]  # first 2 digits of HTS_CODE
                    if entity["label"] == "HTS_CODE"
                    else ""
                ),
            }
        )

    # Sort by window start, then merge overlapping windows
    raw_windows.sort(key=lambda w: w["win_start"])
    merged = []
    for w in raw_windows:
        if merged and w["win_start"] <= merged[-1]["win_end"]:
            prev = merged[-1]
            prev["win_end"] = max(prev["win_end"], w["win_end"])
            # Keep first non-empty HTS code / chapter found
            if not prev["hts_code"] and w["hts_code"]:
                prev["hts_code"] = w["hts_code"]
            if not prev["hts_chapter"] and w["hts_chapter"]:
                prev["hts_chapter"] = w["hts_chapter"]
        else:
            merged.append(dict(w))

    now = datetime.now(timezone.utc)
    chunks = []
    for m in merged:
        chunk_text = " ".join(words[m["win_start"] : m["win_end"]])
        chunks.append(
            {
                "chunk_text": chunk_text,
                "hts_code": m["hts_code"],
                "hts_chapter": m["hts_chapter"],
                "document_number": doc_meta.get("document_number", ""),
                "ingested_at": now,
            }
        )

    logger.info(
        "chunk_document done doc=%s entities=%d chunks=%d",
        doc_meta.get("document_number", "?"),
        len(entities),
        len(chunks),
    )
    return chunks


def _build_word_index(text: str):
    """
    Returns (words, word_start_chars) where word_start_chars[i] is the
    character offset of words[i] in text.
    """
    words = []
    starts = []
    i = 0
    n = len(text)
    while i < n:
        # Skip whitespace
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        # Collect word
        j = i
        while j < n and not text[j].isspace():
            j += 1
        words.append(text[i:j])
        starts.append(i)
        i = j
    return words, starts


def _char_to_word_idx(char_pos: int, word_starts: List[int]) -> int:
    """Binary search to find which word index contains char_pos."""
    lo, hi = 0, len(word_starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if word_starts[mid] <= char_pos:
            lo = mid
        else:
            hi = mid - 1
    return lo
