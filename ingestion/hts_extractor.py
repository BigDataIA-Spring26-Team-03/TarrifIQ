import re
from typing import List

# Patterns in priority order — longest/most specific first.
# seen_spans prevents a position from being claimed by a lower-priority pattern
# after a higher-priority one already matched it.
_PATTERNS = [
    # 10-digit: 8471.30.0100
    (re.compile(r"\b(\d{4}\.\d{2}\.\d{4})\b"), "HTS_CODE"),
    # 8-digit: 8471.30.01
    (re.compile(r"\b(\d{4}\.\d{2}\.\d{2})\b"), "HTS_CODE"),
    # Range: 8471.30 through 8471.49
    (re.compile(r"\b(\d{4}\.\d{2}\s+through\s+\d{4}\.\d{2})\b", re.IGNORECASE), "HTS_RANGE"),
    # 6-digit: 8471.30
    (re.compile(r"\b(\d{4}\.\d{2})\b"), "HTS_CODE"),
    # Chapter: chapter 84
    (re.compile(r"\b(chapter\s+\d{1,2})\b", re.IGNORECASE), "HTS_CHAPTER"),
    # Heading: heading 8471
    (re.compile(r"\b(heading\s+\d{4})\b", re.IGNORECASE), "HTS_HEADING"),
]


def extract_hts_entities(text: str) -> List[dict]:
    """
    Extracts HTS-related entities from plain text using regex patterns.

    Returns a list of dicts sorted by start_char:
      [{"entity_text": str, "label": str, "start_char": int, "end_char": int}]

    Priority order ensures that "8471.30.0100" is labelled HTS_CODE (10-digit),
    not also matched as a 6-digit "8471.30" later.
    """
    seen_spans: set[tuple[int, int]] = set()
    entities: List[dict] = []

    for pattern, label in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()

            # Skip if any character in this span is already claimed
            if any(s <= start < e or s < end <= e for (s, e) in seen_spans):
                continue

            seen_spans.add((start, end))
            entities.append(
                {
                    "entity_text": match.group(1) if match.lastindex else match.group(),
                    "label": label,
                    "start_char": start,
                    "end_char": end,
                }
            )

    entities.sort(key=lambda x: x["start_char"])
    return entities
