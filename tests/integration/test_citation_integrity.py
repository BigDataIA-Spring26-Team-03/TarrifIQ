"""
Layer 5 — Citation Integrity
Validates that all Federal Register citations:
  - Match the standard FR document number format (YYYY-NNNNN or YYYY-NNNNNN)
  - Have required fields (type, id, agency_short)
  - Don't show China-exclusive docs for non-China queries
  - Sources are from known agencies only
"""

import re
import pytest
from tests.conftest import api_required
from tests.integration.test_pipeline import PIPELINE_CASES

FR_DOC_PATTERN = re.compile(r"^\d{4}-\d{5,6}$")
VALID_SOURCES = {"USTR", "CBP", "USITC", "ITA", "EOP", "Census", ""}
VALID_CITATION_TYPES = {"federal_register", "snowflake_hts", "adder_source", "census_bureau"}

# China-exclusive title keywords — should never appear in non-China query citations
CHINA_EXCLUSIVE_KEYWORDS = [
    "products of china",
    "chinese goods",
    "section 301 on china",
    "china tariff action",
    "goods from china",
]

NON_CHINA_QUERIES = [
    ("steel wire from Germany",   "germany"),
    ("wine from France",          "france"),
    ("cotton t-shirts from Mexico", "mexico"),
    ("steel pipes from South Korea", "south korea"),
    ("coffee from Brazil",        "brazil"),
]


@api_required
@pytest.mark.parametrize(
    "query", [c[0] for c in PIPELINE_CASES],
    ids=[c[0][:45] for c in PIPELINE_CASES],
)
def test_fr_citation_doc_number_format(api, query):
    data = api.post("/query", json={"query": query}).json()
    for cit in (data.get("citations") or []):
        if cit.get("type") == "federal_register":
            doc_id = cit.get("id", "")
            assert FR_DOC_PATTERN.match(doc_id), (
                f"Invalid FR doc format: {doc_id!r} in query: {query}"
            )


@api_required
@pytest.mark.parametrize(
    "query", [c[0] for c in PIPELINE_CASES],
    ids=[c[0][:45] for c in PIPELINE_CASES],
)
def test_citation_required_fields(api, query):
    data = api.post("/query", json={"query": query}).json()
    for i, cit in enumerate(data.get("citations") or []):
        assert "type" in cit, f"Citation {i} missing 'type' in: {query}"
        assert "id"   in cit, f"Citation {i} missing 'id' in: {query}"
        assert cit["type"] in VALID_CITATION_TYPES, (
            f"Unknown citation type '{cit['type']}' in: {query}"
        )
        if cit["type"] == "federal_register":
            assert cit.get("agency_short"), (
                f"federal_register citation missing 'agency_short' for doc {cit.get('id')}"
            )


@api_required
@pytest.mark.parametrize(
    "query", [c[0] for c in PIPELINE_CASES],
    ids=[c[0][:45] for c in PIPELINE_CASES],
)
def test_citation_source_is_known_agency(api, query):
    data = api.post("/query", json={"query": query}).json()
    for cit in (data.get("citations") or []):
        src = cit.get("agency_short", "")
        assert src in VALID_SOURCES, (
            f"Unknown agency_short '{src}' in citation for: {query}"
        )


@api_required
@pytest.mark.parametrize(
    "query,country", NON_CHINA_QUERIES,
    ids=[c[0][:45] for c in NON_CHINA_QUERIES],
)
def test_no_china_exclusive_citations_for_non_china(api, query, country):
    data = api.post("/query", json={"query": query}).json()
    for cit in (data.get("citations") or []):
        title = (cit.get("title") or "").lower()
        for kw in CHINA_EXCLUSIVE_KEYWORDS:
            if kw in title and country not in title:
                pytest.fail(
                    f"China-exclusive citation shown for {country!r} query.\n"
                    f"  Query: {query}\n"
                    f"  Citation title: {cit.get('title')}\n"
                    f"  Doc ID: {cit.get('id')}"
                )


@api_required
def test_no_duplicate_citation_ids(api):
    data = api.post("/query", json={"query": "solar panels from China"}).json()
    ids = [c.get("id") for c in (data.get("citations") or []) if c.get("id")]
    assert len(ids) == len(set(ids)), (
        f"Duplicate citation IDs found: {[x for x in ids if ids.count(x) > 1]}"
    )


@api_required
def test_adder_citation_present_when_adder_nonzero(api):
    data = api.post("/query", json={"query": "lithium batteries from China"}).json()
    if (data.get("adder_rate") or 0) > 0:
        types = {c.get("type") for c in (data.get("citations") or [])}
        assert "adder_source" in types or "federal_register" in types, (
            "Expected adder_source or federal_register citation when adder_rate > 0"
        )
