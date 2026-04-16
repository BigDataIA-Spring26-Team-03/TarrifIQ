import requests

BASE_URL = "https://www.federalregister.gov/api/v1/documents.json"

QUERIES = {
    "USTR_tariff_main": {
        "conditions[agencies][]": "trade-representative-office-of-united-states",
        "conditions[term]": "tariff",
    },
    "USTR_exclusions": {
        "conditions[agencies][]": "trade-representative-office-of-united-states",
        "conditions[term]": "product exclusion",
    },
    "USTR_GSP": {
        "conditions[agencies][]": "trade-representative-office-of-united-states",
        "conditions[term]": "generalized system of preferences",
    },
    "EOP_tariff_EOs": {
        "conditions[agencies][]": "executive-office-of-the-president",
        "conditions[term]": "tariff",
    },
    "BIS_section232": {
        "conditions[agencies][]": "industry-and-security-bureau",
        "conditions[term]": "section 232",
    },
    "USITC_hts_amendments": {
        "conditions[agencies][]": "international-trade-commission",
        "conditions[term]": "harmonized tariff schedule amendment",
    },
    "USITC_TRQ": {
        "conditions[agencies][]": "international-trade-commission",
        "conditions[term]": "tariff rate quota",
    },
    "CBP_chapter99": {
        "conditions[agencies][]": "u-s-customs-and-border-protection",
        "conditions[term]": "chapter 99",
    },
    "CBP_section301": {
        "conditions[agencies][]": "u-s-customs-and-border-protection",
        "conditions[term]": "section 301",
    },
    "Treasury_TRQ": {
        "conditions[agencies][]": "treasury-department",
        "conditions[term]": "tariff rate quota",
    },
}

total_raw = 0
print(f"{'Query':<30} {'Count':>8}")
print("-" * 40)

for name, conditions in QUERIES.items():
    params = {
        **conditions,
        "conditions[publication_date][gte]": "2018-01-01",
        "conditions[publication_date][lte]": "2026-04-14",
        "per_page": 1,
        "fields[]": "document_number",
    }

    try:
        resp = requests.get(BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        count = resp.json().get("count", "ERROR")
        total_raw += count if isinstance(count, int) else 0
    except Exception as e:
        count = f"FAILED: {e}"

    print(f"{name:<30} {str(count):>8}")

print("-" * 40)
print(f"{'TOTAL RAW (with overlaps)':<30} {total_raw:>8}")
print()
print("Dropped from previous version:")
print("  - CBP_additional_duties (355) — too noisy")
print("  - Commerce_301 (191)          — heavy overlap with USTR_tariff_main")
print("  - USITC_duty_rate (9)         — fully contained in hts_amendments")
print("  - USTR_USMCA (11)             — contained in USTR_tariff_main")
print()
print("Expected raw total: ~810")
print("Expected unique after dedup: ~760-780")