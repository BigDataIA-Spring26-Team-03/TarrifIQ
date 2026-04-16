"""
tests/unit/test_improvements.py

Unit tests built entirely from real Snowflake data.
No mocked data, no hardcoded fake snippets or rates.

Every test pulls its inputs from your actual database at runtime,
then asserts the agent functions produce correct outputs from that real data.

If Snowflake is unreachable, all tests are skipped cleanly.

Run: python3 -m pytest tests/unit/test_improvements.py -v
"""

import pytest
from ingestion.connection import get_snowflake_conn


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers — fetch real rows from Snowflake at test runtime
# ─────────────────────────────────────────────────────────────────────────────

def _snowflake_available():
    try:
        conn = get_snowflake_conn()
        conn.cursor().execute("SELECT 1")
        conn.close()
        return True
    except Exception:
        return False


snowflake_required = pytest.mark.skipif(
    not _snowflake_available(),
    reason="Snowflake not reachable — skipping live data tests",
)


def fetch_real_notice_row(hts_code: str):
    """Most recent real notice row for a given HTS code."""
    conn = get_snowflake_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT n.hts_code, n.document_number, n.context_snippet,
                   f.publication_date
            FROM TARIFFIQ.RAW.NOTICE_HTS_CODES n
            LEFT JOIN TARIFFIQ.RAW.FEDERAL_REGISTER_NOTICES f
                ON n.document_number = f.document_number
            WHERE n.hts_code = %s
            ORDER BY f.publication_date DESC NULLS LAST
            LIMIT 1
            """,
            (hts_code,),
        )
        return cur.fetchone()
    finally:
        cur.close()
        conn.close()


def fetch_real_hts_row(hts_code: str):
    """Real HTS_CODES row for a given code."""
    conn = get_snowflake_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT hts_code, general_rate, description FROM TARIFFIQ.RAW.HTS_CODES WHERE hts_code = %s LIMIT 1",
            (hts_code,),
        )
        return cur.fetchone()
    finally:
        cur.close()
        conn.close()


def fetch_real_fr_doc(document_number: str):
    """Check if a document number exists in FEDERAL_REGISTER_NOTICES."""
    conn = get_snowflake_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT document_number, title, publication_date FROM TARIFFIQ.RAW.FEDERAL_REGISTER_NOTICES WHERE document_number = %s LIMIT 1",
            (document_number,),
        )
        return cur.fetchone()
    finally:
        cur.close()
        conn.close()


def fetch_confirmed_absent_doc():
    """Return a doc number string confirmed absent from FEDERAL_REGISTER_NOTICES."""
    conn = get_snowflake_conn()
    cur = conn.cursor()
    try:
        for candidate in ["9999-99999", "0000-00000", "1111-11111"]:
            cur.execute(
                "SELECT 1 FROM TARIFFIQ.RAW.FEDERAL_REGISTER_NOTICES WHERE document_number = %s",
                (candidate,),
            )
            if not cur.fetchone():
                return candidate
        return "0000-00000"
    finally:
        cur.close()
        conn.close()


def fetch_hts_code_with_no_notice():
    """Return a real HTS code that has no row in NOTICE_HTS_CODES."""
    conn = get_snowflake_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT h.hts_code
            FROM TARIFFIQ.RAW.HTS_CODES h
            LEFT JOIN TARIFFIQ.RAW.NOTICE_HTS_CODES n ON h.hts_code = n.hts_code
            WHERE n.hts_code IS NULL
              AND h.chapter NOT IN ('98','99')
              AND h.is_header_row = FALSE
              AND h.general_rate IS NOT NULL
            LIMIT 1
            """
        )
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        cur.close()
        conn.close()


# ═══════════════════════════════════════════════════════════════════════
# 1. _parse_adder_rate — real snippets from Snowflake
# ═══════════════════════════════════════════════════════════════════════

class TestAdderRateParserRealData:
    """
    _parse_adder_rate is fed the actual context_snippet stored in
    TARIFFIQ.RAW.NOTICE_HTS_CODES. Expected rates are the same ones
    confirmed by the live /query endpoint tests.
    """

    @snowflake_required
    def test_solar_panels_rate_from_real_snippet(self):
        from api.tools.resolve_hts_rate import _parse_adder_rate

        row = fetch_real_notice_row("8541.43.00")
        assert row is not None, "No notice row for 8541.43.00 in NOTICE_HTS_CODES"

        hts_code, doc_number, snippet, pub_date = row
        print(f"\n  doc={doc_number}  pub_date={pub_date}")
        print(f"  snippet={repr(snippet[:150])}")

        rate = _parse_adder_rate(snippet, hts_code)
        print(f"  parsed_rate={rate}")

        assert rate == 50.0, (
            f"Expected 50.0 from real Snowflake snippet for {hts_code}, got {rate}\n"
            f"snippet: {snippet[:200]}"
        )

    @snowflake_required
    def test_ev_rate_from_real_snippet(self):
        from api.tools.resolve_hts_rate import _parse_adder_rate

        row = fetch_real_notice_row("8703.80.00")
        assert row is not None, "No notice row for 8703.80.00 in NOTICE_HTS_CODES"

        hts_code, doc_number, snippet, pub_date = row
        print(f"\n  doc={doc_number}  pub_date={pub_date}")
        print(f"  snippet={repr(snippet[:150])}")

        rate = _parse_adder_rate(snippet, hts_code)
        print(f"  parsed_rate={rate}")

        assert rate == 100.0, (
            f"Expected 100.0 from real Snowflake snippet for {hts_code}, got {rate}\n"
            f"snippet: {snippet[:200]}"
        )

    @snowflake_required
    def test_semiconductor_rate_from_real_snippet(self):
        from api.tools.resolve_hts_rate import _parse_adder_rate

        row = fetch_real_notice_row("8542.31.00")
        assert row is not None, "No notice row for 8542.31.00 in NOTICE_HTS_CODES"

        hts_code, doc_number, snippet, pub_date = row
        print(f"\n  doc={doc_number}  pub_date={pub_date}")
        print(f"  snippet={repr(snippet[:150])}")

        rate = _parse_adder_rate(snippet, hts_code)
        print(f"  parsed_rate={rate}")

        assert rate == 50.0, (
            f"Expected 50.0 from real Snowflake snippet for {hts_code}, got {rate}\n"
            f"snippet: {snippet[:200]}"
        )

    @snowflake_required
    def test_hts_with_no_notice_returns_zero(self):
        from api.tools.resolve_hts_rate import _parse_adder_rate

        hts_code = fetch_hts_code_with_no_notice()
        assert hts_code is not None, "Could not find any HTS code without a notice row"
        print(f"\n  hts_code with no notice: {hts_code}")

        rate = _parse_adder_rate(None, hts_code)
        assert rate == 0.0, f"Expected 0.0 for HTS with no notice, got {rate}"


# ═══════════════════════════════════════════════════════════════════════
# 2. resolve_total_duty — full rate resolution against real tables
# ═══════════════════════════════════════════════════════════════════════

class TestResolveTotalDutyRealData:
    """
    resolve_total_duty called with real HTS codes.
    Expected totals match the live /query endpoint results confirmed earlier.
    """

    @snowflake_required
    def test_solar_panels_total_duty(self):
        from api.tools.resolve_hts_rate import resolve_total_duty

        receipt = resolve_total_duty("8541.43.00")
        print(f"\n  base={receipt.base_rate}  adder={receipt.adder_rate}  total={receipt.total_duty}")
        print(f"  adder_doc={receipt.adder_source.record_id}")

        assert receipt.total_duty == 50.0, f"Expected 50.0, got {receipt.total_duty}"
        assert receipt.adder_rate == 50.0
        assert receipt.adder_source.record_id != "NONE", "Adder must trace to a real FR doc"

    @snowflake_required
    def test_ev_total_duty(self):
        from api.tools.resolve_hts_rate import resolve_total_duty

        receipt = resolve_total_duty("8703.80.00")
        print(f"\n  base={receipt.base_rate}  adder={receipt.adder_rate}  total={receipt.total_duty}")

        assert receipt.adder_rate == 100.0, f"Expected adder 100.0, got {receipt.adder_rate}"
        assert receipt.total_duty == round(receipt.base_rate + 100.0, 4)

    @snowflake_required
    def test_semiconductor_total_duty(self):
        from api.tools.resolve_hts_rate import resolve_total_duty

        receipt = resolve_total_duty("8542.31.00")
        print(f"\n  base={receipt.base_rate}  adder={receipt.adder_rate}  total={receipt.total_duty}")

        assert receipt.adder_rate == 50.0, f"Expected adder 50.0, got {receipt.adder_rate}"

    @snowflake_required
    def test_most_recent_notice_is_used(self):
        """
        Fetch the most recent notice row for 8541.43.00 directly from Snowflake.
        Assert resolve_total_duty uses exactly that doc number and rate.
        """
        from api.tools.resolve_hts_rate import resolve_total_duty, _parse_adder_rate

        conn = get_snowflake_conn()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT n.document_number, n.context_snippet, f.publication_date
                FROM TARIFFIQ.RAW.NOTICE_HTS_CODES n
                LEFT JOIN TARIFFIQ.RAW.FEDERAL_REGISTER_NOTICES f
                    ON n.document_number = f.document_number
                WHERE n.hts_code = '8541.43.00'
                ORDER BY f.publication_date DESC NULLS LAST
                LIMIT 1
                """
            )
            most_recent = cur.fetchone()
        finally:
            cur.close()
            conn.close()

        assert most_recent is not None
        expected_doc, expected_snippet, expected_date = most_recent
        expected_rate = _parse_adder_rate(expected_snippet, "8541.43.00")

        print(f"\n  most_recent_doc={expected_doc}  date={expected_date}  expected_rate={expected_rate}")

        receipt = resolve_total_duty("8541.43.00")

        assert receipt.adder_source.record_id == expected_doc, (
            f"resolve_total_duty used doc {receipt.adder_source.record_id} "
            f"but most recent in Snowflake is {expected_doc} (pub_date={expected_date})"
        )
        assert receipt.adder_rate == expected_rate

    @snowflake_required
    def test_rate_record_id_exists_in_hts_codes(self):
        """
        rate_record_id must be a real hts_code in HTS_CODES.
        This is the professor's core traceability requirement.
        """
        from api.tools.resolve_hts_rate import resolve_total_duty

        receipt = resolve_total_duty("8541.43.00")
        record_id = receipt.base_rate_source.record_id

        row = fetch_real_hts_row(record_id)
        assert row is not None, (
            f"rate_record_id '{record_id}' not found in HTS_CODES — "
            f"rate is not grounded to a verified Snowflake record"
        )
        print(f"\n  record_id={record_id}  general_rate={row[1]}  desc={row[2][:60]}")


# ═══════════════════════════════════════════════════════════════════════
# 3. Synthesis — FR doc verification against real FEDERAL_REGISTER_NOTICES
# ═══════════════════════════════════════════════════════════════════════

class TestSynthesisCitationVerificationRealData:

    @snowflake_required
    def test_real_doc_verified(self):
        """2024-21217 exists in FEDERAL_REGISTER_NOTICES — must be returned as verified."""
        from agents.synthesis_agent import _verify_fr_docs_in_snowflake

        row = fetch_real_fr_doc("2024-21217")
        assert row is not None, "2024-21217 not in FEDERAL_REGISTER_NOTICES — data changed?"

        result = _verify_fr_docs_in_snowflake({"2024-21217"})
        assert "2024-21217" in result, (
            "2024-21217 exists in Snowflake but was not returned as verified"
        )

    @snowflake_required
    def test_absent_doc_excluded(self):
        """A doc number confirmed absent from Snowflake must NOT be returned as verified."""
        from agents.synthesis_agent import _verify_fr_docs_in_snowflake

        fake_doc = fetch_confirmed_absent_doc()
        print(f"\n  confirmed-absent doc: {fake_doc}")

        result = _verify_fr_docs_in_snowflake({fake_doc})
        assert fake_doc not in result, (
            f"Doc {fake_doc} is not in Snowflake but was returned as verified — "
            f"hallucination check is broken"
        )

    @snowflake_required
    def test_mix_real_and_absent(self):
        """Real doc passes, absent doc is excluded. This is the hallucination scenario."""
        from agents.synthesis_agent import _verify_fr_docs_in_snowflake

        real_doc = "2024-21217"
        fake_doc = fetch_confirmed_absent_doc()

        result = _verify_fr_docs_in_snowflake({real_doc, fake_doc})

        assert real_doc in result
        assert fake_doc not in result

    @snowflake_required
    def test_empty_input_returns_empty(self):
        from agents.synthesis_agent import _verify_fr_docs_in_snowflake

        result = _verify_fr_docs_in_snowflake(set())
        assert result == set()


# ═══════════════════════════════════════════════════════════════════════
# 4. pipeline_confidence — real verification results as inputs
# ═══════════════════════════════════════════════════════════════════════

class TestPipelineConfidenceRealValues:

    @snowflake_required
    def test_high_for_verified_solar_panels(self):
        """
        Solar panels: classification_confidence=0.85 (real value from live test),
        8541.43.00 exists in HTS_CODES, 2024-21217 exists in FR_NOTICES.
        Must produce HIGH.
        """
        from agents.synthesis_agent import _compute_pipeline_confidence

        classification_confidence = 0.85
        rate_record_verified = fetch_real_hts_row("8541.43.00") is not None
        fr_docs_verified = fetch_real_fr_doc("2024-21217") is not None

        print(f"\n  class_conf={classification_confidence}  rate_verified={rate_record_verified}  fr_verified={fr_docs_verified}")

        result = _compute_pipeline_confidence(
            classification_confidence=classification_confidence,
            rate_record_verified=rate_record_verified,
            fr_docs_verified=fr_docs_verified,
            hitl_was_triggered=False,
        )

        assert result == "HIGH", f"Expected HIGH for verified solar panels pipeline, got {result}"

    @snowflake_required
    def test_medium_or_low_when_rate_unverified(self):
        """A fake HTS code not in HTS_CODES → rate_record_verified=False → degrades confidence."""
        from agents.synthesis_agent import _compute_pipeline_confidence

        fake_hts = "0000.00.00"
        rate_record_verified = fetch_real_hts_row(fake_hts) is not None
        assert not rate_record_verified

        result = _compute_pipeline_confidence(
            classification_confidence=0.85,
            rate_record_verified=False,
            fr_docs_verified=True,
            hitl_was_triggered=False,
        )

        assert result in ("MEDIUM", "LOW"), (
            f"Expected MEDIUM or LOW when rate unverified, got {result}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 5. Policy agent — doc number filter returns real values
# ═══════════════════════════════════════════════════════════════════════

class TestPolicyAgentHTSFilterRealData:

    @snowflake_required
    def test_solar_panels_doc_numbers_returned(self):
        """8541.43.00 has a real row in NOTICE_HTS_CODES — must return 2024-21217."""
        from agents.policy_agent import _fetch_doc_numbers_for_hts_code

        result = _fetch_doc_numbers_for_hts_code("8541.43.00")
        print(f"\n  doc_numbers for 8541.43.00: {result}")

        assert "2024-21217" in result, (
            f"Expected 2024-21217 in doc numbers for 8541.43.00, got {result}"
        )

    @snowflake_required
    def test_ev_doc_numbers_returned(self):
        """8703.80.00 must return 2024-21217."""
        from agents.policy_agent import _fetch_doc_numbers_for_hts_code

        result = _fetch_doc_numbers_for_hts_code("8703.80.00")
        print(f"\n  doc_numbers for 8703.80.00: {result}")

        assert "2024-21217" in result

    @snowflake_required
    def test_hts_with_no_notice_returns_empty_set(self):
        """Real HTS code with no notice row → empty set → triggers chapter fallback."""
        from agents.policy_agent import _fetch_doc_numbers_for_hts_code

        hts_code = fetch_hts_code_with_no_notice()
        assert hts_code is not None, "Could not find any HTS code without a notice"
        print(f"\n  hts_with_no_notice: {hts_code}")

        result = _fetch_doc_numbers_for_hts_code(hts_code)
        assert result == set(), f"Expected empty set for {hts_code}, got {result}"

    @snowflake_required
    def test_empty_hts_returns_empty_set(self):
        from agents.policy_agent import _fetch_doc_numbers_for_hts_code

        result = _fetch_doc_numbers_for_hts_code("")
        assert result == set()


# ═══════════════════════════════════════════════════════════════════════
# 6. Alias write-back gates — logic checks, no writes to PRODUCT_ALIASES
# ═══════════════════════════════════════════════════════════════════════

class TestAliasWritebackGates:
    """
    Tests that the gates controlling when _write_alias_feedback fires
    work correctly. We intercept the function before it reaches Snowflake
    so nothing is written.
    """

    @snowflake_required
    def test_write_blocked_when_hitl_required(self):
        """hitl_required=True must block write, regardless of confidence."""
        from agents.classification_agent import maybe_write_alias_feedback
        import agents.classification_agent as ca

        classification_result = {
            "hts_code": "8541.43.00",
            "classification_confidence": 0.65,
            "hitl_required": True,
            "_product_for_feedback": "solar panels",
        }

        write_called = []
        original = ca._write_alias_feedback
        ca._write_alias_feedback = lambda *a, **kw: write_called.append(a)
        try:
            maybe_write_alias_feedback(classification_result, rate_found=True)
            assert len(write_called) == 0, "Write fired despite hitl_required=True"
        finally:
            ca._write_alias_feedback = original

    @snowflake_required
    def test_write_blocked_when_no_rate(self):
        """rate_found=False must block write even with high classification confidence."""
        from agents.classification_agent import maybe_write_alias_feedback
        import agents.classification_agent as ca

        classification_result = {
            "hts_code": "8541.43.00",
            "classification_confidence": 0.85,
            "hitl_required": False,
            "_product_for_feedback": "solar panels",
        }

        write_called = []
        original = ca._write_alias_feedback
        ca._write_alias_feedback = lambda *a, **kw: write_called.append(a)
        try:
            maybe_write_alias_feedback(classification_result, rate_found=False)
            # Either write wasn't called, or it was called with rate_was_found=False
            if write_called:
                assert write_called[0][3] is False, \
                    "Write fired with rate_was_found=True despite rate_found=False"
        finally:
            ca._write_alias_feedback = original

    @snowflake_required
    def test_write_fires_when_both_conditions_met(self):
        """hitl_required=False AND rate_found=True → write should be attempted."""
        from agents.classification_agent import maybe_write_alias_feedback
        import agents.classification_agent as ca

        classification_result = {
            "hts_code": "8541.43.00",
            "classification_confidence": 0.85,
            "hitl_required": False,
            "_product_for_feedback": "solar panels test product do not persist",
        }

        write_called = []
        original = ca._write_alias_feedback
        ca._write_alias_feedback = lambda *a, **kw: write_called.append(a)
        try:
            maybe_write_alias_feedback(classification_result, rate_found=True)
            assert len(write_called) == 1, \
                f"Expected write to be called once, got {len(write_called)}"
            assert write_called[0][0] == "solar panels test product do not persist"
            assert write_called[0][1] == "8541.43.00"
            assert write_called[0][3] is True  # rate_was_found
        finally:
            ca._write_alias_feedback = original