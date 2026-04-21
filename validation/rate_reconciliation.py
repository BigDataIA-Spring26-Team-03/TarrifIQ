"""
Rate reconciliation validation — arithmetic check of tariff components.

Verifies that base_rate + adder_rate ≈ total_duty within tolerance.
"""

from api.schemas import RateReconciliation

TOLERANCE = 0.01  # 0.01% tolerance for floating point arithmetic


def validate(base_rate: float, adder_rate: float, total_duty: float) -> RateReconciliation:
    """
    Validate tariff rate arithmetic: base + adder = total (±tolerance).

    Args:
        base_rate: Base duty rate in percentage (e.g., 2.5)
        adder_rate: Additional duty rate in percentage (e.g., 25.0)
        total_duty: Total duty rate in percentage (e.g., 27.5)

    Returns:
        RateReconciliation with calculation string and pass/fail status.
    """
    base = base_rate or 0.0
    adder = adder_rate or 0.0
    total = total_duty or 0.0

    expected = round(base + adder, 4)
    discrepancy = abs(expected - total)
    check_passed = discrepancy <= TOLERANCE

    calculation = f"{base:.2f}% + {adder:.2f}% = {expected:.2f}% (reported: {total:.2f}%)"
    return RateReconciliation(calculation=calculation, check_passed=check_passed)
