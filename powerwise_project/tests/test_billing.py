import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from powerwise.billing import TariffConfig, calculate_bill


def test_calculate_bill_first_slab():
    result = calculate_bill(50, TariffConfig())
    assert result["slab1_units"] == 50
    assert result["slab2_units"] == 0
    assert result["slab3_units"] == 0
    assert result["total_bill"] == 200


def test_calculate_bill_multiple_slabs():
    result = calculate_bill(350, TariffConfig())
    assert result["slab1_units"] == 100
    assert result["slab2_units"] == 200
    assert result["slab3_units"] == 50
    assert result["total_bill"] == 1700


def test_negative_units_rejected():
    try:
        calculate_bill(-1)
    except ValueError:
        return
    raise AssertionError("Negative consumption should raise ValueError")
