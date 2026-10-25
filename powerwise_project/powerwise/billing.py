"""
billing.py — Electricity Bill Estimation
=========================================
Implements a transparent, configurable tariff calculation.

Tariff structure (Indian residential slab rates — for demonstration only)
--------------------------------------------------------------------------
The rates below are illustrative.  They are clearly labelled in the UI as
"example/user-selected" and NOT official utility rates.

Slab structure (units = kWh consumed in a billing period):
  0  – 100 kWh  :  ₹ 3.00 / kWh   (first slab)
  101 – 300 kWh :  ₹ 5.00 / kWh   (second slab)
  > 300 kWh     :  ₹ 7.00 / kWh   (third slab)
  Fixed charge  :  ₹ 50.00 / month (meter rent / fixed charge)

These values are fully configurable from the UI.

Bill formula
------------
  bill = fixed_charge
       + min(units, slab1_limit) * rate1
       + max(0, min(units - slab1_limit, slab2_limit - slab1_limit)) * rate2
       + max(0, units - slab2_limit) * rate3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Default tariff configuration
# ---------------------------------------------------------------------------

@dataclass
class TariffConfig:
    """Holds the tariff parameters.  All monetary values are in ₹ (INR)."""

    # Slab boundaries in kWh
    slab1_limit: float = 100.0   # first slab covers 0–100 kWh
    slab2_limit: float = 300.0   # second slab covers 101–300 kWh
                                  # third slab covers > 300 kWh

    # Rate per kWh for each slab (₹)
    rate1: float = 3.00
    rate2: float = 5.00
    rate3: float = 7.00

    # Fixed monthly charge (₹)
    fixed_charge: float = 50.00

    # Currency symbol shown in the UI
    currency: str = "₹"

    # Label shown in the UI to remind users this is illustrative
    disclaimer: str = (
        "Tariff rates shown are for demonstration only. "
        "They do NOT represent official rates from any electricity utility."
    )


DEFAULT_TARIFF = TariffConfig()


# ---------------------------------------------------------------------------
# Core billing function
# ---------------------------------------------------------------------------

def calculate_bill(units: float, tariff: TariffConfig = DEFAULT_TARIFF) -> dict:
    """Compute the electricity bill for a given number of units consumed.

    Parameters
    ----------
    units : float
        Total kWh consumed in the billing period.
    tariff : TariffConfig
        Tariff parameters to use.

    Returns
    -------
    dict with keys:
        units_consumed, slab1_units, slab2_units, slab3_units,
        slab1_charge, slab2_charge, slab3_charge,
        energy_charge, fixed_charge, total_bill, currency, tariff
    """
    if units < 0:
        raise ValueError(f"Units consumed cannot be negative. Got: {units}")

    units = float(units)

    # How many units fall in each slab?
    slab1_units = min(units, tariff.slab1_limit)
    slab2_units = max(0.0, min(units - tariff.slab1_limit,
                               tariff.slab2_limit - tariff.slab1_limit))
    slab3_units = max(0.0, units - tariff.slab2_limit)

    slab1_charge = slab1_units * tariff.rate1
    slab2_charge = slab2_units * tariff.rate2
    slab3_charge = slab3_units * tariff.rate3

    energy_charge = slab1_charge + slab2_charge + slab3_charge
    total_bill = energy_charge + tariff.fixed_charge

    return {
        "units_consumed": round(units, 2),
        "slab1_units": round(slab1_units, 2),
        "slab2_units": round(slab2_units, 2),
        "slab3_units": round(slab3_units, 2),
        "slab1_charge": round(slab1_charge, 2),
        "slab2_charge": round(slab2_charge, 2),
        "slab3_charge": round(slab3_charge, 2),
        "energy_charge": round(energy_charge, 2),
        "fixed_charge": round(tariff.fixed_charge, 2),
        "total_bill": round(total_bill, 2),
        "currency": tariff.currency,
        "tariff": tariff,
    }


def estimate_bill_for_period(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    tariff: TariffConfig = DEFAULT_TARIFF,
) -> dict:
    """Estimate bill for a date range from the cleaned DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data with 'date' (datetime) and 'consumption_kwh' columns.
    start_date, end_date : str or None
        ISO format date strings (e.g. '2024-01-01').  If None, uses full range.
    tariff : TariffConfig
        Tariff parameters.

    Returns
    -------
    dict with bill breakdown plus 'start_date', 'end_date', 'n_days'.
    """
    if df.empty:
        return {"error": "No data available to estimate the bill."}

    tmp = df.copy()
    if start_date:
        tmp = tmp[tmp["date"] >= pd.Timestamp(start_date)]
    if end_date:
        tmp = tmp[tmp["date"] <= pd.Timestamp(end_date)]

    if tmp.empty:
        return {"error": "No records found in the selected date range."}

    units = float(tmp["consumption_kwh"].sum())
    result = calculate_bill(units, tariff)
    result["start_date"] = str(tmp["date"].min().date())
    result["end_date"] = str(tmp["date"].max().date())
    result["n_days"] = int(tmp["date"].nunique())
    return result
