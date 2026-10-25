"""
data_processing.py — ETL: Extract, Transform, Load helpers
===========================================================
Responsibilities
- Load CSV / Excel files (Extract).
- Validate schema, dates, and numeric values.
- Clean data: handle missing values, duplicates, outliers, invalid readings (Transform).
- Return a clean DataFrame ready for database insertion (Load step handled in database.py).

Design notes
- Raw data and cleaned data are kept separate so the UI can show both.
- Every cleaning action is logged to a human-readable list returned alongside the data.
- IQR-based outlier detection is used: simple, explainable, no black-box model needed.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column name normalisation map
# Accepts common user-typed variants and maps them to canonical names.
# ---------------------------------------------------------------------------
_COLUMN_ALIASES: dict[str, str] = {
    # date variants
    "date": "date",
    "reading_date": "date",
    "record_date": "date",
    "day": "date",
    "timestamp": "date",
    # consumption variants
    "consumption_kwh": "consumption_kwh",
    "consumption": "consumption_kwh",
    "kwh": "consumption_kwh",
    "units": "consumption_kwh",
    "units_kwh": "consumption_kwh",
    "energy_kwh": "consumption_kwh",
    "electricity_kwh": "consumption_kwh",
    "usage_kwh": "consumption_kwh",
    "power_kwh": "consumption_kwh",
    # optional meter name variants
    "meter": "meter_name",
    "meter_name": "meter_name",
    "meter_id": "meter_name",
    "location": "meter_name",
    "name": "meter_name",
}

REQUIRED_COLUMNS = {"date", "consumption_kwh"}


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_file(uploaded_file) -> tuple[pd.DataFrame | None, str]:
    """Read a Streamlit UploadedFile (CSV or Excel) into a DataFrame.

    Returns (dataframe, error_message).  On success error_message is "".
    """
    name: str = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()

    if len(raw_bytes) == 0:
        return None, "The uploaded file is empty. Please upload a file with data."

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw_bytes))
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw_bytes))
        else:
            return None, (
                f"Unsupported file type: '{uploaded_file.name}'. "
                "Please upload a .csv, .xlsx, or .xls file."
            )
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not read the file: {exc}"

    if df.empty:
        return None, "The file has no rows. Please check that it contains data."

    return df, ""


def normalise_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    """Rename columns using the alias map.

    Returns (renamed_df, list_of_applied_renames, error_message).
    error_message is non-empty when required columns are missing after rename.
    """
    rename_map: dict[str, str] = {}
    notes: list[str] = []
    lower_cols = {c.strip().lower(): c for c in df.columns}

    for lc_col, canonical in _COLUMN_ALIASES.items():
        if lc_col in lower_cols and canonical not in rename_map.values():
            original = lower_cols[lc_col]
            if original != canonical:
                rename_map[original] = canonical
                notes.append(f"Renamed column '{original}' → '{canonical}'.")

    df = df.rename(columns=rename_map)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        available = ", ".join(f"'{c}'" for c in df.columns)
        needed = ", ".join(f"'{c}'" for c in sorted(missing))
        error = (
            f"Required column(s) {needed} not found after normalising column names. "
            f"Available columns are: {available}. "
            "Please rename your columns or use the schema shown in the sidebar."
        )
        return df, notes, error

    return df, notes, ""


# ---------------------------------------------------------------------------
# Validation and cleaning
# ---------------------------------------------------------------------------

def clean_data(
    df: pd.DataFrame,
    outlier_iqr_factor: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Full cleaning pipeline (Transform step of ETL).

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with at least 'date' and 'consumption_kwh' columns.
    outlier_iqr_factor : float
        IQR multiplier for outlier detection (default 3.0 — mild but visible outliers
        are kept so the chart shows them; increase to 5.0 to be more permissive).

    Returns
    -------
    clean_df : pd.DataFrame
        Cleaned data with a DatetimeIndex, sorted by date.
    flagged_df : pd.DataFrame
        Rows removed or heavily modified, with a 'flag_reason' column.
    log : list[str]
        Human-readable log of every action taken and why.
    """
    log: list[str] = []
    flagged_rows: list[pd.DataFrame] = []
    df = df.copy()

    original_count = len(df)
    log.append(f"Loaded {original_count} row(s).")

    # ---- Step 1: parse dates -----------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)
    bad_dates = df["date"].isna()
    if bad_dates.any():
        bad = df[bad_dates].copy()
        bad["flag_reason"] = "Unparseable date"
        flagged_rows.append(bad)
        log.append(
            f"Removed {bad_dates.sum()} row(s) with unparseable dates "
            f"(e.g. index {df.index[bad_dates].tolist()[:5]})."
        )
        df = df[~bad_dates]

    # ---- Step 2: parse consumption -----------------------------------------
    df["consumption_kwh"] = pd.to_numeric(df["consumption_kwh"], errors="coerce")
    bad_numeric = df["consumption_kwh"].isna()
    if bad_numeric.any():
        bad = df[bad_numeric].copy()
        bad["flag_reason"] = "Non-numeric consumption_kwh"
        flagged_rows.append(bad)
        log.append(
            f"Removed {bad_numeric.sum()} row(s) where consumption_kwh "
            "could not be converted to a number."
        )
        df = df[~bad_numeric]

    # ---- Step 3: negative or zero readings ---------------------------------
    # Zero is technically possible (holiday, no usage) but negative is invalid.
    neg = df["consumption_kwh"] < 0
    if neg.any():
        bad = df[neg].copy()
        bad["flag_reason"] = "Negative consumption (invalid)"
        flagged_rows.append(bad)
        log.append(
            f"Removed {neg.sum()} row(s) with negative consumption values. "
            "Negative electricity readings are physically impossible."
        )
        df = df[~neg]

    # ---- Step 4: remove duplicate (date) rows ------------------------------
    dup = df.duplicated(subset=["date"], keep="first")
    if dup.any():
        bad = df[dup].copy()
        bad["flag_reason"] = "Duplicate date — kept first occurrence"
        flagged_rows.append(bad)
        log.append(
            f"Removed {dup.sum()} duplicate row(s) (same date). "
            "Kept the first occurrence for each date."
        )
        df = df[~dup]

    # ---- Step 5: missing consumption — interpolate -------------------------
    # After removing invalids and duplicates, there may be NaN consumptions
    # only if we hit rows with missing values that were numeric-parseable to NaN.
    missing_cons = df["consumption_kwh"].isna()
    if missing_cons.any():
        count = missing_cons.sum()
        df = df.sort_values("date")
        df["consumption_kwh"] = df["consumption_kwh"].interpolate(
            method="linear", limit_direction="both"
        )
        log.append(
            f"Imputed {count} missing consumption value(s) using linear interpolation. "
            "Assumption: consumption changes gradually between adjacent readings."
        )

    # ---- Step 6: sort by date ----------------------------------------------
    df = df.sort_values("date").reset_index(drop=True)

    # ---- Step 7: IQR-based outlier detection (flag only, do NOT remove) ----
    # Outliers are flagged in a separate column so the user is aware but the
    # data is not discarded — outliers may be real high-usage events.
    if len(df) >= 4:
        q1 = df["consumption_kwh"].quantile(0.25)
        q3 = df["consumption_kwh"].quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - outlier_iqr_factor * iqr
        upper_fence = q3 + outlier_iqr_factor * iqr
        outlier_mask = (df["consumption_kwh"] < lower_fence) | (
            df["consumption_kwh"] > upper_fence
        )
        df["is_outlier"] = outlier_mask
        n_out = outlier_mask.sum()
        if n_out:
            log.append(
                f"Flagged {n_out} row(s) as potential outliers "
                f"(IQR factor={outlier_iqr_factor}, fence: "
                f"[{lower_fence:.2f}, {upper_fence:.2f}] kWh). "
                "These rows are KEPT in the dataset but marked 'is_outlier=True'."
            )
        else:
            log.append(
                f"No outliers detected (IQR fence: "
                f"[{lower_fence:.2f}, {upper_fence:.2f}] kWh)."
            )
    else:
        df["is_outlier"] = False
        log.append("Too few rows to apply IQR outlier detection (need ≥ 4).")

    # ---- Step 8: future dates warning (don't remove; may be forecasts) -----
    today = pd.Timestamp.today().normalize()
    future = df["date"] > today
    if future.any():
        log.append(
            f"Note: {future.sum()} row(s) have future dates. "
            "These are kept but may be forecasted or sample values."
        )

    # ---- Summary -----------------------------------------------------------
    removed = original_count - len(df)
    log.append(
        f"Cleaning complete. {len(df)} row(s) retained, "
        f"{removed} row(s) removed."
    )

    flagged_df = (
        pd.concat(flagged_rows, ignore_index=True)
        if flagged_rows
        else pd.DataFrame(columns=df.columns.tolist() + ["flag_reason"])
    )

    return df, flagged_df, log


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def compute_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily readings into calendar months.

    Returns a DataFrame with columns: year_month (Period), consumption_kwh (sum).
    """
    tmp = df.copy()
    tmp["year_month"] = tmp["date"].dt.to_period("M")
    monthly = (
        tmp.groupby("year_month")["consumption_kwh"]
        .sum()
        .reset_index()
        .rename(columns={"consumption_kwh": "consumption_kwh"})
    )
    monthly["year_month_str"] = monthly["year_month"].astype(str)
    return monthly


def descriptive_stats(df: pd.DataFrame) -> dict:
    """Return a dictionary of key descriptive statistics for the UI."""
    c = df["consumption_kwh"]
    return {
        "count": int(c.count()),
        "mean": float(c.mean()),
        "median": float(c.median()),
        "std": float(c.std()),
        "min": float(c.min()),
        "max": float(c.max()),
        "total": float(c.sum()),
        "q25": float(c.quantile(0.25)),
        "q75": float(c.quantile(0.75)),
    }
