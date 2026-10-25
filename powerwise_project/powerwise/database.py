"""
database.py — SQLite persistence via SQLAlchemy (Load step of ETL)
==================================================================
Responsibilities
- Define the ORM model for electricity readings.
- Create the database and table on first use.
- Insert cleaned readings, query all records, and clear the table.

Why SQLAlchemy?
- Provides a clean Python abstraction over raw SQL.
- Makes database operations readable and safe (parameterised queries).
- Core / expression layer is used (not ORM sessions) to keep things simple.

Why SQLite?
- Zero configuration, single file, no server needed.
- Perfect for a student project or local demo.
- Easy to inspect with DB Browser for SQLite.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Boolean,
    create_engine,
    text,
    inspect as sa_inspect,
)

# Default database path — created in the project root
_DEFAULT_DB_PATH = Path(__file__).parent.parent / "powerwise.db"


def get_engine(db_path: Optional[str] = None):
    """Return a SQLAlchemy engine for the given SQLite database path."""
    path = db_path or str(_DEFAULT_DB_PATH)
    return create_engine(f"sqlite:///{path}", echo=False)


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

metadata = MetaData()

readings_table = Table(
    "electricity_readings",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("date", String, nullable=False, unique=True),   # ISO-format string
    Column("consumption_kwh", Float, nullable=False),
    Column("meter_name", String, nullable=True),
    Column("is_outlier", Boolean, nullable=False, default=False),
)


def init_db(db_path: Optional[str] = None) -> None:
    """Create the database file and tables if they do not exist."""
    engine = get_engine(db_path)
    metadata.create_all(engine)


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def insert_readings(df: pd.DataFrame, db_path: Optional[str] = None) -> tuple[int, str]:
    """Insert cleaned readings into the database.

    Duplicate dates are skipped (INSERT OR IGNORE behaviour via a workaround).
    Returns (rows_inserted, error_message).
    """
    engine = get_engine(db_path)
    init_db(db_path)

    records = []
    for _, row in df.iterrows():
        record = {
            "date": str(row["date"].date()),
            "consumption_kwh": float(row["consumption_kwh"]),
            "meter_name": str(row.get("meter_name", "")) if "meter_name" in df.columns else None,
            "is_outlier": bool(row.get("is_outlier", False)),
        }
        records.append(record)

    if not records:
        return 0, "No records to insert."

    try:
        with engine.connect() as conn:
            # Use INSERT OR IGNORE so re-uploads don't raise errors
            conn.execute(
                text(
                    "INSERT OR IGNORE INTO electricity_readings "
                    "(date, consumption_kwh, meter_name, is_outlier) "
                    "VALUES (:date, :consumption_kwh, :meter_name, :is_outlier)"
                ),
                records,
            )
            conn.commit()
        return len(records), ""
    except Exception as exc:  # noqa: BLE001
        return 0, f"Database error during insert: {exc}"


def load_all_readings(db_path: Optional[str] = None) -> pd.DataFrame:
    """Load all readings from the database and return as a DataFrame."""
    engine = get_engine(db_path)
    init_db(db_path)
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT date, consumption_kwh, meter_name, is_outlier "
                    "FROM electricity_readings ORDER BY date"
                )
            )
            rows = result.fetchall()
            cols = result.keys()
        if not rows:
            return pd.DataFrame(columns=["date", "consumption_kwh", "meter_name", "is_outlier"])
        df = pd.DataFrame(rows, columns=list(cols))
        df["date"] = pd.to_datetime(df["date"])
        df["consumption_kwh"] = df["consumption_kwh"].astype(float)
        df["is_outlier"] = df["is_outlier"].astype(bool)
        return df
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame()


def clear_readings(db_path: Optional[str] = None) -> str:
    """Delete all rows from the electricity_readings table."""
    engine = get_engine(db_path)
    init_db(db_path)
    try:
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM electricity_readings"))
            conn.commit()
        return ""
    except Exception as exc:  # noqa: BLE001
        return f"Could not clear database: {exc}"


def get_row_count(db_path: Optional[str] = None) -> int:
    """Return the number of rows currently stored."""
    engine = get_engine(db_path)
    init_db(db_path)
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM electricity_readings")
            )
            return result.scalar() or 0
    except Exception:  # noqa: BLE001
        return 0
