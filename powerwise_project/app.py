"""
app.py — PowerWise Streamlit Application
=========================================
Entry point: streamlit run app.py

Page layout
-----------
Sidebar : navigation and global controls
Main    : pages rendered based on sidebar selection

Pages
1. 📥 Import Data
2. 🧹 Clean & Inspect
3. 📊 Explore & Visualise
4. 🧾 Bill Estimator
5. 🔮 Forecast
6. 🌦️ Weather Insights
7. 💾 Export
"""

import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Ensure the project root is on sys.path so `powerwise` package resolves
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from powerwise.data_processing import (
    clean_data,
    compute_monthly,
    descriptive_stats,
    load_file,
    normalise_columns,
)
from powerwise.database import (
    clear_readings,
    get_row_count,
    init_db,
    insert_readings,
    load_all_readings,
)
from powerwise.billing import (
    DEFAULT_TARIFF,
    TariffConfig,
    calculate_bill,
    estimate_bill_for_period,
)
from powerwise.analysis import (
    plot_daily_consumption,
    plot_monthly_bar,
    plot_consumption_histogram,
    plot_rolling_stats,
    plot_monthly_heatmap,
    plot_actual_vs_predicted,
    plot_future_forecast,
)
from powerwise.weather_api import (
    WeatherAPIError,
    fetch_weather,
    fetch_historical_weather,
    daily_forecast_frame,
    geocode_location,
    weather_code_description,
)
from powerwise.forecasting import (
    MIN_RECORDS_ARIMA,
    MIN_RECORDS_LINEAR,
    forecast_future_arima,
    forecast_future_linear,
    train_arima,
    train_linear_model,
    train_weather_linear_model,
    forecast_future_weather_linear,
)

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PowerWise",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — restrained PowerWise dark theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --pw-bg: #111827;
        --pw-panel: #182235;
        --pw-blue: #38BDF8;
        --pw-amber: #FBBF24;
        --pw-text: #E5E7EB;
        --pw-muted: #9CA3AF;
        --pw-green: #34D399;
        --pw-red: #F87171;
        --pw-border: #334155;
    }
    .stApp { background: var(--pw-bg); color: var(--pw-text); }
    [data-testid="stHeader"] { background: var(--pw-bg); }
    [data-testid="stSidebar"] { background: var(--pw-panel); border-right: 1px solid var(--pw-border); }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p { color: var(--pw-text); }
    h1, h2, h3 { color: var(--pw-text); letter-spacing: -0.015em; }
    p, label, [data-testid="stMarkdownContainer"] { color: #D1D5DB; }
    [data-testid="stMetric"] {
        background: var(--pw-panel);
        border: 1px solid var(--pw-border);
        border-left: 3px solid var(--pw-blue);
        padding: 12px 14px;
        border-radius: 9px;
    }
    [data-testid="stMetricLabel"] { color: var(--pw-muted) !important; }
    [data-testid="stMetricValue"] { color: var(--pw-text) !important; }
    [data-testid="stMetricDelta"] { color: var(--pw-green) !important; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--pw-border) !important; }
    [data-testid="stFileUploader"] section {
        background: #151F30;
        border: 1px dashed var(--pw-blue);
        border-radius: 9px;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 8px;
        border: 1px solid #3B82A0;
        background: #164E63;
        color: #F9FAFB;
        font-weight: 600;
        transition: background-color .15s ease, border-color .15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--pw-blue);
        background: #1D6178;
        color: white;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--pw-border);
        border-radius: 8px;
        overflow: hidden;
    }
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { background-color: #151F30; }
    .pw-hero { padding: 4px 0 18px; margin: 4px 0 22px; border-bottom: 1px solid var(--pw-border); }
    .pw-hero-kicker { color: var(--pw-blue); font-size: .75rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; margin-bottom: 6px; }
    .pw-hero-title { color: var(--pw-text); font-size: clamp(1.45rem, 2.5vw, 1.9rem); font-weight: 700; line-height: 1.2; margin: 0 0 6px; }
    .pw-hero-desc { color: var(--pw-muted); font-size: .95rem; margin: 0; }
    .pw-section { color: var(--pw-blue); font-weight: 700; }
    .pw-status { display: flex; align-items: center; gap: 9px; padding: 9px 11px; border-radius: 8px; background: #151F30; border: 1px solid var(--pw-border); color: var(--pw-text); font-size: .84rem; }
    .pw-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--pw-green); display: inline-block; flex: 0 0 8px; }
    .pw-log { max-height: 240px; overflow-y: auto; padding: 12px 14px; border-radius: 8px; background: #101827; border: 1px solid var(--pw-border); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .83rem; line-height: 1.6; color: var(--pw-text); }
    .pw-total { padding: 15px 17px; border: 1px solid #8A6A18; border-left: 3px solid var(--pw-amber); border-radius: 8px; background: var(--pw-panel); margin: 12px 0; }
    .pw-total-label { color: var(--pw-muted); text-transform: uppercase; font-size: .75rem; letter-spacing: .07em; }
    .pw-total-value { color: var(--pw-amber); font-size: 1.8rem; font-weight: 750; margin: 3px 0 0; }
    .pw-footer { margin-top: 36px; padding: 14px 0 6px; border-top: 1px solid var(--pw-border); color: var(--pw-muted); font-size: .78rem; text-align: center; }
    [data-testid="stExpander"] { border-color: var(--pw-border); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("⚡ PowerWise")
st.sidebar.caption("Electricity Analytics")
st.sidebar.markdown("---")

PAGES = [
    "📥 Import Data",
    "🧹 Clean & Inspect",
    "📊 Explore & Visualise",
    "🧾 Bill Estimator",
    "🔮 Forecast",
    "🌦️ Weather Insights",
    "💾 Export",
]
page = st.sidebar.radio("Navigate to", PAGES)

st.sidebar.markdown("---")
with st.sidebar.expander("Required data columns"):
    st.markdown(
        "- `date` — reading date (YYYY-MM-DD)\n"
        "- `consumption_kwh` — daily kWh\n\n"
        "Optional: `meter_name`"
    )
st.sidebar.markdown("---")
# Download sample file button
sample_path = ROOT / "data" / "sample_electricity_data.csv"
if sample_path.exists():
    with open(sample_path, "rb") as f:
        st.sidebar.download_button(
            label="⬇️ Download Sample CSV",
            data=f.read(),
            file_name="sample_electricity_data.csv",
            mime="text/csv",
        )
    st.sidebar.caption("⚠️ Sample/synthetic data — not real utility data.")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "clean_df" not in st.session_state:
    st.session_state.clean_df = None
if "flagged_df" not in st.session_state:
    st.session_state.flagged_df = None
if "clean_log" not in st.session_state:
    st.session_state.clean_log = []
if "db_loaded" not in st.session_state:
    st.session_state.db_loaded = False

# Initialise DB on startup
init_db()

# Sidebar database status (display only; does not alter database behavior)
try:
    _sidebar_db_count = get_row_count()
    _sidebar_status_text = f"{_sidebar_db_count} saved record(s)" if _sidebar_db_count else "No records yet"
    st.sidebar.markdown(
        f"<div class='pw-status'><span class='pw-dot'></span><span><b>Database Status</b><br>{_sidebar_status_text}</span></div>",
        unsafe_allow_html=True,
    )
except Exception:
    st.sidebar.caption("Database status unavailable")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _require_clean_data():
    """Return clean_df or show an informative warning and return None."""
    df = st.session_state.clean_df
    if df is None or df.empty:
        # Try loading from database
        db_df = load_all_readings()
        if not db_df.empty:
            st.session_state.clean_df = db_df
            return db_df
        st.warning(
            "No cleaned data available yet.  "
            "Go to **📥 Import Data** and upload a file, or load the sample dataset."
        )
        return None
    return df


def _bytes_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _bytes_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="PowerWise")
    return buf.getvalue()



def _hero(icon: str, title: str, description: str) -> None:
    """Render a consistent page banner without changing page behavior."""
    st.markdown(
        f"""<div class="pw-hero">
        <div class="pw-hero-kicker">POWERWISE · ENERGY ANALYTICS</div>
        <div class="pw-hero-title">{icon} &nbsp; {title}</div>
        <p class="pw-hero-desc">{description}</p>
        </div>""",
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 1 — Import Data
# ===========================================================================
if page == "📥 Import Data":
    _hero("📥", "Import Electricity Data", "Bring your daily meter readings into one place to begin the analysis.")
    st.markdown("**STEP 1 OF 3 · IMPORT**  →  **STEP 2 · CLEAN**  →  **STEP 3 · ANALYSE**")
    st.markdown(
        "Upload a **CSV or Excel** file containing electricity consumption records.  \n"
        "The file must have at least two columns: **`date`** and **`consumption_kwh`**.  \n"
        "Use the sample file if you want to try the dashboard before adding your own data."
    )
    st.markdown("<div style='padding:12px 16px;margin:8px 0 16px;border:1px dashed rgba(0,180,216,.6);border-radius:12px;background:rgba(22,33,62,.38);color:#B7C4D8'>⬆️ <b style='color:#EAF7FF'>Drop your CSV or Excel file below</b><br>Required columns: <code>date</code> and <code>consumption_kwh</code> · Excel and CSV supported</div>", unsafe_allow_html=True)

    col_upload, col_sample = st.columns([2, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Max 200 MB. Supported formats: .csv, .xlsx, .xls",
        )

    with col_sample:
        st.markdown("#### Or load sample data")
        if st.button("📂 Load built-in sample dataset", use_container_width=True):
            if sample_path.exists():
                with open(sample_path, "rb") as f:
                    content = f.read()

                class _FakeFile:
                    name = "sample_electricity_data.csv"
                    def read(self):
                        return content

                uploaded = _FakeFile()
                st.info("⚠️ This is **synthetic/sample** data for demonstration only.")
            else:
                st.error(
                    "Sample file not found.  "
                    "Run `python data/generate_sample_data.py` to create it."
                )
                uploaded = None

    if uploaded is not None:
        raw_df, err = load_file(uploaded)
        if err:
            st.error(f"❌ {err}")
        else:
            st.success(f"✅ File loaded: **{uploaded.name}** — {len(raw_df)} rows, {len(raw_df.columns)} columns.")

            # Normalise columns
            norm_df, rename_notes, col_err = normalise_columns(raw_df)
            if rename_notes:
                for note in rename_notes:
                    st.info(f"ℹ️ {note}")
            if col_err:
                st.error(f"❌ {col_err}")
            else:
                st.session_state.raw_df = norm_df
                st.session_state.clean_df = None  # reset previous clean

                st.markdown("#### Data Preview (first 10 rows)")
                st.dataframe(norm_df.head(10), use_container_width=True)

                st.markdown("#### File summary")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total rows", len(norm_df))
                c2.metric("Columns", len(norm_df.columns))
                c3.metric("Missing values", int(norm_df.isnull().sum().sum()))
                c4.metric("Duplicate rows", int(norm_df.duplicated().sum()))

                st.markdown(
                    "_Proceed to **🧹 Clean & Inspect** to validate and clean this data._"
                )


# ===========================================================================
# PAGE 2 — Clean & Inspect
# ===========================================================================
elif page == "🧹 Clean & Inspect":
    _hero("🧹", "Data Cleaning & Inspection", "Validate readings, review quality issues, and prepare reliable data for analysis.")

    raw_df = st.session_state.get("raw_df")
    if raw_df is None:
        st.warning("Please import a file on the **📥 Import Data** page first.")
        st.stop()

    st.markdown("#### Raw Data Preview")
    st.dataframe(raw_df.head(20), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Cleaning pipeline")
    clean_steps_col, sensitivity_col = st.columns([1.6, 1])

    with clean_steps_col:
        st.markdown(
            "The pipeline validates dates and numeric readings, removes negative values "
            "and duplicate dates, interpolates missing values, and flags IQR outliers "
            "without removing them."
        )
        st.caption("Review the raw preview above before applying these checks.")

    with sensitivity_col:
        iqr_factor = st.slider(
            "Outlier sensitivity (IQR factor)",
            min_value=1.5,
            max_value=6.0,
            value=3.0,
            step=0.5,
            help=(
                "Lower = more sensitive (flags more points). "
                "Higher = less sensitive. Default 3.0 is a reasonable balance."
            ),
        )

    if st.button("▶️ Run Cleaning", use_container_width=True, type="primary"):
        with st.spinner("Cleaning data…"):
            clean_df, flagged_df, log = clean_data(raw_df, outlier_iqr_factor=iqr_factor)

        st.session_state.clean_df = clean_df
        st.session_state.flagged_df = flagged_df
        st.session_state.clean_log = log

        # Save to DB
        err = ""
        if not clean_df.empty:
            n_inserted, err = insert_readings(clean_df)
            if err:
                st.warning(f"⚠️ Database: {err}")
            else:
                st.success(f"✅ {n_inserted} cleaned record(s) saved to SQLite database.")

        st.markdown("#### Cleaning Log")
        log_lines = []
        for entry in log:
            if "Removed" in entry or "error" in entry.lower():
                log_lines.append(f"🔴 {entry}")
            elif "Flagged" in entry or "Note" in entry:
                log_lines.append(f"🟡 {entry}")
            else:
                log_lines.append(f"🟢 {entry}")
        st.markdown("<div class='pw-log'>" + "<br>".join(log_lines) + "</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"#### Cleaned Data ({len(clean_df)} rows)")
        st.dataframe(clean_df.head(30), use_container_width=True)

        if not flagged_df.empty:
            with st.expander(f"⚠️ Flagged / Removed Rows ({len(flagged_df)})"):
                st.dataframe(flagged_df, use_container_width=True)
                st.download_button(
                    "⬇️ Download flagged rows (CSV)",
                    data=_bytes_csv(flagged_df),
                    file_name="powerwise_flagged_rows.csv",
                    mime="text/csv",
                )

        db_count = get_row_count()
        st.info(f"📦 SQLite database currently holds **{db_count}** record(s).")

    elif st.session_state.clean_df is not None:
        # Show previously cleaned data
        clean_df = st.session_state.clean_df
        log = st.session_state.clean_log
        st.success(f"✅ Cleaned data already available ({len(clean_df)} rows).")
        st.markdown("#### Cleaning log")
        log_lines = []
        for entry in log:
            if "Removed" in entry or "error" in entry.lower():
                log_lines.append(f"🔴 {entry}")
            elif "Flagged" in entry or "Note" in entry:
                log_lines.append(f"🟡 {entry}")
            else:
                log_lines.append(f"🟢 {entry}")
        st.markdown("<div class='pw-log'>" + "<br>".join(log_lines) + "</div>", unsafe_allow_html=True)

        flagged_df = st.session_state.get("flagged_df")
        if flagged_df is not None and not flagged_df.empty:
            with st.expander(f"⚠️ Flagged / Removed Rows ({len(flagged_df)})"):
                st.dataframe(flagged_df, use_container_width=True)
                st.download_button(
                    "⬇️ Download flagged rows (CSV)",
                    data=_bytes_csv(flagged_df),
                    file_name="powerwise_flagged_rows.csv",
                    mime="text/csv",
                )

        st.dataframe(clean_df.head(20), use_container_width=True)
        st.info(f"SQLite database currently holds **{get_row_count()}** record(s).")


# ===========================================================================
# PAGE 3 — Explore & Visualise
# ===========================================================================
elif page == "📊 Explore & Visualise":
    _hero("📊", "Explore & Visualise", "Understand daily patterns, monthly changes, and unusual consumption at a glance.")

    df = _require_clean_data()
    if df is None:
        st.stop()

    stats = descriptive_stats(df)

    # Compact descriptive summary
    st.markdown("### At a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", stats["count"])
    c2.metric("Mean (kWh/day)", f"{stats['mean']:.2f}")
    c3.metric("Median (kWh/day)", f"{stats['median']:.2f}")
    c4.metric("Total (kWh)", f"{stats['total']:.1f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Std Dev", f"{stats['std']:.2f}")
    c6.metric("Min (kWh)", f"{stats['min']:.2f}")
    c7.metric("Max (kWh)", f"{stats['max']:.2f}")
    c8.metric("Outliers flagged", int(df.get("is_outlier", pd.Series(False)).sum()))

    date_range = f"{df['date'].min().date()} → {df['date'].max().date()}"
    st.caption(f"Date range: {date_range}")

    st.markdown("---")

    # Daily consumption chart
    st.markdown("### Daily Consumption Trend")
    ma_window = st.slider("Moving average window (days)", 3, 30, 7)
    fig1 = plot_daily_consumption(df, ma_window=ma_window)
    st.pyplot(fig1)

    # Monthly bar chart
    monthly = compute_monthly(df)
    if len(monthly) >= 2:
        st.markdown("### Monthly Totals")
        fig2 = plot_monthly_bar(monthly)
        st.pyplot(fig2)

    # Histogram
    st.markdown("### Consumption Distribution")
    fig3 = plot_consumption_histogram(df)
    st.pyplot(fig3)

    # Rolling stats
    st.markdown("### Rolling Statistics")
    fig4 = plot_rolling_stats(df, window=ma_window)
    st.pyplot(fig4)

    # Heatmap (if enough data)
    fig5 = plot_monthly_heatmap(df)
    if fig5 is not None:
        st.markdown("### Avg Consumption by Month & Day of Week")
        st.pyplot(fig5)

    # Raw data table
    with st.expander("📋 View cleaned data table"):
        st.dataframe(df, use_container_width=True)


# ===========================================================================
# PAGE 4 — Bill Estimator
# ===========================================================================
elif page == "🧾 Bill Estimator":
    _hero("🧾", "Electricity Bill Estimator", "Estimate charges transparently with configurable demonstration tariff slabs.")

    df = _require_clean_data()
    if df is None:
        st.stop()

    st.info(
        "⚠️ **Manual tariff disclaimer**: The rates below are configurable examples for demonstration. "
        "They are not official utility rates. Weather data is available separately on the Weather Insights page."
    )

    st.markdown("### Tariff and billing period")
    col_t, col_b = st.columns([1, 1])

    with col_t:
        st.markdown("#### Tariff settings")
        rate1 = st.number_input("Rate — Slab 1 (0–100 kWh), ₹/kWh", value=3.00, step=0.25)
        rate2 = st.number_input("Rate — Slab 2 (101–300 kWh), ₹/kWh", value=5.00, step=0.25)
        rate3 = st.number_input("Rate — Slab 3 (>300 kWh), ₹/kWh", value=7.00, step=0.25)
        fixed = st.number_input("Fixed monthly charge (₹)", value=50.00, step=10.0)

    tariff = TariffConfig(rate1=rate1, rate2=rate2, rate3=rate3, fixed_charge=fixed)

    with col_b:
        st.markdown("#### Billing dates")
        min_date = df["date"].min().date()
        max_date = df["date"].max().date()
        start_date = st.date_input("Billing start date", value=min_date, min_value=min_date, max_value=max_date)
        end_date   = st.date_input("Billing end date",   value=max_date, min_value=min_date, max_value=max_date)

    if st.button("💰 Calculate Bill", type="primary", use_container_width=True):
        if start_date > end_date:
            st.error("Start date must be before end date.")
        else:
            result = estimate_bill_for_period(
                df,
                start_date=str(start_date),
                end_date=str(end_date),
                tariff=tariff,
            )
            if "error" in result:
                st.error(result["error"])
            else:
                st.markdown("### Bill Breakdown")
                st.markdown(
                    f"**Period**: {result['start_date']} to {result['end_date']}  "
                    f"({result['n_days']} day(s))"
                )

                b1, b2, b3 = st.columns(3)
                b1.metric("Total Consumption", f"{result['units_consumed']} kWh")
                b2.metric("Energy Charge", f"₹ {result['energy_charge']:.2f}")
                b3.metric("Fixed Charge", f"₹ {result['fixed_charge']:.2f}")

                st.markdown("#### Slab Breakdown")
                slab_data = {
                    "Slab": ["Slab 1 (0–100 kWh)", "Slab 2 (101–300 kWh)", "Slab 3 (>300 kWh)"],
                    "Units (kWh)": [result["slab1_units"], result["slab2_units"], result["slab3_units"]],
                    "Rate (₹/kWh)": [rate1, rate2, rate3],
                    "Charge (₹)": [result["slab1_charge"], result["slab2_charge"], result["slab3_charge"]],
                }
                st.table(pd.DataFrame(slab_data))

                st.markdown(
                    f"<div class='pw-total'><div class='pw-total-label'>Total estimated bill</div>"
                    f"<div class='pw-total-value'>₹ {result['total_bill']:.2f}</div></div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    "> **Formula**:  \n"
                    "> `Total Bill = Fixed Charge + (Slab1 units × Rate1) + (Slab2 units × Rate2) + (Slab3 units × Rate3)`"
                )


# ===========================================================================
# PAGE 5 — Forecast
# ===========================================================================
elif page == "🔮 Forecast":
    _hero("🔮", "Consumption Forecast", "Forecast household kWh locally from your uploaded history — no paid external forecast API required.")

    df = _require_clean_data()
    if df is None:
        st.stop()

    n_records = len(df)
    record_col, requirements_col = st.columns([1, 2])
    record_col.metric("Available records", n_records)
    requirements_col.markdown(
        "**Minimum records**  \n"
        f"Linear Regression: **{MIN_RECORDS_LINEAR}** · "
        f"ARIMA: **{MIN_RECORDS_ARIMA}**"
    )
    requirements_col.caption("ARIMA is offered only when the dataset meets its record requirement.")

    st.info("Choose the local baseline model or a weather-enhanced model. The weather-enhanced option trains on your consumption history matched with Open-Meteo historical weather, then uses the next 1–7 days of weather forecasts as model inputs. No API key is required.")

    st.markdown("### Forecast setup")

    # Method selection
    available_methods = ["Linear Regression (local)", "Weather-enhanced Linear Regression (Open-Meteo)"]
    if n_records >= MIN_RECORDS_ARIMA:
        available_methods.append("ARIMA (Statsmodels) — experimental")

    method_col, horizon_col = st.columns([1.5, 1])
    with method_col:
        method = st.radio("Forecasting method", available_methods)
    with horizon_col:
        if "Weather-enhanced" in method:
            days_ahead = st.slider("Weather-supported forecast horizon (days)", 1, 7, 7)
        else:
            days_ahead = st.slider("Days to forecast into the future", 7, 90, 30)

    weather_location = "Nagpur"
    if "Weather-enhanced" in method:
        weather_location = st.text_input("Weather location (city)", value="Nagpur", key="pw_forecast_weather_location")
        st.caption("Weather features: daily maximum/minimum temperature, precipitation, and maximum wind speed. Weather influence is learned from your historical consumption/weather matches.")

    if st.button("▶️ Run Forecast", type="primary", use_container_width=True):

        if "Weather-enhanced" in method:
            with st.spinner("Fetching historical weather and training the weather-enhanced model…"):
                try:
                    place_for_model = geocode_location(weather_location)
                    historical_weather = fetch_historical_weather(
                        place_for_model["latitude"], place_for_model["longitude"],
                        str(pd.to_datetime(df["date"]).min().date()),
                        str(pd.to_datetime(df["date"]).max().date()),
                    )
                    result, err = train_weather_linear_model(df, historical_weather)
                except WeatherAPIError as exc:
                    st.error(f"Weather API error: {exc}")
                    st.stop()
        if "Linear" in method or "Weather-enhanced" in method:
            if method == "Linear Regression (local)":
                with st.spinner("Training Linear Regression model…"):
                    result, err = train_linear_model(df)

            if err:
                st.error(f"❌ {err}")
                st.stop()

            st.success("✅ Model trained successfully.")

            # Evaluation section
            st.markdown("### Model Evaluation (held-out test set)")
            st.caption(
                f"Train: {result['n_train']} records | Test: {result['n_test']} records  \n"
                "The test set is the **last** portion of the data — no data leakage."
            )

            metrics = result["metrics"]
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE (kWh)", f"{metrics['MAE']:.3f}")
            m2.metric("RMSE (kWh)", f"{metrics['RMSE']:.3f}")
            if metrics["R²"] is not None:
                m3.metric("R²", f"{metrics['R²']:.3f}")
            else:
                m3.metric("R²", "N/A (< 5 test pts)")

            with st.expander("ℹ️ What do these metrics mean?"):
                st.markdown(
                    "- **MAE** (Mean Absolute Error): Average difference between predicted "
                    "and actual consumption. Lower is better.  \n"
                    "- **RMSE** (Root Mean Squared Error): Like MAE but penalises large "
                    "errors more. Lower is better.  \n"
                    "- **R²** (R-squared): Proportion of variance explained by the model. "
                    "Closer to 1.0 is better. Negative means the model is worse than predicting "
                    "the mean."
                )

            test_df = result["test_df"]
            fig_eval = plot_actual_vs_predicted(
                test_df["date"],
                test_df["consumption_kwh"],
                pd.Series(result["y_pred_test"]),
                title="Actual vs Predicted (Test Set)",
            )
            st.pyplot(fig_eval)

            # Future forecast
            st.markdown(f"### Future Forecast ({days_ahead} days)")
            if "Weather-enhanced" in method:
                try:
                    place_for_forecast = geocode_location(weather_location)
                    weather_payload = fetch_weather(place_for_forecast["latitude"], place_for_forecast["longitude"], days_ahead)
                    future_weather = daily_forecast_frame(weather_payload)
                    future_df = forecast_future_weather_linear(result, future_weather)
                    st.caption(f"Weather features fetched for {place_for_forecast.get('name', weather_location)}. Predictions use actual forecast weather inputs for each returned date.")
                except (WeatherAPIError, ValueError) as exc:
                    st.error(f"Could not create weather-enhanced forecast: {exc}")
                    st.stop()
            else:
                future_df = forecast_future_linear(result, days_ahead=days_ahead)
            fig_future = plot_future_forecast(df.tail(90), future_df[["date", "consumption_kwh"]])
            st.pyplot(fig_future)

            st.dataframe(future_df.rename(columns={"consumption_kwh": "predicted_kwh"}), use_container_width=True)

            if "Weather-enhanced" in method:
                st.warning("Weather-enhanced predictions use weather as model input, but weather does not determine electricity use by itself. Accuracy depends on how many historical consumption dates match the selected location’s weather. Forecast horizon is limited to the available 7-day weather forecast.")
            else:
                st.warning("Predictions assume future patterns match historical ones. Weather and other external factors are not included in this baseline model. Use forecasts for rough planning only.")

            # Download forecast
            st.download_button(
                "⬇️ Download Forecast (CSV)",
                data=_bytes_csv(future_df),
                file_name="powerwise_forecast.csv",
                mime="text/csv",
            )

        elif "ARIMA" in method:
            with st.spinner("Fitting ARIMA model (this may take a moment)…"):
                result, err = train_arima(df)

            if err:
                st.error(f"❌ {err}")
                st.stop()

            st.success("✅ ARIMA model fitted.")

            metrics = result["metrics"]
            m1, m2 = st.columns(2)
            m1.metric("MAE (kWh)", f"{metrics['MAE']:.3f}")
            m2.metric("RMSE (kWh)", f"{metrics['RMSE']:.3f}")

            fig_eval = plot_actual_vs_predicted(
                pd.Series(result["test_dates"]),
                pd.Series(result["y_test"]),
                pd.Series(result["y_pred"]),
                title="ARIMA — Actual vs Predicted (Test Set)",
            )
            st.pyplot(fig_eval)

            st.markdown(f"### Future Forecast ({days_ahead} days)")
            future_df = forecast_future_arima(result, days_ahead=days_ahead)
            fig_future = plot_future_forecast(
                df.tail(90)[["date", "consumption_kwh"]],
                future_df,
            )
            st.pyplot(fig_future)

            st.dataframe(future_df.rename(columns={"consumption_kwh": "predicted_kwh"}), use_container_width=True)

            st.warning(
                "⚠️ **ARIMA Limitations**: ARIMA(1,1,1) is a minimal specification.  "
                "For best results, inspect ACF/PACF plots and tune (p,d,q) parameters.  "
                "This is provided as an experimental feature."
            )

            st.download_button(
                "⬇️ Download Forecast (CSV)",
                data=_bytes_csv(future_df),
                file_name="powerwise_arima_forecast.csv",
                mime="text/csv",
            )


# ===========================================================================
# WEATHER INSIGHTS — Open-Meteo public weather API
# ===========================================================================
elif page == "🌦️ Weather Insights":
    _hero("🌦️", "Weather Insights", "Live conditions and a short-range forecast from the free Open-Meteo API.")
    st.caption("No API key required. Weather is also available as an input to the weather-enhanced household consumption forecast on the Forecast page.")

    location = st.text_input("City or location", value="Nagpur", key="pw_weather_location")
    forecast_days = st.slider("Forecast days", min_value=1, max_value=7, value=7, key="pw_weather_days")

    if st.button("🌤️ Get Weather", type="primary", use_container_width=True):
        try:
            with st.spinner("Finding location and fetching weather data…"):
                place = geocode_location(location)
                weather = fetch_weather(place["latitude"], place["longitude"], forecast_days)
            st.session_state["pw_weather_place"] = place
            st.session_state["pw_weather_data"] = weather
            st.success("Weather data loaded from Open-Meteo.")
        except WeatherAPIError as exc:
            st.error(str(exc))

    place = st.session_state.get("pw_weather_place")
    weather = st.session_state.get("pw_weather_data")
    if place and weather:
        current = weather["current"]
        st.markdown(f"### {place.get('name', location)}, {place.get('admin1', '')}, {place.get('country', '')}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Temperature", f"{current.get('temperature_2m', '—')} °C")
        c2.metric("Feels like", f"{current.get('apparent_temperature', '—')} °C")
        c3.metric("Humidity", f"{current.get('relative_humidity_2m', '—')}%")
        c4.metric("Wind speed", f"{current.get('wind_speed_10m', '—')} km/h")
        st.write(f"**Current conditions:** {weather_code_description(current.get('weather_code'))}")
        daily = weather.get("daily", {})
        rows = []
        for i, day in enumerate(daily.get("time", [])):
            code = daily.get("weather_code", [None] * len(daily.get("time", [])))[i]
            tmax = daily.get("temperature_2m_max", [None] * len(daily.get("time", [])))[i]
            tmin = daily.get("temperature_2m_min", [None] * len(daily.get("time", [])))[i]
            rain = daily.get("precipitation_sum", [None] * len(daily.get("time", [])))[i]
            wind = daily.get("wind_speed_10m_max", [None] * len(daily.get("time", [])))[i]
            rows.append({"Date": day, "Conditions": weather_code_description(code), "Min temp (°C)": tmin, "Max temp (°C)": tmax, "Precipitation (mm)": rain, "Max wind (km/h)": wind})
        st.markdown("### Daily forecast")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download Weather Forecast (CSV)", data=_bytes_csv(pd.DataFrame(rows)), file_name="powerwise_weather_forecast.csv", mime="text/csv")
        st.caption("Source: Open-Meteo. Forecast values can change; this is weather information, not a guarantee of future conditions.")


# ===========================================================================
# PAGE 7 — Export
# ===========================================================================
elif page == "💾 Export":
    _hero("💾", "Export & Reports", "Download cleaned readings, monthly summaries, and a report for your records.")

    df = _require_clean_data()
    if df is None:
        st.stop()

    date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
    st.markdown(f"**Available data**: {len(df)} cleaned records from {date_range}.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Download Cleaned Data")
        st.download_button(
            "⬇️ CSV (cleaned records)",
            data=_bytes_csv(df),
            file_name="powerwise_cleaned_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
        try:
            excel_bytes = _bytes_excel(df)
            st.download_button(
                "⬇️ Excel (cleaned records)",
                data=excel_bytes,
                file_name="powerwise_cleaned_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Excel export unavailable: {e}")

    with col2:
        st.markdown("#### Download Summary Report")
        monthly = compute_monthly(df)
        stats = descriptive_stats(df)

        report_lines = [
            "PowerWise — Electricity Consumption Summary Report",
            "=" * 50,
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
            f"Date range: {date_range}",
            f"Records: {stats['count']}",
            "",
            "Descriptive Statistics",
            "-" * 30,
            f"Mean daily consumption  : {stats['mean']:.2f} kWh",
            f"Median daily consumption: {stats['median']:.2f} kWh",
            f"Std deviation           : {stats['std']:.2f} kWh",
            f"Minimum                 : {stats['min']:.2f} kWh",
            f"Maximum                 : {stats['max']:.2f} kWh",
            f"Total consumption       : {stats['total']:.1f} kWh",
            "",
            "Monthly Totals",
            "-" * 30,
        ]
        for _, row in monthly.iterrows():
            report_lines.append(f"  {row['year_month_str']}: {row['consumption_kwh']:.2f} kWh")

        report_lines += [
            "",
            "NOTE: This report was generated from cleaned/processed data.",
            "If sample/synthetic data was used, it does NOT represent real utility data.",
        ]

        report_text = "\n".join(report_lines)
        st.download_button(
            "⬇️ Summary Report (TXT)",
            data=report_text.encode("utf-8"),
            file_name="powerwise_summary_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # Also offer monthly CSV
        monthly_export = monthly.drop(columns=["year_month"])
        st.download_button(
            "⬇️ Monthly Totals (CSV)",
            data=_bytes_csv(monthly_export),
            file_name="powerwise_monthly_totals.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("#### Database Records")
    db_df = load_all_readings()
    st.write(f"SQLite database holds **{len(db_df)}** record(s).")
    if not db_df.empty:
        with st.expander("View database records"):
            st.dataframe(db_df, use_container_width=True)

        st.download_button(
            "⬇️ Download All DB Records (CSV)",
            data=_bytes_csv(db_df),
            file_name="powerwise_database_export.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown("#### Clear saved records")
    st.warning("This permanently removes all readings from the SQLite database. This action cannot be undone.")
    if st.button("🗑️ Clear Database", type="secondary"):
        err = clear_readings()
        if err:
            st.error(err)
        else:
            st.session_state.clean_df = None
            st.success("Database cleared.  Import new data to start fresh.")


st.markdown(
    "<div class='pw-footer'>PowerWise · Electricity Consumption Analytics · Academic Project</div>",
    unsafe_allow_html=True,
)
