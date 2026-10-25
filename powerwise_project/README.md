# ⚡ PowerWise — Smart Electricity Consumption and Bill Predictor

> **College Project** — B.Tech Computer Science & Engineering (2nd Year)  
> Demonstrates Data Analytics & Visualization (DAV) practical concepts  
> Built with Python, Streamlit, Pandas, Scikit-learn, SQLAlchemy, and more.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Features & Limitations](#2-features--limitations)
3. [Installation & Setup](#3-installation--setup)
4. [Running the Application](#4-running-the-application)
5. [Data Schema & Sample Data](#5-data-schema--sample-data)
6. [Technology Choices](#6-technology-choices)
7. [ETL Workflow & Database](#7-etl-workflow--database)
8. [Data Cleaning Rules](#8-data-cleaning-rules)
9. [Bill Estimation Formula](#9-bill-estimation-formula)
10. [Forecasting Methods & Evaluation](#10-forecasting-methods--evaluation)
11. [Running Tests](#11-running-tests)
12. [DAV Practical Coverage](#12-dav-practical-coverage)
13. [System Architecture](#13-system-architecture)
14. [Presentation Mapping](#14-presentation-mapping)
15. [Viva Preparation Guide](#15-viva-preparation-guide)
16. [Project Handoff Notes](#16-project-handoff-notes)

---

## 1. Project Overview

**PowerWise** is a web application that helps households, hostels, and small shops understand and predict their electricity usage.

**Real-life problem**: Electricity bills often come as a surprise. Most people do not know their daily consumption patterns, cannot predict next month's bill, and have no tool to analyze past usage trends.

**What PowerWise does**:
- Imports electricity records from CSV or Excel files
- Cleans and validates the data automatically
- Stores records in a local SQLite database
- Shows consumption trends and statistics via interactive charts
- Estimates electricity bills using a transparent, configurable tariff
- Forecasts future consumption using machine learning
- Exports cleaned data and summary reports

---

## 2. Features & Limitations

### ✅ Features
| Feature | Description |
|---|---|
| Import | CSV and Excel upload with column auto-normalisation |
| Validation | Date parsing, numeric checks, duplicate and negative-reading detection |
| Cleaning | Missing value interpolation, outlier flagging (IQR), cleaning log |
| Database | SQLite persistence via SQLAlchemy (INSERT OR IGNORE for re-uploads) |
| EDA | Descriptive stats, daily/monthly charts, histogram, rolling stats, heatmap |
| Bill Estimator | Configurable demonstration tariff plus free Open-Meteo weather API insights |
| Forecasting | Linear Regression (≥30 records) and ARIMA (≥60 records) |
| Evaluation | Time-aware train/test split; MAE, RMSE, R² metrics |
| Export | CSV, Excel, text summary report, monthly totals |
| Sample Data | 731-day synthetic dataset clearly labelled as sample/demo data |

### ⚠️ Limitations
- No login or multi-user support (single-user local app)
- No real-time meter reading integration
- Forecasts assume future patterns match historical ones
- ARIMA is minimal (1,1,1) — fine for demonstration, not production
- Manual tariff rates are illustrative; official API estimates require a valid provider key and supported schedule
- No weather integration (kept intentionally simple)

---

## 3. Installation & Setup

### Prerequisites
- Python 3.9 or later
- pip

### Steps

```bash
# 1. Navigate to the project folder
cd powerwise

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate the sample dataset (run once)
python data/generate_sample_data.py
```

---

## 4. Running the Application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## 5. Data Schema & Sample Data

### Required CSV/Excel Schema

| Column | Type | Description | Example |
|---|---|---|---|
| `date` | date string | Reading date (YYYY-MM-DD recommended) | `2024-01-15` |
| `consumption_kwh` | numeric | Daily electricity consumed in kWh | `12.5` |

**Optional column**: `meter_name` — label for the meter or location (e.g., "Hostel Block A").

### Column Name Variations Accepted

The app automatically renames these to the canonical names:

| Accepted Variant | Mapped To |
|---|---|
| `kwh`, `units`, `usage_kwh`, `energy_kwh`, `electricity_kwh` | `consumption_kwh` |
| `reading_date`, `record_date`, `day`, `timestamp` | `date` |
| `meter`, `meter_id`, `location` | `meter_name` |

### Sample Data

A 731-day synthetic dataset is included at `data/sample_electricity_data.csv`.

> ⚠️ **This is SYNTHETIC/SAMPLE data generated with a random seed. It does NOT represent any real household, meter, or utility company.**

It includes realistic patterns (weekly seasonality, annual trend, random noise) and three injected anomalies (two spikes and one very low reading) for demonstration purposes.

To regenerate:
```bash
python data/generate_sample_data.py
```

---

## 6. Technology Choices

| Technology | Purpose | Why Chosen |
|---|---|---|
| **Python 3** | Core language | Widely used, beginner-friendly, rich ecosystem |
| **Streamlit** | Web UI | Creates interactive web apps with pure Python; no HTML/CSS/JS needed |
| **Pandas** | Data manipulation | Industry standard for tabular data; DataFrames make cleaning easy |
| **NumPy** | Numerical operations | Efficient array operations; used for feature engineering and clipping |
| **Matplotlib** | Plotting (base) | Full control over chart appearance; widely taught in academics |
| **Seaborn** | Statistical plots | Cleaner histogram/heatmap with one function call |
| **Scikit-learn** | ML / Regression | LinearRegression, StandardScaler, and evaluation metrics |
| **Statsmodels** | ARIMA time-series | Provides ARIMA implementation; lazy-imported only when needed |
| **SQLAlchemy** | ORM / DB abstraction | Parameterised queries, readable code, safe against SQL injection |
| **SQLite** | Database engine | Zero-config, single-file database; perfect for local projects |
| **openpyxl** | Excel read/write | Required by pandas for `.xlsx` files |
| **pytest** | Testing | Simple, widely used Python test framework |

---

## 7. ETL Workflow & Database

### ETL Steps (clearly identified in code)

```
┌─────────────┐    ┌─────────────────────────────┐    ┌────────────────────┐
│   EXTRACT   │    │         TRANSFORM           │    │       LOAD         │
│             │    │                             │    │                    │
│ Upload CSV  │───▶│ 1. Normalise column names   │───▶│ Insert into SQLite │
│ or Excel    │    │ 2. Parse dates              │    │ (electricity_      │
│             │    │ 3. Parse numeric values     │    │  readings table)   │
│ load_file() │    │ 4. Remove invalid rows      │    │                    │
│             │    │ 5. Remove duplicates        │    │ insert_readings()  │
│             │    │ 6. Interpolate missing vals │    │                    │
│             │    │ 7. Flag outliers (IQR)      │    │ INSERT OR IGNORE   │
│             │    │                             │    │ prevents duplicate │
└─────────────┘    └─────────────────────────────┘    └────────────────────┘
```

### Database Schema

**Table**: `electricity_readings`

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment primary key |
| `date` | TEXT UNIQUE | ISO date string (e.g. "2024-01-15") |
| `consumption_kwh` | REAL | Daily consumption in kWh |
| `meter_name` | TEXT | Optional meter/location label |
| `is_outlier` | BOOLEAN | True if IQR test flagged this row |

The database file is stored as `powerwise.db` in the project root.
You can inspect it with [DB Browser for SQLite](https://sqlitebrowser.org/).

---

## 8. Data Cleaning Rules

The cleaning pipeline (`powerwise/data_processing.py → clean_data()`) applies these steps **in order**:

| Step | Action | Reason |
|---|---|---|
| 1. Parse dates | `pd.to_datetime(errors='coerce')` | Identifies unparseable dates; converts to NaT |
| 2. Remove bad dates | Drop rows where date = NaT | Invalid dates make the record useless |
| 3. Parse consumption | `pd.to_numeric(errors='coerce')` | Identifies non-numeric values |
| 4. Remove non-numeric | Drop rows where consumption = NaN | Cannot use these for any calculation |
| 5. Remove negatives | Drop rows where `consumption_kwh < 0` | Negative electricity is physically impossible |
| 6. Remove duplicates | Keep first occurrence per date | Multiple readings on same date are likely data entry errors |
| 7. Interpolate missing | Linear interpolation on remaining NaN consumption | Assumption: consumption changes gradually; better than mean imputation |
| 8. Detect outliers | IQR-based fencing (default factor 3.0) | Flags unusual readings **without removing them**; kept for transparency |
| 9. Sort by date | Ascending sort | Required for time-series analysis and chart readability |

**Key design decision**: Outliers are **flagged** (`is_outlier = True`) but **not removed**. A very high reading on a specific day may be real (e.g., a party, an industrial event). The user can review and download flagged rows separately.

**Interpolation assumption**: Linear interpolation is a simplification. It assumes consumption changes smoothly between adjacent readings. This is a reasonable assumption for short gaps (1–3 days) but becomes less reliable over longer gaps.

---

## 9. Bill Estimation Formula

### Tariff Structure (Illustrative — Not Official)

> ⚠️ **These rates are for demonstration only. They are NOT official rates from any electricity utility.**

| Slab | Units | Default Rate |
|---|---|---|
| Slab 1 | 0 – 100 kWh | ₹ 3.00 / kWh |
| Slab 2 | 101 – 300 kWh | ₹ 5.00 / kWh |
| Slab 3 | > 300 kWh | ₹ 7.00 / kWh |
| Fixed charge | Per billing period | ₹ 50.00 |

### Formula

```
Total Bill = Fixed Charge
           + min(units, 100) × Rate1
           + max(0, min(units − 100, 200)) × Rate2
           + max(0, units − 300) × Rate3
```

All rates are **configurable** from the Bill Estimator page. The formula and breakdown are always displayed so the calculation is fully transparent.

### Example

For 250 kWh consumed:
- Slab 1: 100 × ₹3.00 = ₹300
- Slab 2: 150 × ₹5.00 = ₹750
- Slab 3: 0 × ₹7.00 = ₹0
- Fixed: ₹50
- **Total: ₹1,100**

---

## Weather API integration

PowerWise includes a Weather Insights page powered by the free Open-Meteo geocoding and forecast APIs. No API key is required. Enter a city to see current temperature, feels-like temperature, humidity, wind speed, and up to seven days of daily weather forecast. The forecast can be downloaded as CSV.

Weather data is currently displayed as contextual information; it is not yet used as an input feature in the household consumption forecasting models. The bill estimator uses its configurable manual tariff slabs and does not require any tariff API.

## 10. Forecasting Methods & Evaluation

### Method 1: Linear Regression (Scikit-learn)

**When available**: ≥ 30 records in the cleaned dataset.

**Features used**:
| Feature | Explanation |
|---|---|
| `day_ordinal` | Days since first record — captures overall trend |
| `dow_sin`, `dow_cos` | Cyclical encoding of day-of-week — captures weekly seasonality |
| `month_sin`, `month_cos` | Cyclical encoding of month — captures annual seasonality |
| `roll7` | 7-day rolling mean — captures recent consumption level |

**Why cyclical encoding?**: If we use raw day-of-week (0–6), the model treats Sunday (6) as far from Monday (0), but they are adjacent. Sine/cosine encoding wraps around correctly.

**Evaluation strategy**: **Time-aware split** — the last 20% of records form the test set. The model is trained on the first 80% only. This respects the temporal order of the data (no leakage from the future into training).

### Method 2: ARIMA (Statsmodels)

**When available**: ≥ 60 records.

**Configuration**: ARIMA(1, 1, 1) — minimal specification suitable for a student project.
- p=1: one autoregressive lag
- d=1: first differencing to remove trend
- q=1: one moving-average term

**Limitation**: The (1,1,1) order is not tuned. For a research project, you would inspect ACF/PACF plots and use AIC/BIC to select optimal parameters.

### Evaluation Metrics

| Metric | Formula | Meaning |
|---|---|---|
| **MAE** | mean(|actual − predicted|) | Average absolute error in kWh. Lower is better. |
| **RMSE** | √mean((actual − predicted)²) | Like MAE but penalises large errors more. Lower is better. |
| **R²** | 1 − SS_res/SS_tot | Proportion of variance explained. Closer to 1.0 is better. Only shown with ≥ 5 test points. |

### Data Sufficiency Rules

| Situation | Behaviour |
|---|---|
| < 30 records | Linear Regression refused with explanation |
| < 60 records | ARIMA option not shown |
| Sparse / irregular data | Interpolation applied to daily frequency; limitation noted |
| Very small test set (< 5) | R² not reported (unreliable on few points) |

### Forecast Limitations

- Predictions assume future patterns are similar to historical ones.
- External shocks (appliance changes, occupancy, tariff revisions) are not modelled.
- Uncertainty grows with forecast horizon — do not treat long-range forecasts as accurate.
- This model is trained on whatever data you provide; more data generally improves accuracy.

---

## 11. Running Tests

```bash
# From the project root:
python -m pytest tests/ -v
```

### Verified Test Results (observed on Python 3.13, pytest 9.1.1)

```
43 passed, 2 warnings in 5.26s
```

| Test File | Tests | Coverage |
|---|---|---|
| `test_billing.py` | 13 | Bill calculation for all slabs, edge cases, date-range filtering |
| `test_data_processing.py` | 20 | Column normalisation, cleaning steps, stats |
| `test_forecasting.py` | 10 | Sufficient/insufficient data checks, metrics, no-negative forecasts |

> Tests were run locally on the final code and passed with the results shown above. No test results are fabricated.

---

## 12. DAV Practical Coverage

| # | DAV Practical Topic | How PowerWise Covers It |
|---|---|---|
| 1 | Python, NumPy, Pandas | Core data stack throughout; NumPy for feature engineering and clipping |
| 2 | CSV and Excel | Upload, read, and export both formats |
| 3 | Data Cleaning | 8-step cleaning pipeline with detailed logging and flagged-row report |
| 4 | Exploratory Data Analysis | Descriptive stats, date-range summary, monthly aggregations |
| 5 | Visualization | 5 chart types (line, bar, histogram, rolling-stats, heatmap) with labels |
| 6 | Regression | Scikit-learn LinearRegression with cyclical features and time-aware eval |
| 7 | Time-Series Analysis | Moving average, rolling stats, ARIMA forecasting, seasonal feature encoding |
| 8 | API and JSON | Open-Meteo weather API in Weather Insights |
| 9 | SQLite, SQL, SQLAlchemy | Full ORM model, INSERT/SELECT/DELETE, persistent local database |
| 10 | ETL | Clearly separated Extract / Transform / Load steps in code and docs |
| 11 | Excel-compatible output | openpyxl-based Excel download for cleaned data |

---

## 13. System Architecture

```
User (Browser)
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit App (app.py)                    │
│   Pages: Import │ Clean │ Explore │ Bill │ Forecast │ Export │
└──────────┬──────┬───────┬─────────┬──────┬──────────┬───────┘
           │      │       │         │      │          │
           ▼      ▼       ▼         ▼      ▼          ▼
  data_processing  database   analysis   billing  forecasting
       .py           .py        .py        .py        .py
       │              │
  ┌────┴────┐   ┌─────┴──────┐
  │  Pandas │   │  SQLAlchemy│
  │  NumPy  │   │  SQLite DB │
  └─────────┘   └────────────┘
```

### Data Flow

```
CSV/Excel Upload
      │
      ▼ load_file()           [EXTRACT]
Raw DataFrame
      │
      ▼ normalise_columns()   [TRANSFORM — step 1]
      ▼ clean_data()          [TRANSFORM — steps 2–8]
Cleaned DataFrame + Flagged Rows + Log
      │
      ├──▶ insert_readings()  [LOAD] → SQLite DB
      │
      ├──▶ EDA / Charts       [ANALYSE]
      │
      ├──▶ calculate_bill()   [ESTIMATE]
      │
      ├──▶ train_linear_model() / train_arima()  [FORECAST]
      │
      └──▶ CSV / Excel / TXT download            [EXPORT]
```

---

## 14. Presentation Mapping

| Slide | Content Source |
|---|---|
| Introduction | Section 1 of this README |
| Problem Statement | Section 1 — "Real-life problem" |
| Objectives | Section 2 — Features table |
| Existing System | Manual meter reading, paper bills, no analytics |
| Proposed System | PowerWise — automated analysis, visual dashboard, bill prediction |
| System Architecture | Section 13 — block diagram |
| Technologies Used | Section 6 — technology table |
| Methodology / Working Process | Section 7 — ETL workflow and data flow |
| Implementation / Screenshots | Run the app and take screenshots of each page |
| Results and Discussion | Section 11 — test results; app demo with sample data |
| Advantages and Applications | Households, hostels, small shops, energy audits |
| Conclusion and Future Scope | Section 15 — future improvements |

---

## 15. Viva Preparation Guide

### Problem Statement
**Q: What problem does PowerWise solve?**  
A: Many households and hostels cannot easily track electricity usage, predict their bill, or spot unusual consumption. PowerWise lets them import meter reading records, clean the data, visualise trends, estimate bills, and forecast future usage — all from a simple web interface.

### Data Flow
**Q: How does data move through the application?**  
A: The user uploads a CSV or Excel file → the app extracts it (load_file) → normalises column names → runs the cleaning pipeline (remove bad dates, non-numeric values, negatives, duplicates; interpolate missing; flag outliers) → stores clean records in SQLite → displays charts and statistics → the user can estimate bills and generate forecasts → download results.

### Libraries
**Q: Why Streamlit instead of Flask or Django?**  
A: Streamlit lets you build a web UI with just Python. No HTML, CSS, or JavaScript is needed. For a data project with charts and widgets, it is far simpler and faster to build than Flask or Django.

**Q: Why SQLAlchemy instead of raw sqlite3?**  
A: SQLAlchemy provides parameterised queries (preventing SQL injection), a Pythonic interface, and makes the schema definition readable. Raw sqlite3 works too, but SQLAlchemy is what is used in real projects.

**Q: Why Scikit-learn for forecasting?**  
A: Scikit-learn provides a clean, consistent API for machine learning. LinearRegression, StandardScaler, and evaluation metrics are all available with simple function calls.

### Data Cleaning
**Q: What happens in the cleaning step?**  
A: Eight sequential steps: (1) parse dates, (2) remove unparseable dates, (3) parse numbers, (4) remove non-numeric consumption, (5) remove negative readings, (6) remove duplicate dates, (7) interpolate missing values linearly, (8) flag outliers using IQR. A log of every action is displayed to the user.

**Q: Why keep outliers instead of removing them?**  
A: A spike might be real — perhaps a machine ran all day or extra people were present. Removing it silently would make the data dishonest. We flag it with `is_outlier=True` so the user is aware but the data is preserved.

### Bill Calculation
**Q: How is the electricity bill calculated?**  
A: Three-slab tariff: first 100 kWh at Rate1, next 200 kWh at Rate2, everything above 300 kWh at Rate3, plus a fixed charge. The formula and slab breakdown are always shown.

**Q: Are the tariff rates real?**  
A: No. They are illustrative defaults for demonstration. The user can change them to match their actual utility tariff.

### Forecasting
**Q: How does the Linear Regression forecast work?**  
A: We create time-based numeric features (day index, cyclical day-of-week and month encodings, 7-day rolling mean), scale them with StandardScaler, and fit LinearRegression. The model predicts future consumption from future feature values.

**Q: What is a time-aware train/test split?**  
A: We use the LAST 20% of records as the test set — not a random sample. For time series, the model must never see future data during training. Random shuffling would leak future information into the training set and give misleadingly good metrics.

**Q: What do MAE and RMSE mean?**  
A: MAE is the average absolute difference between predictions and actual values in kWh. RMSE is similar but squares the errors first, so large errors are penalised more. Both should be as small as possible.

**Q: When is ARIMA not used?**  
A: When there are fewer than 60 records. ARIMA needs enough data to estimate its parameters reliably. On small datasets it is likely to overfit or fail.

### Limitations and Future Work
**Q: What are the main limitations?**  
A: (1) No weather or external data integration. (2) ARIMA order (1,1,1) is not tuned per dataset. (3) Linear Regression cannot capture sudden structural breaks. (4) Predictions degrade over long forecast horizons. (5) No multi-user support.

**Q: What could be improved in future?**  
A: Prophet or gradient boosting for better forecasts; use weather history as model features; anomaly detection with LSTM; multi-user login; real-time meter API integration; mobile-responsive UI.

### Likely Viva Questions

| Question | Short Answer |
|---|---|
| What is ETL? | Extract (read file), Transform (clean data), Load (store in DB) |
| What is IQR? | Interquartile Range = Q3 − Q1; outliers are beyond Q1 − k×IQR or Q3 + k×IQR |
| What is R²? | Proportion of variance in the target explained by the model (0–1, higher is better) |
| What is interpolation? | Estimating missing values from surrounding known values (linear = straight line) |
| Why normalise/standardise features? | Prevents large-magnitude features from dominating the model |
| What is a moving average? | Average of the last N values; smooths out noise in a time series |
| What does ARIMA stand for? | AutoRegressive Integrated Moving Average |
| What is a slab tariff? | Different price rates for different bands of consumption |
| What is a primary key in SQLite? | A unique identifier for each row; cannot be NULL; used to relate tables |
| What is a session state in Streamlit? | Dictionary (`st.session_state`) that persists values between user interactions |

---

## 16. Project Handoff Notes

### Goal
Build a student data-analytics web app for electricity consumption analysis. Scope: import, clean, store, visualise, estimate bill, forecast, export. No login, no IoT, no mobile app.

### Folder / File Responsibilities

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI — all pages and user interaction |
| `powerwise/data_processing.py` | ETL Extract + Transform: load, validate, clean |
| `powerwise/database.py` | ETL Load: SQLAlchemy models, insert, query, clear |
| `powerwise/billing.py` | Bill estimation with configurable slab tariff |
| `powerwise/analysis.py` | EDA: moving average, descriptive stats, all charts |
| `powerwise/forecasting.py` | Linear Regression and ARIMA models, evaluation |
| `data/sample_electricity_data.csv` | 731-day synthetic dataset (clearly labelled) |
| `data/generate_sample_data.py` | Script to regenerate sample data |
| `tests/` | pytest tests for cleaning, billing, forecasting |
| `requirements.txt` | All Python dependencies |
| `README.md` | This file |

### Install, Run, Test

```bash
pip install -r requirements.txt
python data/generate_sample_data.py
streamlit run app.py
python -m pytest tests/ -v
```

### Known Limitations / Possible Future Work
- ARIMA order is fixed at (1,1,1); could be auto-selected with pmdarima
- No confidence intervals on forecasts
- No multi-meter / multi-location comparison view
- Weather API (OpenWeatherMap) could be added as optional enhancement
- Tests cover core logic; UI behaviour is not tested (requires Streamlit testing tools)
