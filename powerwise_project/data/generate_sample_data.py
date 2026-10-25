from pathlib import Path
import numpy as np
import pandas as pd


def main():
    output = Path(__file__).with_name("sample_electricity_data.csv")
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=731, freq="D")
    day = np.arange(len(dates))
    weekly = 0.7 * np.sin(2 * np.pi * dates.dayofweek / 7)
    annual = 1.3 * np.sin(2 * np.pi * day / 365.25)
    trend = 0.0015 * day
    consumption = np.maximum(0.5, 8.5 + weekly + annual + trend + rng.normal(0, 0.9, len(dates)))
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "consumption_kwh": consumption.round(2), "meter_name": "Sample Meter"})
    df.to_csv(output, index=False)
    print(f"Created {output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
