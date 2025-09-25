import pandas as pd
from pathlib import Path

RAW = Path("household_power_consumption.txt")  # UCI/Kaggle minute file (semicolon)
OUT = Path("data/hourly_power.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

usecols = ["Date", "Time", "Global_active_power"]
df = pd.read_csv(
    RAW, sep=";", usecols=usecols, na_values=["?", "nan", ""],
    parse_dates={"datetime": ["Date", "Time"]}, infer_datetime_format=True, low_memory=False
)
df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
df = df.set_index("datetime")

hourly = df["Global_active_power"].resample("1H").mean()
hourly = hourly.fillna(method="ffill", limit=3).dropna()

q_low, q_hi = hourly.quantile([0.001, 0.999])
hourly = hourly.clip(q_low, q_hi)

hourly.to_frame(name="Global_active_power").to_csv(OUT)
print(f"Saved {OUT} ({OUT.stat().st_size/1_048_576:.1f} MB)")
