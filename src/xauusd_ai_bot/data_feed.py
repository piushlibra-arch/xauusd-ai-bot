from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "xauusd_m15.csv"
PROCESSED_FILE = (
    PROJECT_ROOT / "data" / "processed" / "xauusd_m15_clean.csv"
)

COLUMNS = [
    "time",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
]


def load_candles() -> pd.DataFrame:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"CSV not found: {RAW_FILE}")

    # Your MT5 export is UTF-16, comma-separated, without headers.
    df = pd.read_csv(
        RAW_FILE,
        encoding="utf-16",
        sep=",",
        header=None,
        names=COLUMNS,
    )

    df["time"] = pd.to_datetime(
        df["time"],
        format="%Y.%m.%d %H:%M",
        errors="coerce",
    )

    numeric_columns = COLUMNS[1:]
    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    original_rows = len(df)

    df = df.dropna(subset=COLUMNS)
    df = df.sort_values("time")
    df = df.drop_duplicates(subset="time", keep="last")

    valid = (
        (df[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (df["high"] >= df["low"])
        & (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
        & (df["tick_volume"] >= 0)
        & (df["spread"] >= 0)
    )

    df = df.loc[valid].reset_index(drop=True)

    if len(df) < 500:
        raise ValueError(
            f"Only {len(df)} valid candles found; "
            "at least 500 are required."
        )

    PROCESSED_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )
    df.to_csv(PROCESSED_FILE, index=False)

    print("\n--- XAUUSD M15 Data Validation ---")
    print(f"Rows read:       {original_rows:,}")
    print(f"Valid candles:   {len(df):,}")
    print(f"Rows removed:    {original_rows - len(df):,}")
    print(f"First timestamp: {df['time'].iloc[0]}")
    print(f"Last timestamp:  {df['time'].iloc[-1]}")
    print(f"Output:          {PROCESSED_FILE}")
    print("\nLatest 5 candles:")
    print(df.tail().to_string(index=False))

    return df


if __name__ == "__main__":
    load_candles()
