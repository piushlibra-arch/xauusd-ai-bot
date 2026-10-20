from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "xauusd_m15_clean.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "xauusd_features.csv"

HORIZON = 4  # Four M15 candles = one hour


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"time", "open", "high", "low", "close", "tick_volume"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="raise")
    df = df.sort_values("time").reset_index(drop=True)

    if df["time"].duplicated().any():
        raise ValueError("Duplicate candle timestamps found.")

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Returns known at the close of each candle.
    df["return_1"] = close.pct_change(1)
    df["return_4"] = close.pct_change(4)
    df["return_16"] = close.pct_change(16)

    # Trend features.
    df["sma_10_ratio"] = close / close.rolling(10).mean() - 1
    df["sma_20_ratio"] = close / close.rolling(20).mean() - 1
    df["sma_50_ratio"] = close / close.rolling(50).mean() - 1

    # True range and ATR, calculated using current and past candles.
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr_14_ratio"] = (
        true_range.rolling(14).mean() / close
    )

    # Recent realized volatility.
    df["volatility_20"] = df["return_1"].rolling(20).std()

    # Candle shape.
    df["candle_range_ratio"] = (high - low) / close
    df["candle_body_ratio"] = (close - df["open"]) / close

    # Tick activity relative to its recent average.
    volume_mean = df["tick_volume"].rolling(20).mean()
    volume_std = df["tick_volume"].rolling(20).std()

    df["tick_volume_zscore"] = (
        (df["tick_volume"] - volume_mean)
        / volume_std.replace(0, np.nan)
    )

    # Target only: future information must never be a model input.
    future_close = close.shift(-HORIZON)
    df["future_return_4"] = future_close / close - 1
    df["target_up"] = np.where(
        future_close.notna(),
        (future_close > close).astype(int),
        np.nan,
    )

    feature_columns = [
        "return_1",
        "return_4",
        "return_16",
        "sma_10_ratio",
        "sma_20_ratio",
        "sma_50_ratio",
        "atr_14_ratio",
        "volatility_20",
        "candle_range_ratio",
        "candle_body_ratio",
        "tick_volume_zscore",
    ]

    df[feature_columns] = df[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    df = df.dropna(
        subset=feature_columns + ["future_return_4", "target_up"]
    ).reset_index(drop=True)

    return df


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Clean candle data not found: {INPUT_FILE}"
        )

    candles = pd.read_csv(INPUT_FILE)
    features = build_features(candles)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_FILE, index=False)

    print("\n--- Feature Engineering Results ---")
    print(f"Input candles: {len(candles):,}")
    print(f"Usable rows:   {len(features):,}")
    print(f"Features:      11")
    print(f"Target:        Direction over next {HORIZON} candles")
    print(f"Output:        {OUTPUT_FILE}")
    print("\nTarget distribution:")
    print(features["target_up"].value_counts(normalize=True).sort_index())
    print("\nLatest rows:")
    print(features.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
