from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data/processed/xauusd_features.csv"
MODEL_FILE = ROOT / "models/xauusd_direction_model.joblib"
TRADE_FILE = ROOT / "data/processed/backtest_trades.csv"
EQUITY_FILE = ROOT / "data/processed/backtest_equity.csv"

HORIZON = 4
THRESHOLD = 0.55
SPREAD_PRICE = 0.35
SLIPPAGE_PRICE_PER_SIDE = 0.05
COMMISSION_RATE_PER_SIDE = 0.0  # Unverified placeholder

INITIAL_EQUITY = 10_000.0
RISK_PER_TRADE = 0.0025
MAX_DAILY_LOSS = 0.01
STOP_ATR_MULTIPLE = 1.5
MAX_NOTIONAL_MULTIPLE = 1.0


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peaks = equity.cummax()
    return float((equity / peaks - 1.0).min())


def main():
    if not DATA_FILE.exists() or not MODEL_FILE.exists():
        raise FileNotFoundError("Missing features or saved model.")

    df = pd.read_csv(DATA_FILE)
    required = {
        "time", "open", "high", "low", "close", "atr_14_ratio"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    numeric = ["open", "high", "low", "close", "atr_14_ratio"]
    if df[numeric].isna().any().any():
        raise ValueError("Missing values found in price/ATR columns.")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Non-positive prices found.")

    artifact = joblib.load(MODEL_FILE)
    model = artifact["model"]
    features = artifact["features"]
    if not set(features).issubset(df.columns):
        raise ValueError("Model features are missing.")

    test = df.iloc[int(len(df) * 0.80):].copy().reset_index(drop=True)
    if len(test) <= HORIZON + 1:
        raise ValueError("Not enough test candles.")

    test["prob_up"] = model.predict_proba(test[features])[:, 1]
    test["signal"] = np.select(
        [
            test["prob_up"] >= THRESHOLD,
            test["prob_up"] <= 1 - THRESHOLD,
        ],
        [1, -1],
        default=0,
    )

    equity = INITIAL_EQUITY
    peak_equity = INITIAL_EQUITY
    current_day = None
    daily_start_equity = INITIAL_EQUITY
    blocked_day = False
    next_entry_index = 0
    position = None
    trades = []
    equity_points = []

    def mark_to_market(pos, bar):
        """Approximate liquidation equity using the assumed bid/ask model."""
        close = float(bar["close"])
        if pos["direction"] == 1:
            liquidation_price = close - SLIPPAGE_PRICE_PER_SIDE
            pnl = liquidation_price - pos["entry_price"]
        else:
            liquidation_price = (
                close + SPREAD_PRICE + SLIPPAGE_PRICE_PER_SIDE
            )
            pnl = pos["entry_price"] - liquidation_price

        gross_return = pnl / pos["entry_price"]
        net_return = gross_return - 2 * COMMISSION_RATE_PER_SIDE
        return pos["equity_before"] * (
            1 + pos["exposure_fraction"] * net_return
        )

    def close_position(pos, exit_index, exit_price, reason):
        nonlocal equity, peak_equity

        bar = test.loc[exit_index]
        direction = pos["direction"]

        if direction == 1:
            gross_pnl = exit_price - pos["entry_price"]
        else:
            gross_pnl = pos["entry_price"] - exit_price

        gross_return = gross_pnl / pos["entry_price"]
        commission_return = 2 * COMMISSION_RATE_PER_SIDE
        net_return = gross_return - commission_return

        equity_before = pos["equity_before"]
        equity_return = pos["exposure_fraction"] * net_return
        trade_pnl = equity_before * equity_return
        equity = equity_before + trade_pnl
        peak_equity = max(peak_equity, equity)

        trades.append({
            "signal_time": pos["signal_time"],
            "entry_time": pos["entry_time"],
            "exit_time": bar["time"],
            "direction": direction,
            "prob_up": pos["prob_up"],
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "stop_price": pos["stop_price"],
            "exit_reason": reason,
            "stopped": reason == "stop",
            "gross_price_return": gross_return,
            "commission_return_assumption": commission_return,
            "net_return_proxy": net_return,
            "exposure_fraction": pos["exposure_fraction"],
            "target_risk_fraction": RISK_PER_TRADE,
            "equity_before": equity_before,
            "trade_pnl_proxy": trade_pnl,
            "equity": equity,
        })

    for i in range(len(test)):
        bar = test.loc[i]
        day = bar["time"].date()

        if day != current_day:
            current_day = day
            daily_start_equity = equity
            blocked_day = False

        # Manage existing position before considering new entries.
        if position is not None:
            reason = None
            exit_price = None

            # Stop first. OHLC bars cannot tell us the exact intrabar order.
            if position["direction"] == 1:
                if float(bar["low"]) <= position["stop_price"]:
                    reason = "stop"
                    exit_price = (
                        position["stop_price"] - SLIPPAGE_PRICE_PER_SIDE
                    )
            else:
                ask_high = float(bar["high"]) + SPREAD_PRICE
                if ask_high >= position["stop_price"]:
                    reason = "stop"
                    exit_price = (
                        position["stop_price"] + SLIPPAGE_PRICE_PER_SIDE
                    )

            # If no stop was hit, apply the daily loss guard to marked equity.
            if reason is None:
                marked_equity = mark_to_market(position, bar)
                daily_loss_limit = daily_start_equity * (
                    1 - MAX_DAILY_LOSS
                )

                if marked_equity <= daily_loss_limit:
                    reason = "daily_loss"
                    if position["direction"] == 1:
                        exit_price = (
                            float(bar["close"]) - SLIPPAGE_PRICE_PER_SIDE
                        )
                    else:
                        exit_price = (
                            float(bar["close"])
                            + SPREAD_PRICE
                            + SLIPPAGE_PRICE_PER_SIDE
                        )
                    blocked_day = True

            # Time-based exit on the planned horizon.
            if reason is None and i >= position["planned_exit_index"]:
                reason = "time"
                if position["direction"] == 1:
                    exit_price = (
                        float(bar["close"]) - SLIPPAGE_PRICE_PER_SIDE
                    )
                else:
                    exit_price = (
                        float(bar["close"])
                        + SPREAD_PRICE
                        + SLIPPAGE_PRICE_PER_SIDE
                    )

            if reason is not None:
                close_position(position, i, exit_price, reason)
                position = None
                next_entry_index = i + 1

        # Record marked equity on every candle, not just trade exits.
        if position is not None:
            marked_equity = mark_to_market(position, bar)
        else:
            marked_equity = equity

        peak_equity = max(peak_equity, marked_equity)
        equity_points.append({
            "time": bar["time"],
            "realized_equity": equity,
            "marked_equity": marked_equity,
            "peak_marked_equity": peak_equity,
            "drawdown": marked_equity / peak_equity - 1,
        })

        if blocked_day or position is not None:
            continue
        if i < next_entry_index or i >= len(test) - HORIZON:
            continue

        direction = int(test.loc[i, "signal"])
        if direction == 0:
            continue

        entry_index = i + 1
        entry_bar = test.loc[entry_index]
        entry_open = float(entry_bar["open"])
        atr = float(test.loc[i, "atr_14_ratio"] * test.loc[i, "close"])

        if not np.isfinite(atr) or atr <= 0:
            continue

        stop_distance = STOP_ATR_MULTIPLE * atr

        if direction == 1:
            entry_price = (
                entry_open + SPREAD_PRICE + SLIPPAGE_PRICE_PER_SIDE
            )
            stop_price = entry_price - stop_distance
        else:
            entry_price = entry_open - SLIPPAGE_PRICE_PER_SIDE
            stop_price = entry_price + stop_distance

        if entry_price <= 0:
            continue

        stop_fraction = stop_distance / entry_price
        exposure_fraction = RISK_PER_TRADE / stop_fraction

        # Do not distort target risk by silently capping the position.
        if (
            not np.isfinite(exposure_fraction)
            or exposure_fraction <= 0
            or exposure_fraction > MAX_NOTIONAL_MULTIPLE
        ):
            continue

        position = {
            "signal_time": bar["time"],
            "entry_time": entry_bar["time"],
            "direction": direction,
            "prob_up": float(test.loc[i, "prob_up"]),
            "entry_price": entry_price,
            "stop_price": stop_price,
            "equity_before": equity,
            "exposure_fraction": exposure_fraction,
            "planned_exit_index": i + HORIZON,
        }
        next_entry_index = entry_index

    # If the final test candle arrives with a position still open, close at
    # the final available close rather than leaving equity unaccounted for.
    if position is not None:
        final_index = len(test) - 1
        final_bar = test.loc[final_index]
        if position["direction"] == 1:
            final_price = (
                float(final_bar["close"]) - SLIPPAGE_PRICE_PER_SIDE
            )
        else:
            final_price = (
                float(final_bar["close"])
                + SPREAD_PRICE
                + SLIPPAGE_PRICE_PER_SIDE
            )
        close_position(position, final_index, final_price, "end_of_test")

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_points)

    TRADE_FILE.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(TRADE_FILE, index=False)
    equity_df.to_csv(EQUITY_FILE, index=False)

    print("\n--- XAUUSD Research Backtest ---")
    print(f"Test candles: {len(test):,}")
    print(f"Trades: {len(trades_df):,}")
    print(f"Spread assumption: {SPREAD_PRICE:.3f}")
    print(f"Slippage per side: {SLIPPAGE_PRICE_PER_SIDE:.3f}")
    print(f"Commission per side: {COMMISSION_RATE_PER_SIDE:.5%}")
    print("Sizing remains a notional proxy, not broker lot sizing.")

    if not trades_df.empty:
        print(f"Win rate: {(trades_df['net_return_proxy'] > 0).mean():.2%}")
        print(f"Final equity proxy: {equity:.2f}")
        print(f"Max marked-to-market drawdown: {max_drawdown(equity_df['marked_equity']):.2%}")
        print(f"Stop exits: {(trades_df['exit_reason'] == 'stop').sum()}")
        print(f"Daily-loss exits: {(trades_df['exit_reason'] == 'daily_loss').sum()}")
    else:
        print("No trades passed the filters.")

    print(f"Trades saved: {TRADE_FILE}")
    print(f"Equity saved: {EQUITY_FILE}")
    print("Research only. No MT5 orders were placed.")


if __name__ == "__main__":
    main()
