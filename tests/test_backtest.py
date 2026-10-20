import pandas as pd

from xauusd_ai_bot.backtest import max_drawdown


def test_max_drawdown_with_loss():
    equity = pd.Series([10000, 10500, 10200, 9800, 10100])

    result = max_drawdown(equity)

    assert abs(result - (9800 / 10500 - 1)) < 1e-9


def test_max_drawdown_without_loss():
    equity = pd.Series([10000, 10100, 10200, 10300])

    assert max_drawdown(equity) == 0.0


def test_max_drawdown_empty_series():
    assert max_drawdown(pd.Series(dtype=float)) == 0.0
