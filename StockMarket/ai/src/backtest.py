import pandas as pd
import numpy as np

def backtest_signals(df_valid: pd.DataFrame, long_signal: pd.Series | list | np.ndarray):
    """Very naive long/flat backtest on validation slice.
    Assumes no fees/slippage, 100% cash to 100% long.
    """
    s = pd.Series(long_signal, index=df_valid.index[:len(long_signal)]).astype(int)
    s = s.reindex(df_valid.index).fillna(0)

    # Use next day's return when we're long
    ret = df_valid["Close"].pct_change().shift(-1)  # next day return aligned to today
    strategy_ret = ret * s
    equity = (1 + strategy_ret.fillna(0)).cumprod()
    out = pd.DataFrame({
        "signal": s,
        "equity_curve": equity
    })
    return out
