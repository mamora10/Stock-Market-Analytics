import yfinance as yf
import pandas as pd
import numpy as np

def _flatten_columns(cols) -> list[str]:
    if not hasattr(cols, "to_list"):
        return list(cols)
    try:
        return [
            "_".join([str(x) for x in tup if str(x) not in ("", "None", "nan")])
            if isinstance(tup, tuple) else str(tup)
            for tup in cols.to_list()
        ]
    except Exception:
        return [str(c) for c in cols]

def _pick_first(columns: list[str], startswith: tuple[str, ...]) -> str | None:
    for s in startswith:
        cand = [c for c in columns if c.split("_", 1)[0] == s or c.startswith(s)]
        if cand:
            return cand[0]
    return None

def load_history(ticker: str, period: str = "1y") -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = _flatten_columns(df.columns)
        else:
            df.columns = [str(c) for c in df.columns]

        cols = list(df.columns)
        close_col = _pick_first(cols, ("Close", "Adj Close"))
        vol_col   = _pick_first(cols, ("Volume",))

        if close_col is None:
            raise ValueError(f"Could not find a Close/Adj Close column in: {cols}")
        if vol_col is None:
            df["Volume"] = np.nan
            vol_col = "Volume"

        out = pd.DataFrame(index=df.index)
        out["Close"]  = df[close_col].astype(float)
        out["Volume"] = df[vol_col].astype(float)

        for base in ("Open", "High", "Low"):
            cand = _pick_first(cols, (base,))
            if cand is not None:
                out[base] = df[cand].astype(float)

        out = out.dropna(how="any")
        return out
    except Exception as e:
        print("Error loading history:", e)
        return None

def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["Momentum_5"] = df["Close"].pct_change(5)
    df["Volatility_10"] = df["Return"].rolling(10).std()
    df["Volume_Change"] = df["Volume"].pct_change()
    df = df.dropna()
    return df
