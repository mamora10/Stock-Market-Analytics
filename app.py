import streamlit as st
from src.data import load_history, add_basic_indicators
from src.model import make_features, train_baseline
from src.backtest import backtest_signals
import pandas as pd

st.set_page_config(page_title="AI + Stocks (MVP)", layout="wide")

st.title("AI + Stocks: Educational MVP")
st.caption("Toy example. Not investment advice.")

col1, col2, col3 = st.columns([2,1,1])
ticker = col1.text_input("Ticker", value="AAPL")
period = col2.selectbox("History", options=["1y","2y","5y","10y"], index=0)
train_ratio = col3.slider("Train split", min_value=0.5, max_value=0.95, value=0.8, step=0.05)

if st.button("Run"):
    with st.spinner("Loading data..."):
        df = load_history(ticker, period=period)
        if df is None or df.empty:
            st.error("No data returned. Check ticker or try another period.")
            st.stop()
        df = add_basic_indicators(df)

    st.subheader("Price & Indicators")
    st.line_chart(df[["Close","SMA_10","SMA_20"]])

    with st.spinner("Engineering features & training model..."):
        X, y, feature_cols = make_features(df)
        split_idx = int(len(X) * train_ratio)
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        X_test,  y_test  = X.iloc[split_idx:], y.iloc[split_idx:]
        model, metrics, y_prob = train_baseline(X_train, y_train, X_test, y_test)

    st.subheader("Validation Metrics")
    st.write(metrics)

    with st.spinner("Backtesting naive long/flat strategy..."):
        # Simple strategy: go long when prob_up>0.5, else flat
        preds = (y_prob > 0.5).astype(int)
        bt = backtest_signals(df.iloc[split_idx:], preds)
    st.subheader("Backtest (validation period only)")
    st.line_chart(bt[["equity_curve"]])
    st.write(bt.tail(10))

st.sidebar.header("How it works")
st.sidebar.markdown("""
- **Data:** Daily OHLCV from `yfinance`.
- **Features:** % returns, MAs, momentum, volatility.
- **Model:** Logistic Regression (sklearn) → probability next day is up.
- **Backtest:** Long if p>0.5, otherwise flat (no leverage, no fees).
- **Goal:** Learning template; not a trading system.
""")
