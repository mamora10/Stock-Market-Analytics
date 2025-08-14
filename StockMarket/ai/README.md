# AI + Stocks: Streamlit MVP

This is a beginner-friendly app that:
1) Fetches historical stock data (via `yfinance` — no API key needed).
2) Shows an interactive chart and basic indicators.
3) Trains a quick baseline ML model (logistic regression) to predict next-day up/down using simple features.
4) Backtests the naive strategy and reports accuracy and cumulative returns (toy example — **not financial advice**).

## Quickstart

```bash
# 1) Create a fresh virtual env (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## Project layout
```
ai_stock_app/
├─ app.py                 # Streamlit UI
├─ requirements.txt
├─ src/
│  ├─ data.py            # data loading & indicators
│  ├─ model.py           # feature engineering & training
│  └─ backtest.py        # super-simple backtest
└─ README.md
```

## Notes & roadmap
- This is a **teaching** scaffold: tiny features, tiny model, no fancy hyperparams.
- Next steps (suggested):
  - Add more engineered features (RSI, MACD, Bollinger Bands, intraday volatility).
  - Try different targets (next N-day return, probability calibration).
  - Use walk-forward validation instead of a single split.
  - Risk controls (position sizing, stop losses).
  - Add a paper-trading broker (Alpaca, TDA, IBKR — check their docs & compliance).
  - Consider a small vector DB for news embeddings + sentiment (e.g., `sentence-transformers`).

**Disclaimer:** Educational only. Past performance does not guarantee future results.
