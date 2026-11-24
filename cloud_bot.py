import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD"]

print(f"🤖 Uruchamianie bota w chmurze: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

for ticker in TICKERS:
    try:
        df = yf.download(ticker, period="2d", interval="1m", progress=False)
        if df.empty or len(df) < 50: continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()

        current_price = df["Close"].iloc[-1]
        current_ema20 = df["ema20"].iloc[-1]
        current_ema50 = df["ema50"].iloc[-1]
        
        current_signal = 1 if current_ema20 > current_ema50 else 0
        diff_pct = ((current_ema20 / current_ema50) - 1) * 100

        status = "W RYNKU (Trend wzrostowy)" if current_signal == 1 else "CZEKA (Trend spadkowy)"
        print(f"[{ticker}] {status} | Cena: ${current_price:,.2f} | Różnica EMA: {diff_pct:+.3f}%")

    except Exception as e:
        print(f"Błąd dla {ticker}: {e}")