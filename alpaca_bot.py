import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import os
import warnings
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

warnings.filterwarnings("ignore")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

# Handlujemy na akcjach (SPY)
SYMBOL = "SPY"

def send_discord(msg, color=3447003):
    if not DISCORD_WEBHOOK_URL: return
    payload = {"username": "Daily Alpaca Bot", "embeds": [{"title": "🤖 Dzienny Raport", "description": msg, "color": color}]}
    requests.post(DISCORD_WEBHOOK_URL, json=payload)

print(f"Skan dzienny: {datetime.now()}")

account = trading_client.get_account()
# Pobieramy dane DZIENNE (interval="1d")
df = yf.download(SYMBOL, period="6mo", interval="1d", progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
    
df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()

current_price = df["Close"].iloc[-1]
current_signal = 1 if df["ema20"].iloc[-1] > df["ema50"].iloc[-1] else 0

positions = trading_client.get_all_positions()
has_position = any(p.symbol == SYMBOL for p in positions)

# Logika
if current_signal == 1 and not has_position:
    qty = int(float(account.cash) * 0.2 / current_price) # 20% kapitału na pozycję
    if qty > 0:
        order_data = MarketOrderRequest(symbol=SYMBOL, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
        trading_client.submit_order(order_data)
        send_discord(f"🟢 **KUPNO dzienne**\nKupiono {qty} szt. {SYMBOL}.", color=3066993)
        
elif current_signal == 0 and has_position:
    trading_client.close_position(SYMBOL)
    send_discord(f"🔴 **SPRZEDAŻ dzienna**\nZamknięto {SYMBOL}.", color=15158332)
else:
    print("Brak transakcji. Czekamy na sygnał.")