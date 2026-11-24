import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import os
import warnings

warnings.filterwarnings("ignore")

TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD"]
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

def send_discord_message(embed):
    if not DISCORD_WEBHOOK_URL:
        print("Brak URL Webhooka Discord!")
        return
        
    payload = {
        "username": "Quant Trading Bot",
        "avatar_url": "https://cdn-icons-png.flaticon.com/512/2933/2933245.png",
        "embeds": [embed]
    }
    requests.post(DISCORD_WEBHOOK_URL, json=payload)

print(f"🤖 Uruchamianie bota w chmurze: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Budujemy ładną wiadomość dla Discorda (Embed)
fields = []
for ticker in TICKERS:
    try:
        df = yf.download(ticker, period="2d", interval="1m", progress=False)
        if df.empty or len(df) < 50: continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()

        price = df["Close"].iloc[-1]
        diff_pct = ((df["ema20"].iloc[-1] / df["ema50"].iloc[-1]) - 1) * 100
        
        status = "🟢 TREND WZROSTOWY" if diff_pct > 0 else "🔴 TREND SPADKOWY"
        
        fields.append({
            "name": f"{ticker} ({status})",
            "value": f"Cena: **${price:,.2f}**\nRóżnica EMA: `{diff_pct:+.3f}%`",
            "inline": True
        })

    except Exception as e:
        print(f"Błąd dla {ticker}: {e}")

# Główna treść wiadomości
embed = {
    "title": "📊 Raport Rynkowy",
    "description": f"Automatyczny skan wykonany: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC",
    "color": 3447003, # Niebieski kolor paska z boku
    "fields": fields,
    "footer": {
        "text": "System Trend-Following EMA 20/50"
    }
}

send_discord_message(embed)
print("Raport wysłany na Discorda!")