import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime
import warnings
import winsound
import csv
import os

warnings.filterwarnings("ignore")

TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD"]
STOP_LOSS_PCT = 0.015 # 1.5% stop loss od najwyższego punktu
LOG_FILE = "historia_transakcji.csv"

# Inicjalizacja pliku CSV (jeśli nie istnieje)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Data", "Ticker", "Akcja", "Cena", "Zysk/Strata (%)", "Powod"])

# Słownik przechowujący stan portfela
portfolio = {ticker: {"cash": 10000.0, "crypto": 0.0, "pos": 0, "max_price": 0.0} for ticker in TICKERS}

def play_alert(is_buy):
    if is_buy:
        winsound.Beep(1200, 500)
    else:
        winsound.Beep(500, 500)

def log_trade(ticker, action, price, profit, reason):
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ticker, action, f"{price:.2f}", f"{profit:.2f}", reason])

print("🤖 Uruchamianie bota Multi-Asset z Trailing Stop-Loss...")
print("Monitorowane aktywa:", ", ".join(TICKERS))
print("Aby zatrzymać, wciśnij Ctrl+C.\n")

while True:
    print(f"--- Skanowanie rynku: {datetime.now().strftime('%H:%M:%S')} ---")
    
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, period="2d", interval="1m", progress=False)
            
            if df.empty or len(df) < 50:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
            df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()

            current_price = df["Close"].iloc[-1]
            current_ema20 = df["ema20"].iloc[-1]
            current_ema50 = df["ema50"].iloc[-1]

            p = portfolio[ticker]
            total_equity = p["cash"] + (p["crypto"] * current_price)
            current_signal = 1 if current_ema20 > current_ema50 else 0
            diff_pct = ((current_ema20 / current_ema50) - 1) * 100

            # --- LOGIKA SPRZEDAŻY (Trend lub Stop Loss) ---
            if p["pos"] == 1:
                # Aktualizacja najwyższej ceny od momentu zakupu
                if current_price > p["max_price"]:
                    p["max_price"] = current_price
                
                # Sprawdzamy warunki wyjścia
                trend_reversal = current_signal == 0
                stop_loss_hit = current_price <= p["max_price"] * (1 - STOP_LOSS_PCT)

                if trend_reversal or stop_loss_hit:
                    p["cash"] = p["crypto"] * current_price
                    p["crypto"] = 0.0
                    p["pos"] = 0
                    profit_pct = ((p["cash"] / 10000.0) - 1) * 100
                    
                    reason = "Stop Loss" if stop_loss_hit else "Trend Reversal"
                    play_alert(is_buy=False)
                    print(f"  🔴 [{ticker}] SPRZEDAJ! Cena: ${current_price:,.2f} | Zysk: {profit_pct:+.2f}% | Powód: {reason}")
                    log_trade(ticker, "SPRZEDAJ", current_price, profit_pct, reason)

            # --- LOGIKA KUPNA ---
            elif p["pos"] == 0 and current_signal == 1:
                p["crypto"] = p["cash"] / current_price
                p["cash"] = 0.0
                p["pos"] = 1
                p["max_price"] = current_price # Ustawiamy startowy szczyt dla stop-lossa
                play_alert(is_buy=True)
                print(f"  🟢 [{ticker}] KUPUJ! Cena: ${current_price:,.2f} | Wartość: ${total_equity:,.2f}")
                log_trade(ticker, "KUP", current_price, 0.0, "EMA Crossover")

            # --- MONITOROWANIE ---
            else:
                status = "W RYNKU" if p["pos"] == 1 else "CZEKA"
                print(f"  ⚪ [{ticker}] {status:7} | Cena: ${current_price:,.2f} | Różnica EMA: {diff_pct:+.3f}% | Kapitał: ${total_equity:,.2f}")

        except Exception as e:
            print(f"  Błąd dla {ticker}: {e}")
            
        time.sleep(2)

    print("") 
    time.sleep(60)