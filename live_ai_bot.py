import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime
import winsound
import warnings

warnings.filterwarnings("ignore")

TICKER = "BTC-USD"
model = joblib.load('model_ai.pkl') # Wczytujemy zapisany model AI

# Funkcja do liczenia wskaźników na żywo
def get_features(df):
    df["Returns"] = df["Close"].pct_change()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["Volatility"] = df["Returns"].rolling(10).std()
    
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / loss)))
    
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    df["BB_Middle"] = df["Close"].rolling(20).mean()
    df["BB_Std"] = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + (2 * df["BB_Std"])
    df["BB_Lower"] = df["BB_Middle"] - (2 * df["BB_Std"])
    df["BB_Pct"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    
    # Dodajemy EMA20 i EMA50 dla trendu
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    
    return df

features = ["Returns", "SMA_10", "SMA_50", "Volatility", "RSI", "MACD", "MACD_Signal", "BB_Pct"]

print(f"🤖 Uruchamianie Live AI Bota dla {TICKER}...")
print("Bot używa wytrenowanego modelu Random Forest.")

position = 0

while True:
    try:
        # Pobieramy dane z ostatnich 5 dni (potrzebujemy min. 50 okresów na SMA_50)
        df = yf.download(TICKER, period="5d", interval="1m", progress=False)
        
        if df.empty or len(df) < 50:
            time.sleep(60)
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = get_features(df).dropna()
        if df.empty:
            continue

        current_price = df["Close"].iloc[-1]
        
        # AI Przewiduje na podstawie ostatniego wiersza (tuż przed zamknięciem świecy)
        last_row = df[features].iloc[-1:]
        ai_prediction = model.predict(last_row)[0]
        
        # Trend z EMA
        trend_up = df["EMA_20"].iloc[-1] > df["EMA_50"].iloc[-1]
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Logika wejścia: Trend wzrostowy ORAZ AI mówi, że cena wzrośnie
        if trend_up and ai_prediction == 1 and position == 0:
            winsound.Beep(1200, 500)
            print(f"[{timestamp}] 🟢 KUPUJ (AI + Trend)! Cena: ${current_price:,.2f}")
            position = 1
            
        # Logika wyjścia: AI mówi, że cena spadnie (predykcja = 0)
        elif position == 1 and ai_prediction == 0:
            winsound.Beep(500, 500)
            print(f"[{timestamp}] 🔴 SPRZEDAJ (AI - spadek)! Cena: ${current_price:,.2f}")
            position = 0
            
        else:
            status = "W RYNKU" if position == 1 else "CZEKA"
            print(f"[{timestamp}] ⚪ Status: {status} | Cena: ${current_price:,.2f} | AI: {'Wzrost' if ai_prediction==1 else 'Spadek'}")

    except Exception as e:
        print(f"Błąd: {e}")

    time.sleep(60)