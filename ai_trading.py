import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

TICKER = "BTC-USD"
START = "2018-01-01"
END = "2025-01-01"

print("Pobieranie danych...")
df = yf.download(TICKER, start=START, end=END, progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 1. Inżynieria cech (Feature Engineering) - na czym uczy się AI?
df["Returns"] = df["Close"].pct_change()
df["SMA_10"] = df["Close"].rolling(10).mean()
df["SMA_50"] = df["Close"].rolling(50).mean()
df["Volatility"] = df["Returns"].rolling(10).std()

# Wskaźnik RSI (Relative Strength Index)
delta = df["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
df["RSI"] = 100 - (100 / (1 + (gain / loss)))

# 2. Definiujemy Cel (Target) - co AI ma przewidywać?
# 1 = następnego dnia cena wzrośnie, 0 = spadnie
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# Usuwamy puste wiersze (przez średnie i RSI)
df = df.dropna()

# 3. Przygotowanie danych do treningu (Bez Lookahead Bias!)
features = ["Returns", "SMA_10", "SMA_50", "Volatility", "RSI"]
X = df[features]
y = df["Target"]

# Dzielimy dane: 80% na trening (przeszłość), 20% na test (przyszłość)
# WAŻNE: shuffle=False, bo to dane finansowe! Nie mieszamy czasu.
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Trenowanie modelu na {len(X_train)} dniach...")
# 4. Trening modelu Random Forest (Las Losowy)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 5. Testowanie na niewidzianych danych (Out-of-Sample)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Skuteczność modelu AI: {accuracy:.2%}")

# 6. Budowa strategii (EMA + Filtr AI)
df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
df["Trend"] = np.where(df["EMA_20"] > df["EMA_50"], 1, 0)

# Przypisujemy predykcje AI tylko do zbioru testowego
df["AI_Prediction"] = np.nan
df.loc[X_test.index, "AI_Prediction"] = predictions

# Logika: Kupujemy tylko gdy trend rośnie ORAZ AI mówi "1"
df["Signal"] = np.where((df["Trend"] == 1) & (df["AI_Prediction"] == 1), 1, 0)
df["Position"] = df["Signal"].shift(1).fillna(0) # Przesunięcie o 1 dzień!

# 7. Zwroty i metryki (tylko dla okresu testowego)
df_test = df.iloc[split_idx:].copy()
df_test["Strategy_Returns"] = df_test["Position"] * df_test["Returns"]
df_test["Equity"] = (1 + df_test["Strategy_Returns"]).cumprod()
df_test["BuyHold"] = (1 + df_test["Returns"]).cumprod()

# Metryki
cagr = (df_test["Equity"].iloc[-1]) ** (365 / len(df_test)) - 1
sharpe = (df_test["Strategy_Returns"].mean() / df_test["Strategy_Returns"].std()) * np.sqrt(365)

print("\n=== Wyniki Strategii AI (Ostatnie 20% danych) ===")
print(f"CAGR: {cagr:.2%} | Sharpe Ratio: {sharpe:.2f}")

# 8. Wykres
plt.figure(figsize=(12, 6))
plt.plot(df_test.index, df_test["Equity"], label="AI Trend-Following", linewidth=2)
plt.plot(df_test.index, df_test["BuyHold"], label="Buy & Hold", alpha=0.5)
plt.yscale("log")
plt.title("Strategia z AI vs Buy & Hold (Okres Testowy)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()