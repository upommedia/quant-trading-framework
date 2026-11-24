import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

TICKER = "BTC-USD"
START = "2018-01-01"
END = "2025-01-01"

print("Pobieranie danych...")
df = yf.download(TICKER, start=START, end=END, progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# --- 1. ROZBUDOWANY FEATURE ENGINEERING ---
df["Returns"] = df["Close"].pct_change()
df["SMA_10"] = df["Close"].rolling(10).mean()
df["SMA_50"] = df["Close"].rolling(50).mean()
df["Volatility"] = df["Returns"].rolling(10).std()

# RSI
delta = df["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
df["RSI"] = 100 - (100 / (1 + (gain / loss)))

# NOWE: MACD (Moving Average Convergence Divergence)
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

# NOWE: Wstęgi Bollingera (Bollinger Bands)
df["BB_Middle"] = df["Close"].rolling(20).mean()
df["BB_Std"] = df["Close"].rolling(20).std()
df["BB_Upper"] = df["BB_Middle"] + (2 * df["BB_Std"])
df["BB_Lower"] = df["BB_Middle"] - (2 * df["BB_Std"])
df["BB_Pct"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"]) # Gdzie jesteśmy w kanale (0 do 1)

# Target
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df = df.dropna()

features = ["Returns", "SMA_10", "SMA_50", "Volatility", "RSI", "MACD", "MACD_Signal", "BB_Pct"]
X = df[features]
y = df["Target"]

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# --- 2. GRID SEARCH (Automatyczne szukanie najlepszych parametrów) ---
print(f"Trenowanie i optymalizacja modelu na {len(X_train)} dniach...")
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Najlepsze parametry: {grid_search.best_params_}")

# --- 3. TEST NA NIEWIDZIANYCH DANYCH ---
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nSkuteczność modelu AI v2: {accuracy:.2%}")

# --- 4. FEATURE IMPORTANCE (Co według AI jest najważniejsze?) ---
importances = best_model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 5))
plt.title("Ważność wskaźników (Feature Importance) według AI")
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Względna ważność")
plt.tight_layout()
plt.show()

# --- 5. BACKTEST STRATEGII ---
df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
df["Trend"] = np.where(df["EMA_20"] > df["EMA_50"], 1, 0)

df["AI_Prediction"] = np.nan
df.loc[X_test.index, "AI_Prediction"] = predictions

df["Signal"] = np.where((df["Trend"] == 1) & (df["AI_Prediction"] == 1), 1, 0)
df["Position"] = df["Signal"].shift(1).fillna(0)

df_test = df.iloc[split_idx:].copy()
df_test["Strategy_Returns"] = df_test["Position"] * df_test["Returns"]
df_test["Equity"] = (1 + df_test["Strategy_Returns"]).cumprod()
df_test["BuyHold"] = (1 + df_test["Returns"]).cumprod()

cagr = (df_test["Equity"].iloc[-1]) ** (365 / len(df_test)) - 1
sharpe = (df_test["Strategy_Returns"].mean() / df_test["Strategy_Returns"].std()) * np.sqrt(365)

print("\n=== Wyniki Strategii AI v2 ===")
print(f"CAGR: {cagr:.2%} | Sharpe Ratio: {sharpe:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(df_test.index, df_test["Equity"], label="AI v2 Trend-Following", linewidth=2)
plt.plot(df_test.index, df_test["BuyHold"], label="Buy & Hold", alpha=0.5)
plt.yscale("log")
plt.title("Strategia AI v2 vs Buy & Hold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

import joblib

# Zapisujemy najlepszy znaleziony model do pliku
joblib.dump(best_model, 'model_ai.pkl')
print("✅ Model AI został zapisany do pliku model_ai.pkl")