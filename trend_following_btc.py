import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TICKER = "BTC-USD"
START = "2015-01-01"
END = "2025-01-01"

# 1. Pobranie danych
df = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df[["Close"]].rename(columns={"Close": "price"}).copy()

# 2. Wskaźniki (średnie kroczące)
df["sma50"] = df["price"].rolling(50).mean()
df["sma200"] = df["price"].rolling(200).mean()

# 3. Sygnał: 1 gdy SMA50 > SMA200, 0 gdy SMA50 < SMA200
df["raw_signal"] = np.where(df["sma50"] > df["sma200"], 1.0, 0.0)

# Przesunięcie o 1 dzień, żeby nie było lookahead bias (wchodzimy na następnej świecy)
df["position"] = pd.Series(df["raw_signal"], index=df.index).shift(1).fillna(0)

# 4. Zwroty i koszty
df["returns"] = df["price"].pct_change()
COST = 0.0004 # 0.04% (krypto ma większe prowizje niż akcje)
trades = df["position"].diff().abs().fillna(0)
df["strategy_returns"] = df["position"] * df["returns"] - COST * trades

df = df.dropna(subset=["sma200", "returns"])

# 5. Equity curve
df["equity_strategy"] = (1 + df["strategy_returns"]).cumprod()
df["equity_bh"] = (1 + df["returns"]).cumprod()

# 6. Metryki
def get_metrics(equity, returns, days_per_year=365): # 365 dla krypto!
    cagr = (equity.iloc[-1]) ** (days_per_year / len(equity)) - 1
    mdd = -((equity / equity.cummax()) - 1.0).min()
    win_rate = (returns[returns != 0] > 0).mean()
    return cagr, mdd, win_rate

s_cagr, s_mdd, s_wr = get_metrics(df["equity_strategy"], df["strategy_returns"])
b_cagr, b_mdd, b_wr = get_metrics(df["equity_bh"], df["returns"])

print("=== Trend-Following (SMA 50/200) na BTC ===")
print(f"CAGR: {s_cagr:.2%} | Max DD: {s_mdd:.2%} | Win Rate: {s_wr:.2%} | Dni w pozycji: {int(df['position'].sum())} / {len(df)}")
print("\n=== Buy & Hold BTC ===")
print(f"CAGR: {b_cagr:.2%} | Max DD: {b_mdd:.2%} | Win Rate: {b_wr:.2%}")

# 7. Wykres
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["equity_strategy"], label="Trend-Following (SMA 50/200)", linewidth=1.5)
plt.plot(df.index, df["equity_bh"], label="Buy & Hold BTC", alpha=0.7)
plt.yscale("log") # Skala logarytmiczna - kluczowa dla krypto
plt.title("Equity: Trend-Following na BTC vs Buy & Hold (skala log)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

import numpy as np

print("\n=== Krok 2: Sharpe Ratio i szybszy trend (EMA 20/50) ===")

# 1. Funkcja do liczenia Sharpe'a
def sharpe_ratio(returns, days_per_year=365):
    # Zakładamy stopę wolną od ryzyka = 0
    return (returns.mean() / returns.std()) * np.sqrt(days_per_year)

print(f"Sharpe Ratio (SMA 50/200): {sharpe_ratio(df['strategy_returns']):.2f}")
print(f"Sharpe Ratio (Buy & Hold): {sharpe_ratio(df['returns']):.2f}")

# 2. Szybsza strategia - EMA 20/50
df["ema20"] = df["price"].ewm(span=20, adjust=False).mean()
df["ema50"] = df["price"].ewm(span=50, adjust=False).mean()

# Sygnał: 1 gdy EMA20 > EMA50
df["raw_signal_fast"] = np.where(df["ema20"] > df["ema50"], 1.0, 0.0)
df["position_fast"] = pd.Series(df["raw_signal_fast"], index=df.index).shift(1).fillna(0)

# Zwroty z kosztami
trades_fast = df["position_fast"].diff().abs().fillna(0)
df["strategy_returns_fast"] = df["position_fast"] * df["returns"] - COST * trades_fast

# Equity i metryki
df["equity_fast"] = (1 + df["strategy_returns_fast"]).cumprod()
f_cagr, f_mdd, f_wr = get_metrics(df["equity_fast"], df["strategy_returns_fast"])

print("\n=== EMA 20/50 (Szybszy trend) ===")
print(f"CAGR: {f_cagr:.2%} | Max DD: {f_mdd:.2%} | Win Rate: {f_wr:.2%} | Dni w pozycji: {int(df['position_fast'].sum())} / {len(df)}")
print(f"Sharpe Ratio (EMA 20/50): {sharpe_ratio(df['strategy_returns_fast']):.2f}")

# 3. Wykres - dodajemy EMA 20/50
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["equity_strategy"], label="SMA 50/200 (wolny)", alpha=0.6, linestyle="--")
plt.plot(df.index, df["equity_fast"], label="EMA 20/50 (szybki)", linewidth=1.5)
plt.plot(df.index, df["equity_bh"], label="Buy & Hold BTC", alpha=0.5)
plt.yscale("log")
plt.title("Porównanie: Szybki vs Wolny Trend-Following na BTC")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n=== Krok 3: Trend-Following + Volatility Targeting ===")

# 1. Liczymy zmienność (odchylenie standardowe z 30 dni) i roczną (365 dni)
df["daily_vol"] = df["returns"].rolling(30).std()
df["annual_vol"] = df["daily_vol"] * np.sqrt(365)

# 2. Ustalamy cel zmienności na 40% rocznie (dla krypto to i tak dużo, ale bezpieczniej)
TARGET_VOL = 0.40

# 3. Obliczamy rozmiar pozycji (im większa zmienność, tym mniejsza pozycja)
# Clip(upper=1.0) oznacza, że nie używamy dźwigni (max 100% kapitału)
df["position_size"] = (TARGET_VOL / df["annual_vol"]).clip(upper=1.0)
df["position_size"] = df["position_size"].fillna(0) # dla pierwszych dni z brakiem zmienności

# 4. Nowa pozycja to sygnał (1 lub 0) pomnożony przez rozmiar (np. 0.5)
df["final_position"] = df["position_fast"] * df["position_size"]

# 5. Zwroty z uwzględnieniem kosztów transakcyjnych
trades_vol = df["final_position"].diff().abs().fillna(0)
df["strategy_returns_vol"] = df["final_position"] * df["returns"] - COST * trades_vol

# 6. Equity i metryki
df["equity_vol"] = (1 + df["strategy_returns_vol"]).cumprod()

# Używamy funkcji get_metrics, którą napisaliśmy wcześniej
v_cagr, v_mdd, v_wr = get_metrics(df["equity_vol"], df["strategy_returns_vol"])

print(f"CAGR: {v_cagr:.2%} | Max DD: {v_mdd:.2%} | Win Rate: {v_wr:.2%}")
print(f"Sharpe Ratio (Vol Target): {sharpe_ratio(df['strategy_returns_vol']):.2f}")

# 7. Wykres porównawczy
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["equity_fast"], label="EMA 20/50 (bez zarządzania ryzykiem)", alpha=0.5, linestyle="--")
plt.plot(df.index, df["equity_vol"], label="EMA 20/50 + Volatility Targeting", linewidth=1.5, color="green")
plt.plot(df.index, df["equity_bh"], label="Buy & Hold BTC", alpha=0.5)
plt.yscale("log")
plt.title("Wpływ zarządzania wielkością pozycji na BTC")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n=== Krok 4: Out-of-Sample Test (2021 - 2025) ===")

# 1. Pobieramy dane jeszcze raz
df_oos = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
if isinstance(df_oos.columns, pd.MultiIndex):
    df_oos.columns = df_oos.columns.get_level_values(0)
df_oos = df_oos[["Close"]].rename(columns={"Close": "price"}).copy()

# 2. Liczymy wskaźniki na CAŁYCH danych (żeby średnie się rozgrzały)
df_oos["returns"] = df_oos["price"].pct_change()
df_oos["ema20"] = df_oos["price"].ewm(span=20, adjust=False).mean()
df_oos["ema50"] = df_oos["price"].ewm(span=50, adjust=False).mean()
df_oos["daily_vol"] = df_oos["returns"].rolling(30).std()
df_oos["annual_vol"] = df_oos["daily_vol"] * np.sqrt(365)

# 3. Budujemy sygnały
raw_sig = np.where(df_oos["ema20"] > df_oos["ema50"], 1.0, 0.0)
pos = pd.Series(raw_sig, index=df_oos.index).shift(1).fillna(0)
size = (0.40 / df_oos["annual_vol"]).clip(upper=1.0).fillna(0)
df_oos["final_pos"] = pos * size

# 4. Zwroty i koszty
trades_oos = df_oos["final_pos"].diff().abs().fillna(0)
df_oos["strat_ret"] = df_oos["final_pos"] * df_oos["returns"] - COST * trades_oos

# --- TUTAJ JEST KLUCZ: UCINAMY DANE DO LAT 2021-2025 ---
df_oos = df_oos.loc["2021-01-01":]

# 5. Equity i metryki
df_oos["equity_strat"] = (1 + df_oos["strat_ret"]).cumprod()
df_oos["equity_bh"] = (1 + df_oos["returns"]).cumprod()

oos_cagr, oos_mdd, oos_wr = get_metrics(df_oos["equity_strat"], df_oos["strat_ret"])
oos_sharpe = sharpe_ratio(df_oos["strat_ret"])

bh_oos_cagr, bh_oos_mdd, _ = get_metrics(df_oos["equity_bh"], df_oos["returns"])

print(f"Strategia (2021-2025): CAGR: {oos_cagr:.2%} | Max DD: {oos_mdd:.2%} | Sharpe: {oos_sharpe:.2f}")
print(f"Buy & Hold (2021-2025): CAGR: {bh_oos_cagr:.2%} | Max DD: {bh_oos_mdd:.2%}")

# 6. Wykres
plt.figure(figsize=(12, 6))
plt.plot(df_oos.index, df_oos["equity_strat"], label="Strategia (Out-of-Sample)", linewidth=2)
plt.plot(df_oos.index, df_oos["equity_bh"], label="Buy & Hold BTC", alpha=0.6)
plt.yscale("log")
plt.title("Out-of-Sample Test: 2021 - 2025 (Niewidziane dane)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()