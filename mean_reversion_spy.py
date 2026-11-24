import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TICKER = "TLT"
START = "2015-01-01"
END = "2025-01-01"

df = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)

# --- FIX: spłaszczenie MultiIndex ---
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[["Close"]].rename(columns={"Close": "price"}).copy()

print(df.head())
print(f"Liczba obserwacji: {len(df)}")

# Parametry strategii
LOOKBACK = 20
ENTRY_Z = -1.0
EXIT_Z = 0.0

# Wskaźniki
df["sma"] = df["price"].rolling(LOOKBACK).mean()
df["std"] = df["price"].rolling(LOOKBACK).std()
df["zscore"] = (df["price"] - df["sma"]) / df["std"]

# Sygnał surowy: 1 = jesteśmy w pozycji, 0 = poza
# Wejście gdy z < ENTRY_Z; wyjście gdy z > EXIT_Z
raw_signal = pd.Series(np.nan, index=df.index)
raw_signal[df["zscore"] < ENTRY_Z] = 1
raw_signal[df["zscore"] > EXIT_Z] = 0
raw_signal = raw_signal.ffill().fillna(0)

# Przesunięcie o 1 dzień: decyzja podjęta na koniec dnia t, wykonanie na t+1
df["position"] = raw_signal.shift(1).fillna(0)

# Zwroty
df["returns"] = df["price"].pct_change()
df["strategy_returns"] = df["position"] * df["returns"]

# --- KOSZTY TRANSAKCYJNE ---
COST = 0.0002  # 0.02% prowizji + spread na transakcję
# Różnica w pozycji (1 = wejście, -1 = wyjście), suma wartości bezwzględnych to liczba transakcji
trades = df["position"].diff().abs().fillna(0)
df["strategy_returns"] = df["position"] * df["returns"] - COST * trades

# Usunięcie pierwszych wierszy z NaN (okres rozgrzewania średniej)
df = df.dropna(subset=["zscore", "returns"])

print(df[["price", "zscore", "position", "strategy_returns"]].head(10))

def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Compound Annual Growth Rate na podstawie equity curve."""
    total_return = equity.iloc[-1] / equity.iloc[0]
    n_years = len(equity) / periods_per_year
    return total_return ** (1 / n_years) - 1

def max_drawdown(equity: pd.Series) -> float:
    """Maksymalne obsunięcie (jako liczba dodatnia, np. 0.25 = 25%)."""
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    return -drawdown.min()

def win_rate(trade_returns: pd.Series) -> float:
    """Odsetek transakcji z dodatnim zwrotem."""
    if len(trade_returns) == 0:
        return 0.0
    return (trade_returns > 0).mean()

# Equity curves
df["equity_strategy"] = (1 + df["strategy_returns"]).cumprod()
df["equity_bh"] = (1 + df["returns"]).cumprod()

# Metryki dla strategii
strat_cagr = cagr(df["equity_strategy"])
strat_mdd = max_drawdown(df["equity_strategy"])
strat_win = win_rate(df.loc[df["strategy_returns"] != 0, "strategy_returns"])

# Metryki dla buy & hold
bh_cagr = cagr(df["equity_bh"])
bh_mdd = max_drawdown(df["equity_bh"])
bh_win = win_rate(df.loc[df["returns"] != 0, "returns"])

print("=== Strategia mean-reversion ===")
print(f"CAGR:          {strat_cagr:.2%}")
print(f"Max Drawdown:  {strat_mdd:.2%}")
print(f"Win Rate:      {strat_win:.2%}")
print(f"Liczba dni w pozycji: {int(df['position'].sum())}")

print("\n=== Buy & Hold ===")
print(f"CAGR:          {bh_cagr:.2%}")
print(f"Max Drawdown:  {bh_mdd:.2%}")
print(f"Win Rate:      {bh_win:.2%}")

# Wykres equity curves
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df["equity_strategy"], label="Mean-reversion", linewidth=1.5)
ax.plot(df.index, df["equity_bh"], label="Buy & Hold SPY", linewidth=1.5, alpha=0.8)
ax.set_title("Equity curve: strategia mean-reversion vs Buy & Hold (SPY)")
ax.set_ylabel("Kapitał (start = 1.0)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- PARAMETER SWEEP ---
print("\n=== Parameter Sweep (CAGR) ===")
lookbacks = [10, 20, 50]
entry_zs = [-0.5, -1.0, -1.5, -2.0]

# Pobieramy czyste dane jeszcze raz, żeby nie mieszać z naszą strategią
df_sweep = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
if isinstance(df_sweep.columns, pd.MultiIndex):
    df_sweep.columns = df_sweep.columns.get_level_values(0)
df_sweep = df_sweep[["Close"]].rename(columns={"Close": "price"}).copy()
df_sweep["returns"] = df_sweep["price"].pct_change()

results = []
for lb in lookbacks:
    for ez in entry_zs:
        # Obliczanie wskaźników
        sma = df_sweep["price"].rolling(lb).mean()
        std = df_sweep["price"].rolling(lb).std()
        z = (df_sweep["price"] - sma) / std
        
        # Sygnał
        sig = pd.Series(np.nan, index=df_sweep.index)
        sig[z < ez] = 1
        sig[z > 0] = 0
        pos = sig.ffill().fillna(0).shift(1).fillna(0)
        
        # Zwroty i koszty
        tr = pos * df_sweep["returns"] - 0.0002 * pos.diff().abs().fillna(0)
        eq = (1 + tr).cumprod().dropna()
        
        if len(eq) > 0:
            cagr_val = (eq.iloc[-1]) ** (252 / len(eq)) - 1
            results.append({"Lookback": lb, "Entry_Z": ez, "CAGR": cagr_val})

# Wyświetlenie wyników jako tabeli
res_df = pd.DataFrame(results).pivot(index="Lookback", columns="Entry_Z", values="CAGR")
print((res_df * 100).round(2).astype(str) + "%")

print("\n=== Strategia: Mean-reversion + 200-dniowy filtr trendu ===")

# Pobieramy czyste dane (jak do sweepa)
df_trend = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
if isinstance(df_trend.columns, pd.MultiIndex):
    df_trend.columns = df_trend.columns.get_level_values(0)
df_trend = df_trend[["Close"]].rename(columns={"Close": "price"}).copy()
df_trend["returns"] = df_trend["price"].pct_change()

# Wskaźniki
df_trend["sma20"] = df_trend["price"].rolling(20).mean()
df_trend["std20"] = df_trend["price"].rolling(20).std()
df_trend["zscore"] = (df_trend["price"] - df_trend["sma20"]) / df_trend["std20"]
df_trend["sma200"] = df_trend["price"].rolling(200).mean()

# Sygnał wejścia (jak wcześniej)
raw_sig_trend = pd.Series(np.nan, index=df_trend.index)
raw_sig_trend[(df_trend["zscore"] < -1.0) & (df_trend["price"] > df_trend["sma200"])] = 1
raw_sig_trend[df_trend["zscore"] > 0.0] = 0

# NOWOŚĆ: Wymuszony exit, jeśli cena spadnie poniżej SMA200
raw_sig_trend[df_trend["price"] < df_trend["sma200"]] = 0

df_trend["position"] = raw_sig_trend.ffill().fillna(0).shift(1).fillna(0)

df_trend["position"] = raw_sig_trend.ffill().fillna(0).shift(1).fillna(0)

# Zwroty z kosztami
trades_trend = df_trend["position"].diff().abs().fillna(0)
df_trend["strategy_returns"] = df_trend["position"] * df_trend["returns"] - 0.0002 * trades_trend

df_trend = df_trend.dropna(subset=["zscore", "sma200", "returns"])

# Equity
df_trend["equity_strategy"] = (1 + df_trend["strategy_returns"]).cumprod()
df_trend["equity_bh"] = (1 + df_trend["returns"]).cumprod()

# Metryki
print(f"CAGR:          {cagr(df_trend['equity_strategy']):.2%}")
print(f"Max Drawdown:  {max_drawdown(df_trend['equity_strategy']):.2%}")
print(f"Win Rate:      {win_rate(df_trend.loc[df_trend['strategy_returns'] != 0, 'strategy_returns']):.2%}")
print(f"Liczba dni w pozycji: {int(df_trend['position'].sum())} (z {len(df_trend)})")

# Wykres porównawczy
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_trend.index, df_trend["equity_strategy"], label="Mean-reversion + Trend Filter")
ax.plot(df_trend.index, df_trend["equity_bh"], label="Buy & Hold SPY", alpha=0.8)
ax.set_title("Equity: Mean-reversion z filtrem trendu vs Buy & Hold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()