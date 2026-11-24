import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

START = "2020-01-01"
END = "2025-01-01"
TICKER_1 = "MA"
TICKER_2 = "V"

# 1. Pobieranie danych
df1 = yf.download(TICKER_1, start=START, end=END, progress=False)
if isinstance(df1.columns, pd.MultiIndex):
    df1.columns = df1.columns.get_level_values(0)
    
df2 = yf.download(TICKER_2, start=START, end=END, progress=False)
if isinstance(df2.columns, pd.MultiIndex):
    df2.columns = df2.columns.get_level_values(0)

base_df = pd.DataFrame({TICKER_1: df1["Close"], TICKER_2: df2["Close"]}).dropna()

# 2. Funkcja backtestu
def run_pairs_backtest(lookback, entry_z, exit_z):
    df = base_df.copy()
    
    # Kroczący Hedge Ratio
    hedge_ratios = []
    for i in range(len(df)):
        if i < lookback:
            hedge_ratios.append(np.nan)
        else:
            y = df[TICKER_1].iloc[i-lookback:i]
            x = sm.add_constant(df[TICKER_2].iloc[i-lookback:i])
            model = sm.OLS(y, x).fit()
            hedge_ratios.append(model.params.iloc[1])
            
    df["Hedge_Ratio"] = hedge_ratios
    df["Spread"] = df[TICKER_1] - (df["Hedge_Ratio"] * df[TICKER_2])
    df["Mean"] = df["Spread"].rolling(lookback).mean()
    df["Std"] = df["Spread"].rolling(lookback).std()
    df["Zscore"] = (df["Spread"] - df["Mean"]) / df["Std"]
    
    # Logika wejścia/wyjścia
    df["Position"] = np.nan
    df.loc[df["Zscore"] < -entry_z, "Position"] = 1
    df.loc[df["Zscore"] > entry_z, "Position"] = -1
    df.loc[abs(df["Zscore"]) < exit_z, "Position"] = 0
    df["Position"] = df["Position"].ffill().fillna(0).shift(1).fillna(0)
    
    # Zwroty w dolarach
    df["Price_Change_1"] = df[TICKER_1].diff()
    df["Price_Change_2"] = df[TICKER_2].diff()
    df["Spread_Change"] = df["Price_Change_1"] - (df["Hedge_Ratio"] * df["Price_Change_2"])
    df["Capital_Deployed"] = df[TICKER_1] + (df["Hedge_Ratio"] * df[TICKER_2])
    
    df["Strategy_Ret"] = (df["Position"] * df["Spread_Change"]) / df["Capital_Deployed"]
    df["Trades"] = df["Position"].diff().abs().fillna(0)
    df["Strategy_Ret"] = df["Strategy_Ret"] - (0.0004 * df["Trades"])
    
    df = df.dropna()
    if len(df) == 0 or df["Strategy_Ret"].std() == 0:
        return -99, -99
        
    equity = (1 + df["Strategy_Ret"]).cumprod()
    
    # Obliczanie CAGR (zabezpieczone przed ujemnym kapitałem)
    final_eq = equity.iloc[-1]
    if final_eq > 0:
        cagr = (final_eq) ** (252 / len(df)) - 1
    else:
        cagr = -1.0
        
    sharpe = (df["Strategy_Ret"].mean() / df["Strategy_Ret"].std()) * np.sqrt(252)
    
    return sharpe, cagr

# 3. Siatka parametrów
lookbacks = [20, 40, 60]
entry_zs = [1.5, 2.0, 2.5]

print("=== Parameter Sweep dla Pairs Trading (MA/V) ===")
print("Lookback | Entry Z | Sharpe | CAGR")
print("-" * 40)

best_sharpe = -99
best_params = {}

for lb in lookbacks:
    for ez in entry_zs:
        sharpe, cagr = run_pairs_backtest(lb, ez, 0.5)
        print(f"{lb:8} | {ez:7.1f} | {sharpe:6.2f} | {cagr:6.2%}")
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = {"lookback": lb, "entry_z": ez}

print("\nNajlepsze parametry:", best_params)
print(f"Najlepszy Sharpe: {best_sharpe:.2f}")