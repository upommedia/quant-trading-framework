import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TICKER = "BTC-USD"
START = "2018-01-01"
END = "2025-01-01"

df = yf.download(TICKER, start=START, end=END, progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df[["Close"]].rename(columns={"Close": "price"}).copy()

# Wskaźniki (Trend Following)
df["returns"] = df["price"].pct_change()
df["ema20"] = df["price"].ewm(span=20, adjust=False).mean()
df["ema50"] = df["price"].ewm(span=50, adjust=False).mean()

# Sygnał
df["signal"] = np.where(df["ema20"] > df["ema50"], 1.0, 0.0)
df["position_base"] = df["signal"].shift(1).fillna(0)
df["strat_returns_base"] = df["position_base"] * df["returns"]

# --- OBLICZANIE KELLY'EGO ---
# Bierzemy ostatnie 90 dni transakcji, żeby oszacować Win Rate i Risk/Reward
LOOKBACK_KELLY = 90
wins = []
losses = []

# Symulujemy historię dla okna kroczącego
for i in range(len(df)):
    if df["position_base"].iloc[i] == 1:
        ret = df["returns"].iloc[i]
        if ret > 0:
            wins.append(ret)
        else:
            losses.append(abs(ret))

# Statystyki (używamy średnich z całej historii dla uproszczenia)
win_rate = len(wins) / (len(wins) + len(losses)) if (len(wins) + len(losses)) > 0 else 0.5
avg_win = np.mean(wins) if wins else 0.01
avg_loss = np.mean(losses) if losses else 0.01
risk_reward = avg_win / avg_loss

# Wzór Kelly'ego
kelly_pct = win_rate - ((1 - win_rate) / risk_reward)

print(f"=== Statystyki strategii ===")
print(f"Win Rate: {win_rate:.2%}")
print(f"Średni zysk (win): {avg_win:.2%}")
print(f"Średnia strata (loss): {avg_loss:.2%}")
print(f"Risk/Reward Ratio: {risk_reward:.2f}")
print(f"Optymalny ułamek kapitału (Kelly %): {kelly_pct:.2%}")

# Zabezpieczenie: Kelly często daje absurdalnie wysokie wyniki (np. 200%), więc ucinamy do 100% (max 1.0)
kelly_position_size = np.clip(kelly_pct, 0.0, 1.0)

# --- NOWA STRATEGIA Z KELLY ---
df["strat_returns_kelly"] = df["position_base"] * df["returns"] * kelly_position_size

df["equity_base"] = (1 + df["strat_returns_base"]).cumprod()
df["equity_kelly"] = (1 + df["strat_returns_kelly"]).cumprod()
df["equity_bh"] = (1 + df["returns"]).cumprod()

# Wykres
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["equity_base"], label="Trend Following (100% kapitału)", alpha=0.5)
plt.plot(df.index, df["equity_kelly"], label=f"Trend Following + Kelly ({kelly_position_size:.1%} kapitału)", linewidth=2, color="green")
plt.plot(df.index, df["equity_bh"], label="Buy & Hold", alpha=0.4)
plt.yscale("log")
plt.title("Wpływ Kryterium Kelly'ego na kapitał")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()