import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Definiujemy koszyk aktywów (Krypto, Złoto, Akcje)
TICKERS = ["BTC-USD", "ETH-USD", "GLD", "SPY"]
START = "2017-01-01" # Od 2017, bo wtedy zaczęło się ETH
END = "2025-01-01"
COST = 0.0004

# Słownik na wyniki strategii dla każdego aktywa
portfolio_returns = {}

for ticker in TICKERS:
    print(f"Przetwarzanie: {ticker}...")
    # 1. Pobieranie
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].rename(columns={"Close": "price"}).copy()
    
    # 2. Wskaźniki (zabezpieczenie przed dzieleniem przez zero przy braku danych)
    df["returns"] = df["price"].pct_change()
    df["ema20"] = df["price"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["price"].ewm(span=50, adjust=False).mean()
    df["daily_vol"] = df["returns"].rolling(30).std()
    df["annual_vol"] = df["daily_vol"] * np.sqrt(365)
    
    # 3. Sygnały i Volatility Targeting
    raw_sig = np.where(df["ema20"] > df["ema50"], 1.0, 0.0)
    pos = pd.Series(raw_sig, index=df.index).shift(1).fillna(0)
    
    # Celujemy w 40% rocznej zmienności (dla złota i SPY to będzie max 100%, dla krypto mniej)
    size = (0.40 / df["annual_vol"]).clip(upper=1.0).fillna(0)
    final_pos = pos * size
    
    # 4. Zwroty
    trades = final_pos.diff().abs().fillna(0)
    df["strat_ret"] = final_pos * df["returns"] - COST * trades
    
    # Zapisujemy tylko kolumnę ze zwrotami do słownika
    portfolio_returns[ticker] = df["strat_ret"]

# --- SKŁADANIE PORTFELA ---
# Łączymy zwroty w jedną tabelę (DataFrame). Różne aktywa mają różne daty startu, więc będą NaN.
returns_df = pd.DataFrame(portfolio_returns).dropna()

# Portfel równo-ważony: dzielimy kapitał po 25% na każde aktywo
weights = np.array([0.25, 0.25, 0.25, 0.25])
returns_df["portfolio_ret"] = (returns_df[TICKERS] * weights).sum(axis=1)

# 5. Equity i metryki
def get_metrics(equity, returns):
    cagr = (equity.iloc[-1]) ** (252 / len(equity)) - 1
    mdd = -((equity / equity.cummax()) - 1.0).min()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    return cagr, mdd, sharpe

# Obliczamy equity curve dla portfela
equity_portfolio = (1 + returns_df["portfolio_ret"]).cumprod()

# Obliczamy metryki
p_cagr, p_mdd, p_sharpe = get_metrics(equity_portfolio, returns_df["portfolio_ret"])

print("\n=== WYNIKI PORTFELA (Równo-ważony) ===")
print(f"CAGR: {p_cagr:.2%} | Max DD: {p_mdd:.2%} | Sharpe: {p_sharpe:.2f}")

# 6. Wykres
plt.figure(figsize=(12, 6))
plt.plot(equity_portfolio.index, equity_portfolio, label="Portfel Wieloaktywowy", linewidth=2, color="black")

# Dla porównania dodajemy pojedyncze aktywa (BTC i SPY)
for ticker in ["BTC-USD", "SPY"]:
    eq_single = (1 + returns_df[ticker]).cumprod()
    plt.plot(eq_single.index, eq_single, label=f"{ticker} Solo", alpha=0.4)

plt.yscale("log")
plt.title("Dywersyfikacja: Portfel Wieloaktywowy vs Pojedyncze Aktywa")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n=== Krok 2: Parytet Ryzyka (Inverse Volatility) ===")

# 1. Liczymy 30-dniową zmienność dla każdego aktywa (z oryginalnych zwrotów strategii)
# Im wyższa zmienność, tym mniejsza waga
vol_df = returns_df[TICKERS].rolling(30).std()

# 2. Odwrotność zmienności
inv_vol = 1.0 / vol_df

# 3. Normalizacja wag (żeby sumowały się do 1.0 czyli 100% kapitału)
weights_dynamic = inv_vol.div(inv_vol.sum(axis=1), axis=0)

# 4. Zwroty portfela ważone dynamicznie
returns_df["portfolio_ret_rp"] = (returns_df[TICKERS] * weights_dynamic).sum(axis=1)

# 5. Equity i metryki
equity_rp = (1 + returns_df["portfolio_ret_rp"].fillna(0)).cumprod()

rp_cagr, rp_mdd, rp_sharpe = get_metrics(equity_rp, returns_df["portfolio_ret_rp"].fillna(0))

print(f"CAGR: {rp_cagr:.2%} | Max DD: {rp_mdd:.2%} | Sharpe: {rp_sharpe:.2f}")

# 6. Wykres porównawczy
plt.figure(figsize=(12, 6))
plt.plot(equity_portfolio.index, equity_portfolio, label="Równo-ważony (25% każde)", alpha=0.6, linestyle="--")
plt.plot(equity_rp.index, equity_rp, label="Parytet Ryzyka (Inverse Vol)", linewidth=2, color="purple")
plt.yscale("log")
plt.title("Portfel: Równo-ważony vs Parytet Ryzyka")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()