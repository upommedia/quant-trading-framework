import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tickers = ['BTC-USD', 'ETH-USD', 'SPY', 'GLD']
start = '2018-01-01'
end = '2025-01-01'

print("Pobieranie danych...")
data = yf.download(tickers, start=start, end=end, progress=False)['Close']
data = data.ffill().dropna() 
returns = data.pct_change().dropna()

print("Uruchamianie dynamicznej optymalizacji (Rolling 252 dni)...")
LOOKBACK = 252
REBALANCE = 21 

# Pusta lista na zwroty portfela
portfolio_returns = []

for i in range(LOOKBACK, len(returns), REBALANCE):
    # 1. Dane z ostatniego roku
    train_window = returns.iloc[i-LOOKBACK:i]
    num_assets = len(tickers)
    mean_returns = train_window.mean() * 252
    cov_matrix = train_window.cov() * 252
    
    # 2. Losowanie 500 portfeli (Grid Search)
    best_sharpe = -999
    best_weights = np.array([1.0/num_assets] * num_assets) # Domyślnie 25% na każde
    
    for _ in range(500):
        w = np.random.random(num_assets)
        w /= np.sum(w)
        
        p_ret = np.sum(mean_returns * w)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        
        if p_vol > 0 and not np.isnan(p_vol):
            sharpe = p_ret / p_vol
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = w
                
    # 3. Aplikowanie wag na następny miesiąc
    test_window = returns.iloc[i:i+REBALANCE]
    
    # KLUCZOWA POPRAWKA: Używamy funkcji .dot() z Pandas, która jest odporna na błędy
    # Mnoży każdy wiersz (dzień) przez wektor wag
    monthly_returns = test_window.dot(best_weights)
    portfolio_returns.extend(monthly_returns.tolist())

# Tworzymy equity curve z zebranych zwrotów
# Bierzemy indeks dat z ostatniego okna (po prostu odcinamy pierwszy rok)
dates = returns.index[LOOKBACK:]
dynamic_eq = (1 + pd.Series(portfolio_returns, index=dates)).cumprod()

# Buy & Hold SPY dla porównania
bh_eq = (1 + returns['SPY'].iloc[LOOKBACK:]).cumprod() 

def get_metrics(eq):
    if eq.iloc[-1] <= 0: return -1, 1, -99
    cagr = (eq.iloc[-1]) ** (252 / len(eq)) - 1
    mdd = -((eq / eq.cummax()) - 1.0).min()
    rets = eq.pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    return cagr, mdd, sharpe

d_cagr, d_mdd, d_sharpe = get_metrics(dynamic_eq)
b_cagr, b_mdd, b_sharpe = get_metrics(bh_eq)

print("\n=== Wyniki Dynamicznego Portfela (Out-of-Sample) ===")
print(f"Dynamiczny Markowitz: CAGR: {d_cagr:.2%} | Max DD: {d_mdd:.2%} | Sharpe: {d_sharpe:.2f}")
print(f"Buy & Hold (SPY):     CAGR: {b_cagr:.2%} | Max DD: {b_mdd:.2%} | Sharpe: {b_sharpe:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(dynamic_eq.index, dynamic_eq, label="Dynamiczny Markowitz", linewidth=2)
plt.plot(bh_eq.index, bh_eq, label="Buy & Hold SPY", alpha=0.6)
plt.yscale("log")
plt.title("Dynamiczna Optymalizacja Portfela vs Buy & Hold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()