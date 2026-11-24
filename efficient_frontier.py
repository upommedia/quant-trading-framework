import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Definiujemy koszyk aktywów
tickers = ['BTC-USD', 'ETH-USD', 'SPY', 'GLD']
start = '2020-01-01'
end = '2025-01-01'

print("Pobieranie danych...")
# Pobieramy dane i łączymy je w jedną tabelę
data = yf.download(tickers, start=start, end=end, progress=False)['Close']
data = data.dropna()

# Obliczamy dzienne zwroty
returns = data.pct_change().dropna()

# Roczna stopa zwrotu i macierz kowariancji (ryzyko)
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

# --- SYMULACJA MONTE CARLO (Losowe portfele) ---
num_portfolios = 10000
results = np.zeros((3, num_portfolios))
weights_record = []

for i in range(num_portfolios):
    # Losujemy wagi (sumują się do 1)
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    weights_record.append(weights)
    
    # Oczekiwany zwrot portfela
    portfolio_return = np.sum(mean_returns * weights)
    
    # Ryzyko portfela (odchylenie standardowe)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Zapisujemy wyniki
    results[0,i] = portfolio_std_dev * 100 # Ryzyko w %
    results[1,i] = portfolio_return * 100 # Zysk w %
    results[2,i] = (portfolio_return / portfolio_std_dev) # Sharpe Ratio (bez stopy wolnej od ryzyka)

# --- ZNAJDOWANIE NAJLEPSZYCH PORTFELI ---
# 1. Maksymalny Sharpe Ratio
max_sharpe_idx = np.argmax(results[2])
max_sharpe_risk = results[0, max_sharpe_idx]
max_sharpe_return = results[1, max_sharpe_idx]
max_sharpe_weights = weights_record[max_sharpe_idx]

# 2. Minimalne ryzyko (Min Volatility)
min_vol_idx = np.argmin(results[0])
min_vol_risk = results[0, min_vol_idx]
min_vol_return = results[1, min_vol_idx]
min_vol_weights = weights_record[min_vol_idx]

print("\n=== PORTFEL O MAKSYMALNYM SHARPE RATIO ===")
print(f"Oczekiwany Zysk: {max_sharpe_return:.2f}% | Ryzyko: {max_sharpe_risk:.2f}% | Sharpe: {results[2, max_sharpe_idx]:.2f}")
for i, ticker in enumerate(tickers):
    print(f"{ticker}: {max_sharpe_weights[i]*100:.1f}%")

print("\n=== PORTFEL O MINIMALNYM RYZYKU ===")
print(f"Oczekiwany Zysk: {min_vol_return:.2f}% | Ryzyko: {min_vol_risk:.2f}% | Sharpe: {results[2, min_vol_idx]:.2f}")
for i, ticker in enumerate(tickers):
    print(f"{ticker}: {min_vol_weights[i]*100:.1f}%")

# --- WYKRES ---
plt.figure(figsize=(12, 7))
# Rysujemy chmurę 10,000 portfeli
sc = plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', marker='o', s=10, alpha=0.3)
plt.colorbar(sc, label='Sharpe Ratio')

# Zaznaczamy nasze dwa najlepsze portfele
plt.scatter(max_sharpe_risk, max_sharpe_return, marker='*', color='red', s=500, label='Max Sharpe Ratio')
plt.scatter(min_vol_risk, min_vol_return, marker='X', color='blue', s=200, label='Min Ryzyko (Min Volatility)')

plt.title('Efektywna Granica Markowitza (10,000 symulacji)')
plt.xlabel('Ryzyko (Odchylenie standardowe) [%]')
plt.ylabel('Oczekiwany roczny zysk [%]')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()