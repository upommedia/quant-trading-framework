import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Konfiguracja strony
st.set_page_config(page_title="Quant Trading Dashboard", layout="wide")
st.title("📈 Profesjonalny Dashboard Tradingowy")
st.markdown("Aplikacja oparta na strategii Trend-Following (EMA 20/50) z Trailing Stop-Loss.")

# Panel boczny (Sidebar) - parametry
st.sidebar.header("Ustawienia")
ticker = st.sidebar.selectbox("Wybierz aktywo:", ["BTC-USD", "ETH-USD", "SOL-USD", "SPY", "GLD"])
start_date = st.sidebar.date_input("Data startowa:", pd.to_datetime("2021-01-01"))
stop_loss_pct = st.sidebar.slider("Trailing Stop-Loss (%)", min_value=0.5, max_value=5.0, value=1.5, step=0.1) / 100.0

# Pobieranie danych (cache, żeby nie pobierać ich przy każdym kliknięciu)
@st.cache_data
def load_data(ticker, start):
    df = yf.download(ticker, start=start, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

df = load_data(ticker, start_date)

if df.empty:
    st.error("Brak danych dla wybranego aktywa.")
    st.stop()

# Obliczanie wskaźników
df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()

# Symulacja strategii (dla wykresu i metryk)
position = 0
cash = 10000.0
crypto = 0.0
max_price = 0.0
equity_curve = []
trades = []

for i in range(len(df)):
    price = df["Close"].iloc[i]
    ema20 = df["ema20"].iloc[i]
    ema50 = df["ema50"].iloc[i]
    date = df.index[i]
    
    current_signal = 1 if ema20 > ema50 else 0
    
    # Logika sprzedaży
    if position == 1:
        if price > max_price: max_price = price
        
        trend_reversal = current_signal == 0
        stop_loss_hit = price <= max_price * (1 - stop_loss_pct)
        
        if trend_reversal or stop_loss_hit:
            cash = crypto * price
            crypto = 0.0
            position = 0
            profit = ((cash / 10000.0) - 1) * 100
            reason = "Stop Loss" if stop_loss_hit else "Trend"
            trades.append({"Data": date, "Akcja": "SPRZEDAJ", "Cena": price, "Zysk (%)": profit, "Powód": reason})
            
    # Logika kupna
    elif position == 0 and current_signal == 1:
        crypto = cash / price
        cash = 0.0
        position = 1
        max_price = price
        trades.append({"Data": date, "Akcja": "KUP", "Cena": price, "Zysk (%)": 0.0, "Powód": "EMA"})
        
    equity_curve.append(cash + (crypto * price))

df["Equity"] = equity_curve
df["BuyHold"] = 10000.0 * (df["Close"] / df["Close"].iloc[0])

# Obliczanie metryk
def calc_metrics(equity):
    returns = equity.pct_change().dropna()
    cagr = (equity.iloc[-1] / 10000.0) ** (365 / len(equity)) - 1
    mdd = -((equity / equity.cummax()) - 1.0).min()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(365) if returns.std() > 0 else 0
    return cagr, mdd, sharpe

cagr, mdd, sharpe = calc_metrics(df["Equity"])

# --- WYŚWIETLANIE W PRZEGLĄDARCE ---

# 1. Metryki w kolumnach
col1, col2, col3, col4 = st.columns(4)
col1.metric("Końcowy Kapitał", f"${df['Equity'].iloc[-1]:,.2f}")
col2.metric("CAGR", f"{cagr:.2%}")
col3.metric("Max Drawdown", f"{mdd:.2%}")
col4.metric("Sharpe Ratio", f"{sharpe:.2f}")

# 2. Wykres świecowy z EMA i znacznikami transakcji
fig = go.Figure()

# Świece
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Cena"))

# EMA
fig.add_trace(go.Scatter(x=df.index, y=df["ema20"], line=dict(color='blue', width=1), name="EMA 20"))
fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], line=dict(color='orange', width=1), name="EMA 50"))

# Znaczniki Kupna i Sprzedaży
trades_df = pd.DataFrame(trades)
if not trades_df.empty:
    buys = trades_df[trades_df["Akcja"] == "KUP"]
    sells = trades_df[trades_df["Akcja"] == "SPRZEDAJ"]
    
    fig.add_trace(go.Scatter(x=buys["Data"], y=buys["Cena"], mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'), name="Kupno"))
    fig.add_trace(go.Scatter(x=sells["Data"], y=sells["Cena"], mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name="Sprzedaż"))

fig.update_layout(title=f"Wykres {ticker} z sygnałami", xaxis_rangeslider_visible=False, template="plotly_dark", height=600)
st.plotly_chart(fig, use_container_width=True)

# 3. Krzywa kapitału
fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(x=df.index, y=df["Equity"], line=dict(color='green', width=2), name="Strategia"))
fig_eq.add_trace(go.Scatter(x=df.index, y=df["BuyHold"], line=dict(color='gray', width=1, dash='dash'), name="Buy & Hold"))
fig_eq.update_layout(title="Krzywa kapitału (Equity Curve)", template="plotly_dark", height=400)
st.plotly_chart(fig_eq, use_container_width=True)

# 4. Tabela transakcji
st.subheader("Historia transakcji")
if not trades_df.empty:
    st.dataframe(trades_df.sort_values(by="Data", ascending=False), use_container_width=True)
else:
    st.info("Brak transakcji w wybranym okresie.")