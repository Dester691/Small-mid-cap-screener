"""
screener_hype_small_mid_optim.py

Detección de movimientos explosivos en Small/Mid Caps (hype-driven):
 - capitalización 300M – 10B USD
 - histórico reducido: 15 días
 - volumen 2x media, subidas rápidas, gap-up
 - clusters de señales en ventana corta
 - hype social (tweets recientes con $ticker)

Salida: CSV con score total y ranking.
Requisitos:
 pip install yfinance pandas numpy snscrape requests
"""

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import subprocess
import requests
from io import StringIO

# -----------------------------
# Parámetros
# -----------------------------
DAYS_HISTORY = 15
MAX_TICKERS = 200
VOL_MULT = 2.0
RET_1D = 0.10
RET_3D = 0.20
GAP_UP = 0.05
CLUSTER_WIN = 3
CLUSTER_MIN = 2
MCAP_MIN = 3e8
MCAP_MAX = 1e10

# -----------------------------
# Descarga de tickers
# -----------------------------
def get_tickers(limit=MAX_TICKERS):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    tickers = []

    # NASDAQ-100
    try:
        url_nasdaq = "https://en.wikipedia.org/wiki/NASDAQ-100"
        html_nasdaq = requests.get(url_nasdaq, headers=headers, timeout=10).text
        tables_nasdaq = pd.read_html(StringIO(html_nasdaq))
        for t in tables_nasdaq:
            if "Ticker" in t.columns or "Symbol" in t.columns:
                col = "Ticker" if "Ticker" in t.columns else "Symbol"
                tickers += t[col].astype(str).tolist()
                break
    except Exception as e:
        print("⚠️ No se pudo obtener NASDAQ-100:", e)

    # S&P500 (NYSE)
    try:
        url_sp = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        html_sp = requests.get(url_sp, headers=headers, timeout=10).text
        tables_sp = pd.read_html(StringIO(html_sp))
        nyse = tables_sp[0]["Symbol"].astype(str).tolist()
        tickers += nyse
    except Exception as e:
        print("⚠️ No se pudo obtener S&P500:", e)

    tickers = list(dict.fromkeys(tickers))
    tickers = [t.replace(".", "-") for t in tickers]
    if not tickers:
        tickers = ["BYND","GME","AMC","CVNA","PLTR","SPWR"]
        print("⚠️ Wikipedia no accesible. Usando fallback:", tickers)
    return tickers[:limit]

# -----------------------------
# Filtrado Small/Mid Cap
# -----------------------------
def filter_small_mid(tickers):
    filtered = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            cap = info.get("marketCap", 0)
            if MCAP_MIN <= cap <= MCAP_MAX:
                filtered.append(t)
        except Exception:
            continue
    print(f"✅ {len(filtered)} tickers Small/Mid Cap encontrados")
    return filtered

# -----------------------------
# Carga de datos
# -----------------------------
def fetch_data(ticker):
    start_date = (datetime.date.today() - datetime.timedelta(days=DAYS_HISTORY)).isoformat()
    df = yf.download(ticker, start=start_date, end=datetime.date.today().isoformat(),
                     interval="1d", auto_adjust=True, progress=False)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [str(c).lower() for c in df.columns]
    return df.dropna(subset=["close", "volume"])

# -----------------------------
# Señales técnicas
# -----------------------------
def compute_signals(df):
    df = df.copy()
    df["vol_ma"] = df["volume"].rolling(5).mean()
    df["vol_spike"] = df["volume"] > VOL_MULT * df["vol_ma"]
    df["ret_1d"] = df["close"].pct_change()
    df["ret_3d"] = df["close"].pct_change(3)
    df["price_spike"] = (df["ret_1d"] > RET_1D) | (df["ret_3d"] > RET_3D)
    df["gap_up"] = (df["open"] > (1 + GAP_UP) * df["close"].shift(1)) & (df["close"] > df["open"])
    df["signal"] = df[["vol_spike", "price_spike", "gap_up"]].any(axis=1)
    df["cluster"] = df["signal"].rolling(CLUSTER_WIN).sum() >= CLUSTER_MIN
    return df

# -----------------------------
# Menciones sociales
# -----------------------------
def get_tweet_count(ticker, days=2):
    try:
        cmd = f"snscrape --max-results 1000 twitter-search '$${ticker} since:{(datetime.date.today() - datetime.timedelta(days=days)).isoformat()}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        lines = result.stdout.strip().split("\n")
        return len(lines)
    except Exception:
        return 0

# -----------------------------
# Scoring
# -----------------------------
def score(df, tweet_count):
    if df.empty: return 0
    recent = df.tail(CLUSTER_WIN)
    score = (
        recent["vol_spike"].sum() * 2 +
        recent["price_spike"].sum() * 3 +
        recent["gap_up"].sum() * 1.5 +
        int(recent["cluster"].iloc[-1]) * 3
    )
    score += min(tweet_count / 50, 5)
    return round(score, 2)

# -----------------------------
# Screener
# -----------------------------
def run_screener():
    tickers_all = get_tickers()
    tickers = filter_small_mid(tickers_all)
    results = []
    for i, t in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {t} ...", end="\r")
        df = fetch_data(t)
        if df.empty or len(df) < 5:
            continue
        df = compute_signals(df)
        tw = get_tweet_count(t)
        sc = score(df, tw)
        if sc > 0:
            results.append({
                "ticker": t,
                "score": sc,
                "tweets": tw,
                "last_date": df.index[-1].strftime("%Y-%m-%d"),
                "last_close": round(df["close"].iloc[-1], 2)
            })
    out = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    out.to_csv("screener_hype_small_mid.csv", index=False)
    print("\nScreener completado → screener_hype_small_mid.csv")
    return out

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    dfres = run_screener()
    print(dfres.head(20).to_string(index=False))
