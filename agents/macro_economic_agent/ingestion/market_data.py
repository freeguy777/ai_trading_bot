# agents/macro_economic_agent/ingestion/market_data.py
"""
Price & macro data helpers

* get_price_series_yf(symbols, start, end)
    - 일별 종가 시계열 (yfinance)
* get_macro_series_fred(series, start, end, freq)
    - FRED 거시 지표 시계열 (pandas‑datareader)
      freq: 'D'  일, 'ME' 월말, 'MS' 월초, 'Q' 분기
"""

from __future__ import annotations

from datetime import datetime
from typing import List
import time

import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

# ──────────────────────────────────────────────────────────
def get_price_series_yf(
    symbols: List[str],
    start: str = "2020-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download daily prices via yfinance."""
    frames = []
    for sym in symbols:
        try:
            df = yf.download(sym, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                print(f"[WARN] {sym} returned empty frame")
                continue
            df = df[["Close"]].rename(columns={"Close": sym})
            df.index.name = "date"
            frames.append(df)
            time.sleep(1)  # API 호출 사이에 1초 딜레이 추가
        except Exception as e:
            print(f"[ERROR] Failed to download {sym}: {str(e)}")
            continue

    if not frames:
        raise RuntimeError("No price data via yfinance.")

    return pd.concat(frames, axis=1).reset_index()


# ──────────────────────────────────────────────────────────
def get_macro_series_fred(
    series: List[str],
    start: str = "1990-01-01",
    end: str | None = None,
    freq: str = "ME",
) -> pd.DataFrame:
    """
    Download macro series from FRED.
    freq: 'D', 'MS' (month‑start), 'ME' (month‑end), 'Q', etc.
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    frames = []

    for code in series:
        try:
            df = web.DataReader(code, "fred", start, end)
        except Exception as err:
            print(f"[WARN] {code} download failed → {err}")
            continue
        frames.append(df.rename(columns={code: code}))

    if not frames:
        raise RuntimeError("No macro data via FRED.")

    out = pd.concat(frames, axis=1).resample(freq).last().interpolate()
    out.index.name = "date"
    return out.reset_index()


# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    prices = get_price_series_yf(["SPY", "TLT"], start="2024-01-01")
    print("Price\n", prices.head(), sep="")

    macro = get_macro_series_fred(["GDP", "CPIAUCSL", "FEDFUNDS"])
    print("\nMacro\n", macro.tail(), sep="")
