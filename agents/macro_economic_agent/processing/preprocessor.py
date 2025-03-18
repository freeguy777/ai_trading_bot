# agents/macro_economic_agent/processing/preprocessor.py
from __future__ import annotations
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """
    Merge price and macro frames → clean feature matrix.
    """

    def __init__(
        self,
        log_cols: List[str] | None = None,
        zscore_cols: List[str] | None = None,
    ):
        self.log_cols = log_cols or []
        self.zscore_cols = zscore_cols or []
        self.scaler: StandardScaler | None = None

    # ------------------------------------------------------------------ #
    
    def _flatten(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure columns are single‑level *and* 'date' is a column."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
            df.columns.name = None

        if '' in df.columns:
            df = df.rename(columns={'': 'date'})

        if "date" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex) or df.index.name == "date":
                df = df.reset_index()
                if "index" in df.columns and "date" not in df.columns:
                    df.rename(columns={"index": "date"}, inplace=True)

            if "date" not in df.columns:
                    raise ValueError(f"Failed to ensure 'date' column exists in DataFrame after flattening. Columns: {df.columns}, Index name: {df.index.name}")

        return df



    def _align_merge(self, frames: List[pd.DataFrame]) -> pd.DataFrame:
        out = frames[0].copy()
        for df in frames[1:]:
            out = out.merge(df, on="date", how="outer")
        out.sort_values("date", inplace=True)
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    def transform(
        self,
        price_df: pd.DataFrame,
        macro_df: pd.DataFrame,
    ) -> pd.DataFrame:
        # 1) 날짜 컬럼 표준화 -----------------------------------
        price_df["date"] = pd.to_datetime(price_df["date"]).dt.tz_localize(None)
        macro_df["date"] = pd.to_datetime(macro_df["date"]).dt.tz_localize(None)

        # 2) 병합 ---------------------------------------------
        price_df  = self._flatten(price_df)
        macro_df  = self._flatten(macro_df)
        df = self._align_merge([price_df, macro_df])

        # 3) 결측치 보간 --------------------------------------
        df.interpolate(method="linear", inplace=True, limit_direction="both")

        # 4) 로그변환 -----------------------------------------
        for c in self.log_cols:
            if c in df.columns:
                df[c] = np.log(df[c].replace(0, np.nan))

        # 5) Z‑score 정규화 -----------------------------------
        if self.zscore_cols:
            self.scaler = StandardScaler()
            df[self.zscore_cols] = self.scaler.fit_transform(df[self.zscore_cols])

        return df



# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # 데모용 간단 테스트 (실제 파이프라인에서는 import 후 호출)
    import os
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    agents_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(agents_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from macro_economic_agent.ingestion.market_data import get_price_series_yf, get_macro_series_fred
    from macro_economic_agent.ingestion.news_fetcher import NewsFetcher
    from macro_economic_agent.processing.sentiment_analyzer import SentimentAnalyzer
    from macro_economic_agent.config_loader import ConfigLoader

    cfg = ConfigLoader.load()

    price = get_price_series_yf(["SPY", "TLT", "GLD", "USO", "UUP"], "2024-01-01")
    macro = get_macro_series_fred(["GDP", "CPIAUCSL", "FEDFUNDS"])
    news  = NewsFetcher(["SPY", "TLT"]).fetch_news(limit_per_ticker=3)
    #senti = SentimentAnalyzer(cfg["GEMINI_API_KEY"]).analyze_sentiment(news)

    proc  = DataPreprocessor(
        log_cols=["SPY", "TLT", "CPIAUCSL"],
        zscore_cols=["GDP", "FEDFUNDS"],
    )
    feat = proc.transform(price, macro)
    print(feat.tail(30))
