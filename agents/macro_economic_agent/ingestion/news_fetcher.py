# agents/macro_economic_agent/ingestion/news_fetcher.py
import yfinance as yf
from typing import List, Dict


class NewsFetcher:
    """
    Collects news articles for given tickers via yfinance.
    Robust against malformed records (None, non‑dict, missing fields).
    """

    def __init__(self, tickers: List[str], debug: bool = False) -> None:
        if not tickers:
            raise ValueError("At least one ticker symbol is required.")
        self.tickers = tickers
        self.debug = debug

    # --------------------------------------------------------------------- #
    def fetch_news(self, limit_per_ticker: int = 10) -> List[Dict]:
        """
        Returns
        -------
        List[Dict]
            [
              {ticker, title, publisher, link, published_at},
              ...
            ]
        """
        articles: List[Dict] = []

        for symbol in self.tickers:
            try:
                raw_items = yf.Ticker(symbol).news or []
            except Exception as e:
                print(f"[ERROR] Could not load news list for {symbol}: {e}")
                continue

            # Slice after fetch to avoid IndexError on short lists
            for idx, item in enumerate(raw_items[:limit_per_ticker]):
                try:
                    # ── ① 형태 검증 ──────────────────────────────────────────
                    if not isinstance(item, dict):
                        if self.debug:
                            print(f"[SKIP] {symbol}[{idx}] not dict → {item}")
                        continue

                    content = item.get("content")
                    if not isinstance(content, dict):
                        if self.debug:
                            print(f"[SKIP] {symbol}[{idx}] content not dict/None")
                        continue

                    title = content.get("title")
                    if not title:                       # 제목 없는 건 무시
                        if self.debug:
                            print(f"[SKIP] {symbol}[{idx}] missing title")
                        continue
                    # ───────────────────────────────────────────────────────

                    articles.append(
                        {
                            "ticker": symbol,
                            "title": title,
                            "publisher": content.get("provider", {})
                            .get("displayName", "Unknown"),
                            "link": content.get("clickThroughUrl", {}).get("url", ""),
                            "published_at": content.get("pubDate", ""),
                        }
                    )

                except Exception as rec_err:
                    # 레코드 하나만 실패하고 루프 계속
                    if self.debug:
                        print(f"[SKIP] {symbol}[{idx}] errored: {rec_err}")

        return articles
