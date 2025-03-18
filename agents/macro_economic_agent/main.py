import os
import sys
import yfinance as yf
import pandas as pd
#####절대 경로 임포트 사용
current_dir = os.path.dirname(os.path.abspath(__file__))
agents_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(agents_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#news_fetcher.py, sentiment_analyzer.py
from agents.macro_economic_agent.config_loader import ConfigLoader
from agents.macro_economic_agent.ingestion.news_fetcher import NewsFetcher
from agents.macro_economic_agent.processing.sentiment_analyzer import SentimentAnalyzer
from agents.macro_economic_agent.config_loader import ConfigLoader
from agents.macro_economic_agent.ingestion.market_data import get_price_series_yf,get_macro_series_fred
from agents.macro_economic_agent.ingestion.geopolitical_events import GeoPoliticalEventsFetcher
from agents.macro_economic_agent.processing.preprocessor import DataPreprocessor
from agents.macro_economic_agent.analysis.simple_analysis import generate_investment_report

def run_macro_economic_analysis() -> str:
    """매크로 경제 분석 에이전트 파이프라인을 실행하고 최종 투자 보고서를 반환합니다."""
    cfg = ConfigLoader.load()
    print("뉴스 수집중...")
    news  = NewsFetcher(["SPY", "TLT", "UUP"]).fetch_news(limit_per_ticker=3)               #뉴스 수집
    print("뉴스 감성 수집중...")
    scored_news = SentimentAnalyzer(cfg["GEMINI_API_KEY"]).analyze_sentiment(news)          #뉴스 감성수집
    print("마켓 데이터 수집중...")
    price = get_price_series_yf(["SPY", "TLT", "GLD", "USO", "UUP"], start="2024-01-01")    #마켓 데이터
    macro = get_macro_series_fred(["GDP", "CPIAUCSL", "FEDFUNDS"])                          #마켓 데이터
    print("지정학 데이터 수집중...")
    events = GeoPoliticalEventsFetcher(cfg).fetch_events()                                  #지정학 데이터 수집
    print("데이터 전처리 중...")
    proc  = DataPreprocessor(                                                               #데이터 전처리
        log_cols=["SPY", "TLT", "CPIAUCSL"],
        zscore_cols=["GDP", "FEDFUNDS"],
    )
    feat = proc.transform(price, macro)                                                     #데이터 전처리
    print("보고서 생성 함수 호출중...")
    report = generate_investment_report(                                                    # LLM 보고서 생성 함수 호출
        feat=feat,
        scored_news=scored_news,
        events=events,
        cfg=cfg
    )
    return report

# 스크립트 직접 실행 시
if __name__ == "__main__":
    # 분석 실행 및 보고서 받기
    final_report = run_macro_economic_analysis()

    # 최종 LLM 보고서 출력
    print("\n" + "="*50 + "\n▶ 최종 투자 분석 보고서\n" + "="*50)
    print(final_report)

    # (선택적) 필요하다면 여기서 중간 결과들을 다시 계산/로드하여 출력할 수 있습니다.
    # 예: cfg = ConfigLoader.load()
    #     news = NewsFetcher(...).fetch_news(...)
    #     print("\n▶ 뉴스 데이터 (샘플)\n", news[:2])











