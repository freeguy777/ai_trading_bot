import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

#####절대 경로 임포트 사용
current_dir = os.path.dirname(os.path.abspath(__file__))
agents_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(agents_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.research_agent.get_news import fetch_news, fetch_finance_data
from agents.research_agent.analysis_news import (
    analyze_news_sentiment,
    analyze_stock_impact,
    generate_financial_report,
    # generate_sentiment_trend_chart, # generate_financial_report 내부에서 호출됨
    # summarize_news_by_sentiment # generate_financial_report 내부에서 호출됨
)

# .env 파일에서 환경 변수 로드
load_dotenv()

def api_key_init():
    """
    환경 변수에서 API 키를 가져오는 함수
    
    Returns:
        dict: API 키와 관련 정보를 포함하는 딕셔너리
    """
    alphavantage_api_key = os.getenv("ALPHAVENTAGE_API")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    # 경고 메시지 검사 및 출력
    if not alphavantage_api_key:
        print("경고: ALPHAVANTAGE_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    if not gemini_api_key:
        print("경고: GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        print("Gemini API 키가 없으면 뉴스 요약 기능이 제한됩니다.")
    
    print(f"Alpha Vantage Key Loaded in api_key_init: {alphavantage_api_key is not None}") # 로드 확인
    
    config = {
        "alphavantage_api_key": alphavantage_api_key,
        "gemini_api_key": gemini_api_key
    }
    
    return config


def analyze_stock(ticker, period_days=10, verbose=False):
    """
    뉴스 및 재무 데이터에 기반하여 주식을 분석하는 주 함수입니다.
    
    Args:
        ticker (str): 주식 티커 기호
        period_days (int): 뉴스를 찾기 위해 되돌아볼 일수
        verbose (bool): 진행 상황 및 결과를 출력할지 여부

    Returns:
        str: 마크다운 형식의 뉴스 분석 보고서
    """
    # API 키 초기화
    config = api_key_init()
    api_key = config["alphavantage_api_key"]
    gemini_key = config["gemini_api_key"]
    
    # 날짜 범위 계산
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=period_days)).strftime("%Y-%m-%d")

    if verbose:
        print(f"{start_date}부터 {end_date}까지 {ticker} 분석 중...")

    # API 키 유효성 검사 (호출 전에 확인)
    if not api_key:
         print("오류: Alpha Vantage API 키가 전달되지 않아 분석을 진행할 수 없습니다.")
         return "오류: Alpha Vantage API 키가 설정되지 않았습니다."

    # 뉴스 데이터 가져오기 (API 키 전달)
    if verbose:
        print("뉴스 데이터 가져오는 중 (Alpha Vantage)...")
    news_articles = fetch_news(ticker, start_date, end_date, api_key=api_key) # api_key 전달
    if not news_articles and verbose:
         print(f"{ticker}: 해당 기간의 뉴스를 찾지 못했거나 가져오는 데 실패했습니다.")
    elif verbose:
         print(f"{len(news_articles)}개의 뉴스 기사를 찾았습니다.")


    # 재무 데이터 가져오기 (API 키 전달)
    if verbose:
        print("회사 개요 정보 가져오는 중 (Alpha Vantage)...")
    finance_data = fetch_finance_data(ticker, api_key=api_key) # api_key 전달
    if not finance_data and verbose:
        print(f"{ticker}: 회사 개요 정보를 가져오는 데 실패했습니다.")
    elif verbose:
        company_name = finance_data.get("Name", "알 수 없음")
        print(f"회사 정보 로드됨: {company_name}")


    # 뉴스 감성 분석
    if verbose:
        print("뉴스 감성 분석 중...")
    news_with_sentiment, sentiment_summary = analyze_news_sentiment(news_articles)

    # 잠재적 주식 영향 분석
    if verbose:
        print("잠재적 주식 영향 분석 중...")
    impact_analysis = analyze_stock_impact(sentiment_summary, finance_data)

    # 뉴스 분석 보고서 생성
    if verbose:
        print("뉴스 분석 보고서 생성 중...")
    report = generate_financial_report(
        ticker,
        news_with_sentiment,
        sentiment_summary,
        impact_analysis,
        finance_data,
        start_date,
        end_date,
        gemini_key=gemini_key # 필요 시 전달
    )

    if verbose:
        print("\n========== 전체 보고서 ==========\n")
        print(report)
        print("\n=================================\n")

    return report

# 사용 예시
if __name__ == "__main__":
    ticker = input("주식 티커 기호 입력 (예: AAPL): ").upper()
    try:
        period_days = int(input("분석 기간(일) 입력 (기본값: 30): "))
        if period_days <= 0:
             period_days = 30
             print("분석 기간은 1일 이상이어야 합니다. 기본값 30일로 설정합니다.")
    except ValueError:
        period_days = 30
        print("잘못된 입력입니다. 기본값 30일로 설정합니다.")

    # API 키 초기화 검사 없이 바로 analyze_stock 호출
    report = analyze_stock(ticker, period_days)