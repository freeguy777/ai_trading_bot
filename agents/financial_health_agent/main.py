# main.py
import os
import sys
from dotenv import load_dotenv

#####절대 경로 임포트 사용
current_dir = os.path.dirname(os.path.abspath(__file__))
agents_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(agents_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.financial_health_agent.get_financial import get_financial_statements # 재무 데이터 가져오기 모듈 임포트
from agents.financial_health_agent.llm_analysis  import analyze_financials_with_llm# LLM 분석 모듈 임포트

def run_financial_analysis(ticker: str) -> str:
    """
    재무 데이터를 가져오고 분석 리포트를 생성하는 전체 과정을 관리합니다.

    Args:
        ticker (str): 주식 티커 심볼.

    Returns:
        str: 재무 분석 리포트 문자열 또는 오류 메시지.
    """
    # .env 파일에서 API 키 로드
    load_dotenv(dotenv_path="config/.env")  # 상대 경로 사용
    fmp_api_key = os.getenv("FMP_API_KEY")

    if not fmp_api_key:
        return "오류: 환경 변수에서 FMP_API_KEY를 찾을 수 없습니다. .env 파일을 확인하세요."

    # --- 단계 1: 재무 데이터 가져오기 ---
    financial_data = get_financial_statements(ticker, fmp_api_key, years=3)
    if not financial_data:
        return f"오류: 티커 '{ticker}'에 대한 재무 데이터가 없거나 가져올 수 없습니다."

    # --- 단계 2: LLM으로 데이터 분석 ---
    # 참고: LLM API 키(예: OPENAI_API_KEY)는 llm_analysis.py 내부에서 처리하거나,
    # 여기서 로드하여 필요한 경우 함수에 전달할 수 있습니다.
    try:
        analysis_report = analyze_financials_with_llm(financial_data, ticker)
        return analysis_report
    except Exception as e:
        return f"티커 '{ticker}'에 대한 LLM 분석 중 오류 발생: {e}"


if __name__ == "__main__":
    print("--- 재무제표 분석기 ---")
    
    # 사용자로부터 티커 입력 받기
    ticker_input = input("주식 티커 심볼을 입력하세요 (예: AAPL, MSFT): ").strip().upper()

    if not ticker_input:
        print("티커가 입력되지 않았습니다. 종료합니다.")
    else:
        # 분석 실행
        print(f"\n{ticker_input} 분석을 시작합니다...")
        final_report = run_financial_analysis(ticker_input)
        
        # 결과 출력
        print("\n--- 분석 리포트 ---")
        print(final_report)
        print("\n-----------------------")