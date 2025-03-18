# get_financial.py
import requests
import os
from typing import Dict, List, Optional, Any

def get_financial_statements(ticker: str, api_key: str, years: int = 3) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """
    주어진 티커에 대해 지정된 연수만큼 FMP API에서 연간 손익계산서, 대차대조표, 현금흐름표를 가져옵니다.

    Args:
        ticker (str): 주식 티커 심볼 (예: 'AAPL').
        api_key (str): FMP API 키.
        years (int): 가져올 과거 연간 재무제표의 수. 기본값은 3년.

    Returns:
        Optional[Dict[str, List[Dict[str, Any]]]]: 재무제표('income', 'balance', 'cashflow')를 포함하는 딕셔너리 또는 오류 발생 시 None.
        하나의 재무제표라도 가져오기 실패하면 None을 반환합니다.
    """
    base_url = "https://financialmodelingprep.com/api/v3/"
    statements = {
        "income": "income-statement",          # 손익계산서
        "balance": "balance-sheet-statement", # 대차대조표
        "cashflow": "cash-flow-statement"      # 현금흐름표
    }
    financial_data = {}
    
    print(f"{ticker}에 대한 {years}년 재무 데이터 가져오는 중...")

    for key, statement_type in statements.items():
        url = f"{base_url}{statement_type}/{ticker}?period=annual&limit={years}&apikey={api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()  # 요청 실패 시 (4xx 또는 5xx 상태 코드) 예외 발생
            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                 # FMP는 가끔 요청한 limit보다 적은 연도의 데이터만 있을 경우 더 많이 반환할 수 있으므로, 'years' 만큼만 가져옴
                financial_data[key] = data[:years] 
                print(f"{key.capitalize()} Statement 가져오기 성공.")
            elif isinstance(data, dict) and 'Error Message' in data:
                 # FMP API에서 에러 메시지를 반환한 경우
                 print(f"{ticker}의 {key.capitalize()} Statement 가져오기 오류: {data['Error Message']}")
                 return None
            else:
                # 데이터가 없거나 예상치 못한 형식인 경우
                print(f"경고: {ticker}의 {key.capitalize()} Statement 데이터를 찾을 수 없거나 예상치 못한 형식입니다.")
                financial_data[key] = [] # 시도했음을 나타내기 위해 빈 리스트 추가

        except requests.exceptions.RequestException as e:
            # 네트워크 요청 관련 오류
            print(f"{ticker}의 {key.capitalize()} Statement 가져오기 오류: {e}")
            return None
        except Exception as e:
            # 그 외 예기치 못한 오류
            print(f"{key.capitalize()} Statement 가져오는 중 예기치 못한 오류 발생: {e}")
            return None

    # 모든 재무제표 데이터를 가져왔는지 확인 (빈 리스트 포함)
    if len(financial_data) == len(statements):
        # 최소한 하나의 재무제표에 데이터가 있는지 확인
        if any(len(v) > 0 for v in financial_data.values()):
             print(f"{ticker} 데이터 가져오기 완료.")
             return financial_data
        else:
            # 데이터가 전혀 없는 경우
            print(f"지난 {years}년간 {ticker}의 재무제표 데이터를 찾을 수 없습니다.")
            return None # 데이터 없음을 명확히 하기 위해 None 반환
    else:
        # 일부 재무제표를 가져오지 못한 경우
        print(f"{ticker}의 필요한 모든 재무제표를 가져오지 못했습니다.")
        return None


if __name__ == '__main__':
    # 예제 사용법 (.env 파일에 FMP_API_KEY 필요)
    from dotenv import load_dotenv
    load_dotenv() # .env 파일에서 환경 변수 로드
    test_api_key = os.getenv("FMP_API_KEY")
    test_ticker = "MSFT" # 예제 티커

    if not test_api_key:
        print("오류: .env 파일에서 FMP_API_KEY를 찾을 수 없습니다.")
    else:
        data = get_financial_statements(test_ticker, test_api_key, years=3)
        if data:
            print(f"\n--- {test_ticker} 샘플 데이터 ---")
            for statement_type, statements_list in data.items():
                print(f"\n{statement_type.capitalize()} Statement ({len(statements_list)}년):")
                if statements_list:
                    # 가장 최근 재무제표의 필드 예시 출력
                    print(f"  최근 연도 필드 예시: {list(statements_list[0].keys())[:10]}...") 
                    # 각 연도의 주요 지표 출력
                    if statement_type == 'income' and 'netIncome' in statements_list[0]:
                         for stmt in statements_list:
                             print(f"  {stmt.get('date')}: 순이익 = {stmt.get('netIncome', 'N/A')}")
                    elif statement_type == 'balance' and 'totalAssets' in statements_list[0]:
                         for stmt in statements_list:
                             print(f"  {stmt.get('date')}: 총자산 = {stmt.get('totalAssets', 'N/A')}")
                    elif statement_type == 'cashflow' and 'operatingCashFlow' in statements_list[0]:
                         for stmt in statements_list:
                             print(f"  {stmt.get('date')}: 영업 현금 흐름 = {stmt.get('operatingCashFlow', 'N/A')}")
                else:
                     print("  사용 가능한 데이터가 없습니다.")
        else:
            print(f"\n{test_ticker}의 재무 데이터를 가져올 수 없습니다.")