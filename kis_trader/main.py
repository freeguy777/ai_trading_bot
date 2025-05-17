# main.py
import mojito # KIS 라이브러리 import (이전 코드에서는 koreainvestment 였으나, 여기서는 mojito로 되어있어 그대로 사용)
import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional, Union
from bs4 import BeautifulSoup
import urllib.request as req
# .env 파일에서 환경 변수 로드
load_dotenv(dotenv_path="config/.env")  # 상대 경로 사용

# 환경 변수에서 설정 값 로드
APP_KEY = os.getenv("KIS_API_KEY")
APP_SECRET = os.getenv("KIS_API_SECRET")
ACCOUNT_NO_FULL = os.getenv("KIS_INVEST_ACCOUNT_NO_FULL") # 라이브러리는 'XXXXXXXX-XX' 형식 필요

# .env 파일 로드 확인
if not all([APP_KEY, APP_SECRET, ACCOUNT_NO_FULL]):
    raise ValueError("오류: .env 파일에 필요한 모든 환경 변수(KIS_API_KEY, KIS_API_SECRET, KIS_INVEST_ACCOUNT_NO_FULL)를 설정해야 합니다.")

def get_current_price(
    ticker: str,
    exchange_name: str = "나스닥",
) -> Union[float, None]:
    """
    mojito 브로커를 이용해 실시간(또는 직전 체결) 가격을 가져온다.
    반환: 가격(float) | None  (호출 실패 시 None)
    """
    try:
        price_info = _get_broker(exchange_name).fetch_price(ticker)
        return float(price_info["output"]["last"])
    except Exception as e:
        print(f"⚠️  get_current_price 실패: {e}")
        return None
    
#def get_stock_quantity(symbol: str, exchange_name: str) -> int:




# --- 외부 호출용 주문 실행 함수 ---
def execute_kis_order(ticker: str, quantity: int, order_type: str, action: str, exchange_name: str,price: Optional[int] = None) -> bool:
    """
    한국투자증권 API를 사용하여 해외주식 주문을 실행합니다.

    Args:
        ticker (str): 주문할 종목의 티커 (예: "TSLA")
        quantity (int): 주문 수량
        order_type (str): 주문 유형 ("시장가" 또는 "지정가")
        action (str): 주문 동작 ("매수" 또는 "매도")
        exchange_name (str): 거래소 한글 이름 (예: "나스닥", "뉴욕" 등 mojito 라이브러리 기준)
        price (int, optional): 지정가 주문 시 가격. 시장가 주문 시 무시됩니다. Defaults to 0.

    Returns:
        bool: 주문 성공 시 True, 실패 시 False
    """
    print(f"\n=== KIS 주문 요청 시작 ===")
    print(f"티커: {ticker}, 수량: {quantity}, 유형: {order_type}, 동작: {action}, 거래소: {exchange_name}, 가격: {price if order_type == '지정가' else 'N/A'}")
    print("==========================")

    # 입력값 검증
    if order_type not in ["시장가", "지정가"]:
        print(f"오류: 유효하지 않은 주문 유형입니다: '{order_type}'. '시장가' 또는 '지정가'를 사용하세요.")
        return False
    if action not in ["매수", "매도"]:
        print(f"오류: 유효하지 않은 주문 동작입니다: '{action}'. '매수' 또는 '매도'를 사용하세요.")
        return False
    # 지정가 주문이라면 price가 반드시 전달되어야 하고, 0 초과여야 함
    if order_type == "지정가" and (price is None or price <= 0):
        print(f"오류: 지정가 주문 시 price 파라미터를 양수로 전달해야 합니다. 입력된 값: {price}")
        return False
    # exchange_name은 mojito 라이브러리 내부에서 처리하므로 여기서는 기본 검증 생략

    try:
        # KoreaInvestment 클래스 인스턴스 생성 (실전 투자)
        broker = mojito.KoreaInvestment(
            api_key=APP_KEY,
            api_secret=APP_SECRET,
            acc_no=ACCOUNT_NO_FULL,
            exchange=exchange_name,
            mock=False # 실전 투자
        )
        print("KIS Broker 객체 초기화 완료.")

        order_result = None
        # 주문 실행
        price_raw=broker.fetch_price(ticker)['output']['last']
        current_price = round(float(price_raw), 2)
        
        if action == "매수":
            if order_type == "시장가" or "지정가":
                print(f"시장가[{current_price}] 매수 주문 전송 시도...")
                order_result = broker.create_oversea_order("buy", symbol=ticker, price = current_price, quantity=quantity, order_type="00")
        elif action == "매도":
            if order_type == "시장가" or "지정가":
                print(f"시장가[{current_price}] 매도 주문 전송 시도...")
                order_result = broker.create_oversea_order("sell", symbol=ticker, price = current_price, quantity=quantity, order_type="00")

        # 결과 처리
        print("--- KIS 주문 결과 ---")
        print(json.dumps(order_result, indent=2, ensure_ascii=False))
        print("---------------------")

        if order_result and order_result.get("rt_cd") == "0": # KIS API 성공 코드 '0'
            print("주문 성공 (rt_cd == '0').")
            return True
        else:
            error_msg = order_result.get('msg1', '알 수 없는 오류')
            error_code = order_result.get('msg_cd', 'N/A')
            print(f"주문 실패: {error_msg} (에러 코드: {error_code})")
            return False

    except Exception as e:
        print(f"주문 처리 중 예외 발생: {e}")
        # 필요 시 상세 오류 로깅 추가 (예: traceback)
        # import traceback
        # print(traceback.format_exc())
        return False

def get_exchange_rate(exchange_name: str = "나스닥") -> float:
    """
    현재 기준환율을 소수로 반환합니다.
    """
    url = "https://finance.naver.com/marketindex"
    res = req.urlopen(url)
    soup = BeautifulSoup(res, "html.parser")
    price = soup.select_one("a.head.usd > div.head_info > span.value").string
    price = price.replace(",", "")
    return price

def _get_broker(exchange_name: str = "나스닥") -> mojito.KoreaInvestment:
    """KIS 브로커 인스턴스를 반환합니다."""
    return mojito.KoreaInvestment(
        api_key=APP_KEY,
        api_secret=APP_SECRET,
        acc_no=ACCOUNT_NO_FULL,
        exchange=exchange_name,
        mock=False
    )

def get_total_cash_krw(exchange_name: str = "나스닥") -> float:
    """
    내 전체 예수금(원)을 소수로 반환합니다.
    """
    balance = _get_broker(exchange_name).fetch_present_balance()
    return balance['output3']['tot_frcr_cblc_smtl']

def get_total_cash_usd(exchange_name: str = "나스닥") -> float:
    """
    내 전체 예수금(달러)을 소수로 반환합니다.
    """
    balance = float(get_total_cash_krw(exchange_name="나스닥")) / float(get_exchange_rate(exchange_name="나스닥"))
    return balance

def get_present_balance(exchange_name: str = "나스닥") -> float:
    """
    내 전체 외화잔고 합계를 소수로 반환합니다.
    """
    balance = _get_broker(exchange_name).fetch_present_balance()
    return balance['output2'][0]['frcr_drwg_psbl_amt_1']

def get_average_price(ticker: str, exchange_name: str = "나스닥") -> float:
    """
    특정 종목의 평균 매수가격을 반환합니다.
    (purchase_cost = amount - profit, avg_price = purchase_cost / qty)
    """
    balance = _get_broker(exchange_name).fetch_balance()
    
    for item in balance["output1"]:
    # 2) 해당 항목의 종목 코드(pdno)가 "TSLA"이면
        if item.get("ovrs_pdno") == ticker:
            # 3) 문자열을 실수(float)로 변환해 저장
            avg_price = float(item["pchs_avg_pric"])
            return round(avg_price, 2)

    return -1

def get_purchase_amount(ticker: str, exchange_name: str = "나스닥") -> int:
    """
    특정 종목의 해당 종목의 외화 기준 매입금액을 반환합니다.
    """
    balance = _get_broker(exchange_name).fetch_balance()
    
    for item in balance["output1"]:
    # 2) 해당 항목의 종목 코드(pdno)가 "TSLA"이면
        if item.get("ovrs_pdno") == ticker:
            # 3) 문자열을 실수(float)로 변환해 저장
            avg_unpr3 = item["frcr_pchs_amt1"]
            return avg_unpr3

    return -1

def get_holding_amount(ticker: str, exchange_name: str = "나스닥") -> int:
    """
    특정 종목의 보유 수량을 반환합니다.
    """
    balance = _get_broker(exchange_name).fetch_balance()
    
    for item in balance["output1"]:
    # 2) 해당 항목의 종목 코드(pdno)가 "TSLA"이면
        if item.get("ovrs_pdno") == ticker:
            # 3) 문자열을 실수(float)로 변환해 저장
            avg_unpr3 = item["ovrs_cblc_qty"]
            return avg_unpr3

    return -1


def get_order_sell_quantity(ticker: str, exchange_name: str = "나스닥") -> int:
    """
    특정 종목의 매도가능 수량을 반환합니다.
    """
    balance = _get_broker(exchange_name).fetch_balance()
    
    for item in balance["output1"]:
    # 2) 해당 항목의 종목 코드(pdno)가 "TSLA"이면
        if item.get("ovrs_pdno") == ticker:
            # 3) 문자열을 실수(float)로 변환해 저장
            avg_unpr3 = item["ord_psbl_qty"]
            return avg_unpr3

    return -1

def get_account_profit_rate() -> float | None:
    """
    한국투자증권 API를 사용하여 지정된 계좌의 해외 주식 총 수익률을 조회합니다.
    올해 1월 1일부터 현재까지의 수익률을 조회합니다.

    Args:
        access_token: 발급받은 접근 토큰 (Bearer 제외).
        
    Returns:
        조회된 총 수익률 (float, 예: 10.5는 10.5%). 오류 발생 시 None.
    """
    token_for_header = None
    try:
        # 0. mojito 브로커 인스턴스를 통해 Access Token 가져오기
        broker = _get_broker(exchange_name="나스닥")  # 토큰 발급/로드를 위해 broker 객체 생성

        if hasattr(broker, 'access_token') and broker.access_token:
            token_for_header = broker.access_token  # "Bearer <token_value>" 형태
            # print(f"Debug: Retrieved token from broker: {token_for_header[:20]}...") # 토큰 일부 출력 (디버깅용)
        else:
            print("Error: 'access_token' attribute not found or is empty in mojito broker object.")
            print("mojito 라이브러리가 정상적으로 초기화되었는지 확인하세요. (예: token.dat 파일, API 키/시크릿 문제)")
            return None

    except Exception as e_token_init:
        print(f"Error: Access Token 가져오는 중 또는 broker 초기화 중 예외 발생: {e_token_init}")
        traceback.print_exc()
        return None

    if not token_for_header:
        # 위의 else 블록에서 이미 처리되었어야 하지만, 안전장치로 둡니다.
        print("Error: Failed to retrieve a valid token_for_header from broker.")
        return None

    # 1. 조회 기간 설정
    now = datetime.now()
    current_year = now.year
    inqr_strt_dt = f"{current_year}0101"
    inqr_end_dt = now.strftime("%Y%m%d")

    # 2. API 엔드포인트 및 헤더 설정
    base_url = "https://openapi.koreainvestment.com:9443" # 실서버 URL
    path = "/uapi/overseas-stock/v1/trading/inquire-period-profit"
    url = f"{base_url}{path}"

    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": token_for_header,  # broker.access_token (이미 "Bearer " 포함)을 직접 사용
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "TTTS3039R", # 실전 TR_ID
        "custtype": "P",      # 개인 고객
    }

    # 3. 요청 파라미터 설정
    try:
        account_prefix = ACCOUNT_NO_FULL.split('-')[0]
        account_suffix = ACCOUNT_NO_FULL.split('-')[1]
    except IndexError:
        print(f"오류: ACCOUNT_NO_FULL ('{ACCOUNT_NO_FULL}') 형식이 '계좌번호-상품코드'가 아닙니다.")
        return None

    params = {
        "CANO": account_prefix,
        "ACNT_PRDT_CD": account_suffix,
        "OVRS_EXCG_CD": "NASD", # 예: 나스닥 (필요시 이 값을 변경하거나 인자로 받도록 수정)
        "NATN_CD": "840",       # 예: 미국 (ISO 3166-1 Numeric)
        "CRCY_CD": "USD",       # 예: USD
        "PDNO": "",             # 상품번호 (공란 시 전체 가능성, API 명세 확인 필요)
        "INQR_STRT_DT": inqr_strt_dt,
        "INQR_END_DT": inqr_end_dt,
        "WCRC_FRCR_DVSN_CD": "02", # 원화('02') 기준 (API 명세에 따름)
        "CTX_AREA_FK200": "",
        "CTX_AREA_NK200": ""
    }

    #print(f"Requesting URL (TTTS3039R): {url}")
    #print(f"Params (TTTS3039R): {params}")

    # 4. HTTP GET 요청
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        #print(f"API Response Data (TTTS3039R): {json.dumps(data, indent=2, ensure_ascii=False)}")

        if data.get("rt_cd") != "0":
            error_msg = data.get("msg1", "알 수 없는 API 오류")
            print(f"API Error (TTTS3039R) (rt_cd: {data.get('rt_cd')}): {error_msg} (msg_cd: {data.get('msg_cd')})")
            return None

        if "output2" in data and data["output2"] and "tot_pftrt" in data["output2"]:
            profit_rate_str = data["output2"]["tot_pftrt"]
            if profit_rate_str and profit_rate_str.strip():
                try:
                    return float(profit_rate_str)
                except ValueError:
                    print(f"Error (TTTS3039R): 'tot_pftrt' 값 '{profit_rate_str}'을 float으로 변환할 수 없습니다.")
                    return None
            else:
                print("API 응답(TTTS3039R)에 'tot_pftrt' 필드가 비어있거나 유효하지 않습니다.")
                return None
        else:
            print("API 응답(TTTS3039R)에서 'output2' 또는 'tot_pftrt' 필드를 찾을 수 없습니다.")
            return None

    except requests.exceptions.Timeout:
        print(f"HTTP 요청 시간 초과 (TTTS3039R): {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"HTTP 요청 실패 (TTTS3039R): {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"HTTP Error Response (TTTS3039R): {e.response.status_code} - {e.response.text}")
            except Exception:
                print(f"HTTP Error Response status (TTTS3039R): {e.response.status_code}")
        return None
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"응답 파싱 또는 키 접근 오류 (TTTS3039R): {e}")
        if 'response' in locals() and response is not None:
             print(f"Response text (TTTS3039R): {response.text}")
        return None
    except Exception as e:
        print(f"알 수 없는 오류 발생 in get_total_overseas_profit_rate (TTTS3039R): {e}")
        traceback.print_exc()
        return None
    
# --- 테스트용 실행 로직 ---
if __name__ == "__main__":
    print("\n--- execute_kis_order 함수 테스트 시작 ---")
    #success1 = execute_kis_order("QQQ", 1, "시장가", "매수", "나스닥")
    #print(f"테스트 1 결과: {'성공' if success1 else '실패'}")
    #print("-" * 30)

    sample_ticker = "NVDA"  # 테스트할 종목 티커로 변경 가능
    profit_rate = get_account_profit_rate()                                         #전체 계좌 수익률
    total_cash_withholdings_krw = get_total_cash_krw(exchange_name="나스닥")        #총 외화 잔고 합계
    total_cash_withholdings_usd = get_total_cash_usd(exchange_name="나스닥")        #총 외화 잔고 합계
    cash_balance = get_present_balance(exchange_name="나스닥")                      #외화 남은 잔고 합계
    average_price = get_average_price(sample_ticker, exchange_name="나스닥")        #평단가
    purchase_amount = get_purchase_amount(sample_ticker, exchange_name="나스닥")    #해당 종목의 외화 기준 매입금액
    current_price = get_current_price(sample_ticker, exchange_name="나스닥")        #특정 보유 종목의 현재 가격
    holding_amount = get_holding_amount(sample_ticker, exchange_name="나스닥")      #보유 수량
    sell_quantity = get_order_sell_quantity(sample_ticker, exchange_name="나스닥")  #매도 가능 수량
    exchange_rate = get_exchange_rate(exchange_name="나스닥")                       #기준 환율

    print(f"[전체수익률] {profit_rate}")
    print(f"[총 예수금(KRW)] {total_cash_withholdings_krw}")
    print(f"[총 예수금(USD)] {total_cash_withholdings_usd}")
    print(f"[가용 예수금] {cash_balance}")
    print(f"[기준환율] {exchange_rate}")
    print(f"[평단가] {sample_ticker}: {average_price}")
    print(f"[매입금액] {sample_ticker}: {purchase_amount}")
    print(f"[현재 가격] {sample_ticker}: {current_price}")
    print(f"[보유 수량] {sample_ticker}: {holding_amount}")
    print(f"[매도 가능 수량] {sample_ticker}: {sell_quantity}")
    
    #print(f'[보유 수량] {sample_ticker}: {get_stock_quantity(symbol=sample_ticker, exchange_name="나스닥")}')
    print("--- 잔고 조회 테스트 종료 ---")

    '''
    # 테스트 케이스 1: 시장가 매수 (성공 예상)
    print("\n[테스트 1] 시장가 매수 (TQQQ 1주)")
    # success1 = execute_kis_order(ticker="TQQQ", quantity=1, order_type="시장가", action="매수", exchange_name="나스닥") # 원본
    success1 = execute_kis_order("TQQQ", 1, "시장가", "매수", "나스닥") # 한 줄 수정 (price는 기본값 0 사용)
    print(f"테스트 1 결과: {'성공' if success1 else '실패'}")
    print("-" * 30)

    # 테스트 케이스 2: 지정가 매도 (성공/실패는 가격에 따라 다름)
    print("\n[테스트 2] 지정가 매도 (TQQQ 1주, 가격: 50)")
    # success2 = execute_kis_order(ticker="TQQQ", quantity=1, order_type="지정가", action="매도", exchange_name="나스닥", price=50) # 원본
    success2 = execute_kis_order("TQQQ", 1, "지정가", "매도", "나스닥", 50) # 한 줄 수정 (price 명시적 전달)
    print(f"테스트 2 결과: {'성공' if success2 else '실패'}")
    print("-" * 30)

    # 테스트 케이스 3: 유효하지 않은 주문 유형 (실패 예상)
    print("\n[테스트 3] 유효하지 않은 주문 유형")
    # success3 = execute_kis_order(ticker="AAPL", quantity=1, order_type="조건부지정가", action="매수", exchange_name="나스닥") # 원본
    success3 = execute_kis_order("AAPL", 1, "조건부지정가", "매수", "나스닥") # 한 줄 수정
    print(f"테스트 3 결과: {'성공' if success3 else '실패'} (예상: 실패)")
    print("-" * 30)

    # 테스트 케이스 4: 지정가 주문 시 가격 누락 (실패 예상)
    print("\n[테스트 4] 지정가 주문 시 가격 누락")
    # success4 = execute_kis_order(ticker="MSFT", quantity=1, order_type="지정가", action="매수", exchange_name="나스닥", price=0) # 원본
    success4 = execute_kis_order("MSFT", 1, "지정가", "매수", "나스닥", 0) # 한 줄 수정 (price=0 명시적 전달)
    print(f"테스트 4 결과: {'성공' if success4 else '실패'} (예상: 실패)")
    print("-" * 30)

    print("\n--- 함수 테스트 종료 ---")
    '''