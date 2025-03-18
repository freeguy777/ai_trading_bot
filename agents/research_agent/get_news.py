import os
import requests
import json
from datetime import datetime, timedelta

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# fetch_news 함수가 api_key를 인자로 받도록 수정
def fetch_news(ticker, start_date, end_date, api_key): # api_key 파라미터 추가
    """
    특정 주식 티커와 날짜 범위에 대한 뉴스를 Alpha Vantage API를 사용하여 가져옵니다.
    API 키를 파라미터로 받습니다.
    """
    # 내부에서 os.getenv 호출 제거
    # if not api_key: # 검사는 호출하는 쪽(main)에서 수행
    #     print("Alpha Vantage API 키가 없어 뉴스를 가져올 수 없습니다.")
    #     return []

    try:
        start_dt_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt_obj = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1) - timedelta(microseconds=1)
        time_from = start_dt_obj.strftime("%Y%m%dT%H%M")
        time_to = end_dt_obj.strftime("%Y%m%dT%H%M")
    except ValueError:
        print(f"날짜 형식 오류: start_date='{start_date}', end_date='{end_date}'. 'YYYY-MM-DD' 형식을 사용하세요.")
        return []

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": time_from,
        "time_to": time_to,
        "limit": 100,
        "apikey": api_key # 전달받은 api_key 사용
    }

    try:
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if "Information" in data or "Error Message" in data:
            print(f"Alpha Vantage API 오류: {data.get('Information', data.get('Error Message', '알 수 없는 오류'))}")
            return []

        news_results = []
        if "feed" in data and isinstance(data["feed"], list):
            for article in data["feed"]:
                date_str = article.get("time_published", "")
                date_obj = None
                formatted_date = date_str
                if date_str:
                    try:
                        date_obj = datetime.strptime(date_str, "%Y%m%dT%H%M%S")
                        formatted_date = date_obj.strftime("%Y-%m-%d")
                    except ValueError as e:
                        print(f"Alpha Vantage 날짜 파싱 오류 '{date_str}': {e}")

                if date_obj and date_obj > end_dt_obj:
                    continue

                news_results.append({
                    "title": article.get("title", ""),
                    "link": article.get("url", ""),
                    "snippet": article.get("summary", ""),
                    "date": formatted_date,
                    "source": article.get("source", ""),
                    "date_obj": date_obj,
                    "overall_sentiment_score": article.get("overall_sentiment_score"),
                    "overall_sentiment_label": article.get("overall_sentiment_label")
                })

            if news_results:
                news_results = sorted(
                    news_results,
                    key=lambda x: x.get("date_obj") if x.get("date_obj") else datetime.min,
                    reverse=True
                )
        else:
             # API 호출은 성공했으나 feed가 없는 경우 (예: 'Information' 키만 있음)
             if not ("Information" in data or "Error Message" in data):
                print(f"{ticker}: Alpha Vantage에서 해당 기간의 뉴스 피드를 찾을 수 없습니다.")


        return news_results

    except requests.exceptions.RequestException as e:
        print(f"Alpha Vantage 뉴스 API 요청 오류: {e}")
        return []
    except json.JSONDecodeError:
        print(f"Alpha Vantage 뉴스 API 응답 JSON 디코딩 오류")
        return []
    except Exception as e:
        print(f"뉴스 처리 중 예기치 않은 오류 발생: {e}")
        return []


# fetch_finance_data 함수가 api_key를 인자로 받도록 수정
def fetch_finance_data(ticker, api_key): # api_key 파라미터 추가
    """
    Alpha Vantage API를 사용하여 특정 주식 티커에 대한 회사 개요 정보를 가져옵니다.
    API 키를 파라미터로 받습니다.
    """
    # 내부에서 os.getenv 호출 제거 및 디버깅 코드 제거
    # if not api_key: # 검사는 호출하는 쪽(main)에서 수행
    #     print("Alpha Vantage API 키가 없어 재무 데이터를 가져올 수 없습니다.")
    #     return {}

    params = {
        "function": "OVERVIEW",
        "symbol": ticker,
        "apikey": api_key # 전달받은 api_key 사용
    }

    try:
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if not data or "Information" in data or "Error Message" in data:
            error_msg = data.get('Information', data.get('Error Message', '회사 정보를 찾을 수 없거나 API 오류 발생'))
            # 빈 응답 {} 도 여기에 해당될 수 있음
            if not data:
                 print(f"{ticker}: Alpha Vantage에서 빈 응답을 받았습니다. (회사 개요 정보 없음)")
            else:
                 print(f"Alpha Vantage API 오류 ({ticker}): {error_msg}")

            # Overview의 경우, 유효하지 않은 티커 등에 대해 빈 객체 {}를 반환하는 경우가 정상일 수 있음
            # 따라서 단순히 {}를 반환
            return {}

        return data

    except requests.exceptions.RequestException as e:
        print(f"Alpha Vantage 재무 데이터 API 요청 오류: {e}")
        return {}
    except json.JSONDecodeError:
        print(f"Alpha Vantage 재무 데이터 API 응답 JSON 디코딩 오류")
        return {}
    except Exception as e:
        print(f"재무 데이터 처리 중 예기치 않은 오류 발생: {e}")
        return {}