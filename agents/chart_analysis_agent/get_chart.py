import mojito
import pprint
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv(dotenv_path="config/.env")  # 상대 경로 사용


@dataclass
class KisConfig:
    """ 한국투자증권 API 접속 정보 """
    api_key: str
    api_secret: str
    account_no_full: str
    
def get_foreign_chart_ohlcv(symbol: str, exchange: str, config: KisConfig) -> pd.DataFrame:
    """
    해외 주식 시장의 OHLCV 데이터를 가져오는 함수
    
    Args:
        symbol: 주식 심볼 (예: 'AAPL'은 애플)
        exchange: 거래소 코드 (예: '나스닥', '뉴욕')
        config: KIS API 설정
        
    Returns:
        OHLCV 데이터를 포함한 DataFrame
    """
    broker = mojito.KoreaInvestment(
        api_key=config.api_key,
        api_secret=config.api_secret,
        acc_no=config.account_no_full,
        exchange=exchange  # 거래소 설정
    )

    try:
        # 해외 주식 OHLCV 데이터 가져오기
        resp = broker.fetch_ohlcv_overesea(
            symbol=symbol,
            timeframe='D',  # 일봉 데이터
            end_day="",     # 빈 값이면 오늘까지
            adj_price=True  # 수정주가 반영
        )
        
        # API 응답 처리
        if 'output2' in resp and len(resp['output2']) > 0:
            df = pd.DataFrame(resp['output2'])
            
            # 날짜 칼럼을 datetime으로 변환하고 인덱스로 설정
            # 필드명은 실제 응답에 맞게 조정 필요
            if 'xymd' in df.columns:
                df['date'] = pd.to_datetime(df['xymd'], format="%Y%m%d")
            elif 'ymd' in df.columns:
                df['date'] = pd.to_datetime(df['ymd'], format="%Y%m%d")
            else:
                # 다른 가능한 날짜 필드를 찾아봄
                date_fields = [col for col in df.columns if 'date' in col.lower() or 'ymd' in col.lower()]
                if date_fields:
                    df['date'] = pd.to_datetime(df[date_fields[0]], format="%Y%m%d")
                else:
                    print("날짜 필드를 찾을 수 없습니다.")
                    return pd.DataFrame()
                    
            df.set_index('date', inplace=True)
            
            # 칼럼 이름 매핑 (실제 응답에 맞게 조정 필요)
            price_columns = {
                'open': ['open', 'oprc', 'prev'],
                'high': ['high', 'hgpr', 'hprc'],
                'low': ['low', 'lwpr', 'lprc'],
                'close': ['clos', 'clpr', 'cls', 'last']
            }
            
            result_df = pd.DataFrame(index=df.index)
            
            # 적절한 열 이름 찾기
            for target_col, possible_cols in price_columns.items():
                for col in possible_cols:
                    if col in df.columns:
                        result_df[target_col] = df[col]
                        break
            
            # 모든 필수 칼럼이 있는지 확인
            if all(col in result_df.columns for col in ['open', 'high', 'low', 'close']):
                # float으로 명시적 형변환
                result_df = result_df.astype(float)
                # 날짜순으로 정렬
                result_df = result_df.sort_index(ascending=True)
                result_df.index.name = "date"
                return result_df
            else:
                print("필수 OHLC 칼럼을 찾을 수 없습니다.")
                print("사용 가능한 칼럼:", df.columns.tolist())
                return pd.DataFrame()

        else:
            print("데이터를 가져오지 못했습니다. 응답:", resp)
            return pd.DataFrame()

    except Exception as e:
        print(f"에러 발생: {e}")
        return pd.DataFrame()



def analyze_chart(symbol: str, exchange: str, config: KisConfig, ma_windows=[5, 20, 60], rsi_period=14, macd_periods=(12, 26, 9), 
                  bb_window=20, stoch_periods=(14, 3)) -> pd.DataFrame:
    """
    주어진 OHLCV 데이터에 대해 여러 기술적 지표를 계산하는 함수
    
    Args:
        symbol: 주식 심볼
        exchange: 거래소 코드
        config: KIS API 설정
        ma_windows: 이동평균선 기간 리스트
        rsi_period: RSI 계산 기간
        macd_periods: MACD 계산 기간 (fast, slow, signal)
        bb_window: 볼린저 밴드 기간
        stoch_periods: 스토캐스틱 계산 기간 (k_period, d_period)
        
    Returns:
        원본 데이터와 계산된 기술적 지표들이 통합된 데이터프레임
    """
    df = get_foreign_chart_ohlcv(symbol, exchange, config)
    
    # 데이터프레임 복사본 생성
    result_df = df.copy()
    
    # 여기에 기술적 지표 계산을 추가할 수 있음
    # 현재는 OHLCV 데이터만 반환
    
    return result_df


    
def main() : 
    api_key = os.getenv("KIS_API_KEY")
    api_secret = os.getenv("KIS_API_SECRET")
    account_no_full = os.getenv("KIS_INVEST_ACCOUNT_NO_FULL")

    config = KisConfig(api_key=api_key, api_secret=api_secret, account_no_full=account_no_full)

    # 나스닥에서 애플 주식 데이터 가져오기
    symbol = "AAPL"
    exchange = "나스닥"  # 한글로 거래소 지정 (mojito 라이브러리 요구사항)
    
    print(f"\n===== {symbol} 데이터 가져오는 중 =====")
    df = get_foreign_chart_ohlcv(symbol, exchange, config)
    
    if not df.empty:
        # 결과 출력
        print(f"\n===== {symbol} 기술적 지표 분석 결과 =====")
        print(df)

        print("\n===== CSV 변환 =====")
        df.to_csv(f"{symbol}_stock_data.csv", index=True)
        
        # 선택적: mplfinance로 시각화
        # mpf.plot(df, type='candle', title=f'{symbol} 주가 차트', volume=False)
    else:
        print("분석 또는 시각화할 데이터가 없습니다")
    
if __name__ == "__main__":
    main()