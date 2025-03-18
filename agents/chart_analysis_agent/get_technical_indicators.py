import pandas as pd
import numpy as np
from datetime import datetime

def calculate_atr(df, period=14):
    """
    ATR(Average True Range) 계산
    
    Args:
        df: OHLCV 데이터프레임
        period: ATR 계산 기간
        
    Returns:
        ATR 시리즈
    """
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    atr = true_range.rolling(window=period).mean()
    
    return atr

# VWAP(Volume Weighted Average Price) 계산
def calculate_vwap(df, period=1):
    """
    거래량 가중 평균 가격(VWAP) 계산
    
    Args:
        df: OHLCV 데이터프레임
        period: 계산 기간 (일)
        
    Returns:
        VWAP 시리즈
    """
    if 'volume' not in df.columns:
        print("거래량 데이터가 없어 VWAP를 계산할 수 없습니다.")
        return pd.Series(index=df.index)
    
    # 전형적 가격 계산
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # 일일 VWAP 계산
    cumulative_tp_vol = 0
    cumulative_vol = 0
    vwap = []
    
    for i in range(len(df)):
        if i % period == 0:  # 새 기간 시작
            cumulative_tp_vol = 0
            cumulative_vol = 0
        
        cumulative_tp_vol += typical_price.iloc[i] * df['volume'].iloc[i]
        cumulative_vol += df['volume'].iloc[i]
        
        if cumulative_vol == 0:
            vwap.append(None)
        else:
            vwap.append(cumulative_tp_vol / cumulative_vol)
    
    return pd.Series(vwap, index=df.index)

# 가격 모멘텀 발산 지표 (Price Momentum Oscillator)
def calculate_pmo(df, period_short=35, period_long=20):
    """
    가격 모멘텀 발산 지표(PMO) 계산
    
    Args:
        df: OHLCV 데이터프레임
        period_short: 단기 이동평균 기간
        period_long: 장기 이동평균 기간
        
    Returns:
        PMO 시리즈와 PMO 신호선 시리즈가 포함된 데이터프레임
    """
    # 일일 변화율 계산
    rate_of_change = df['close'].pct_change() * 100
    
    # 단기 EMA
    ema_short = rate_of_change.ewm(span=period_short, adjust=False).mean()
    
    # 장기 EMA
    ema_long = ema_short.ewm(span=period_long, adjust=False).mean()
    
    # 10을 곱해 스케일 조정
    pmo = ema_long * 10
    
    # 신호선 (PMO의 10일 EMA)
    pmo_signal = pmo.ewm(span=10, adjust=False).mean()
    
    return pd.DataFrame({
        'PMO': pmo,
        'PMO_Signal': pmo_signal
    })

# 차이크 발진기 (Chaikin Oscillator)
def calculate_chaikin_oscillator(df, period_short=3, period_long=10):
    """
    차이크 발진기(Chaikin Oscillator) 계산
    
    Args:
        df: OHLCV 데이터프레임
        period_short: 단기 이동평균 기간
        period_long: 장기 이동평균 기간
        
    Returns:
        차이크 발진기 시리즈
    """
    if 'volume' not in df.columns:
        print("거래량 데이터가 없어 차이크 발진기를 계산할 수 없습니다.")
        return pd.Series(index=df.index)
    
    # 누적 분포선(Accumulation Distribution Line) 계산
    high_low = df['high'] - df['low']
    close_low = df['close'] - df['low']
    high_close = df['high'] - df['close']
    
    # 0으로 나누기 방지
    high_low = high_low.replace(0, 1e-10)
    
    money_flow_multiplier = ((close_low - high_close) / high_low)
    money_flow_volume = money_flow_multiplier * df['volume']
    adl = money_flow_volume.cumsum()
    
    # 단기 및 장기 EMA 계산
    ema_short = adl.ewm(span=period_short, adjust=False).mean()
    ema_long = adl.ewm(span=period_long, adjust=False).mean()
    
    # 차이크 발진기 = 단기 EMA - 장기 EMA
    chaikin_osc = ema_short - ema_long
    
    return chaikin_osc

# 슈퍼트렌드 지표 계산
def calculate_supertrend(df, period=10, multiplier=3.0):
    """
    슈퍼트렌드 지표 계산
    
    Args:
        df: OHLCV 데이터프레임
        period: ATR 계산 기간
        multiplier: ATR 승수
        
    Returns:
        슈퍼트렌드, 상단선, 하단선, 추세 방향이 포함된 데이터프레임
    """
    # ATR 계산
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    atr = true_range.rolling(window=period).mean()
    
    # 기본 상단선과 하단선
    hl2 = (df['high'] + df['low']) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # 슈퍼트렌드 계산
    supertrend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)  # 1: 상승 추세, -1: 하락 추세
    
    # 첫 번째 값 설정
    supertrend.iloc[period-1] = lower_band.iloc[period-1]
    
    # 나머지 계산
    for i in range(period, len(df)):
        curr_close = df['close'].iloc[i]
        prev_supertrend = supertrend.iloc[i-1]
        curr_upper = upper_band.iloc[i]
        curr_lower = lower_band.iloc[i]
        prev_direction = direction.iloc[i-1]
        
        # 상승 추세
        if prev_supertrend <= prev_direction * curr_close:
            # 추세 유지
            supertrend.iloc[i] = curr_lower if prev_direction == 1 else curr_upper
            direction.iloc[i] = prev_direction
        else:
            # 추세 반전
            supertrend.iloc[i] = curr_upper if prev_direction == 1 else curr_lower
            direction.iloc[i] = -prev_direction
    
    return pd.DataFrame({
        'Supertrend': supertrend,
        'UpperBand': upper_band,
        'LowerBand': lower_band,
        'Direction': direction
    })

# 켈트너 채널 계산
def calculate_keltner_channel(df, ema_period=20, atr_period=10, multiplier=2.0):
    """
    켈트너 채널 계산
    
    Args:
        df: OHLCV 데이터프레임
        ema_period: EMA 기간
        atr_period: ATR 기간
        multiplier: ATR 승수
        
    Returns:
        중앙선, 상단선, 하단선이 포함된 딕셔너리
    """
    # 중심선 (EMA)
    middle_line = df['close'].ewm(span=ema_period, adjust=False).mean()
    
    # ATR 계산
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=atr_period).mean()
    
    # 상단과 하단 밴드
    upper_line = middle_line + (multiplier * atr)
    lower_line = middle_line - (multiplier * atr)
    
    return {
        'middle_line': middle_line,
        'upper_line': upper_line,
        'lower_line': lower_line
    }

# 캔들스틱 패턴 감지 함수
def detect_candlestick_patterns(df):
    """
    주요 캔들스틱 패턴 감지
    
    Args:
        df: OHLCV 데이터프레임
        
    Returns:
        패턴이 감지된 데이터프레임
    """
    result_df = df.copy()
    
    # 각 캔들의 특성 계산
    result_df['BodySize'] = abs(result_df['close'] - result_df['open'])
    result_df['UpperShadow'] = result_df['high'] - result_df[['open', 'close']].max(axis=1)
    result_df['LowerShadow'] = result_df[['open', 'close']].min(axis=1) - result_df['low']
    result_df['IsBullish'] = result_df['close'] > result_df['open']
    result_df['IsBearish'] = result_df['close'] < result_df['open']
    
    # 도지 패턴 (Doji) - 시가와 종가가 거의 같음
    doji_threshold = 0.1  # 몸통 크기가 전체 범위의 10% 미만
    result_df['Doji'] = (result_df['BodySize'] / (result_df['high'] - result_df['low'])) < doji_threshold
    
    # 망치형 (Hammer) - 아래 그림자가 길고, 위 그림자가 짧음, 몸통이 작음
    hammer_body_threshold = 0.3  # 몸통 크기가 전체 범위의 30% 미만
    hammer_lower_shadow = 2  # 아래 그림자가 몸통의 2배 이상
    result_df['Hammer'] = (
        (result_df['BodySize'] / (result_df['high'] - result_df['low']) < hammer_body_threshold) &
        (result_df['LowerShadow'] > hammer_lower_shadow * result_df['BodySize']) &
        (result_df['UpperShadow'] < result_df['BodySize'])
    )
    
    # 역망치형 (Inverted Hammer)
    result_df['InvertedHammer'] = (
        (result_df['BodySize'] / (result_df['high'] - result_df['low']) < hammer_body_threshold) &
        (result_df['UpperShadow'] > hammer_lower_shadow * result_df['BodySize']) &
        (result_df['LowerShadow'] < result_df['BodySize'])
    )
    
    # 별형 (Shooting Star) - 역망치와 비슷하지만 하락 추세 끝에 나타남
    result_df['ShootingStar'] = result_df['InvertedHammer'] & (result_df['close'].shift(1) > result_df['open'].shift(1))
    
    # 십자형 (Marubozu) - 몸통이 크고 그림자가 거의 없음
    marubozu_body_threshold = 0.8  # 몸통 크기가 전체 범위의 80% 이상
    result_df['Marubozu'] = (result_df['BodySize'] / (result_df['high'] - result_df['low']) > marubozu_body_threshold)
    
    # 장대양봉 (Long Bullish)
    long_body_threshold = 2  # 평균 몸통 크기의 2배 이상
    avg_body_size = result_df['BodySize'].rolling(window=10).mean()
    result_df['LongBullish'] = (result_df['IsBullish']) & (result_df['BodySize'] > long_body_threshold * avg_body_size)
    
    # 장대음봉 (Long Bearish)
    result_df['LongBearish'] = (result_df['IsBearish']) & (result_df['BodySize'] > long_body_threshold * avg_body_size)
    
    # 2개의 캔들 패턴 (패턴이 형성된 곳에 표시)
    
    # 잉여 상승형 (Bullish Engulfing)
    result_df['BullishEngulfing'] = (
        (result_df['IsBullish']) &
        (result_df['IsBearish'].shift(1)) &
        (result_df['open'] < result_df['close'].shift(1)) &
        (result_df['close'] > result_df['open'].shift(1))
    )
    
    # 잉여 하락형 (Bearish Engulfing)
    result_df['BearishEngulfing'] = (
        (result_df['IsBearish']) &
        (result_df['IsBullish'].shift(1)) &
        (result_df['open'] > result_df['close'].shift(1)) &
        (result_df['close'] < result_df['open'].shift(1))
    )
    
    # 외부 상승형 (Piercing Line)
    result_df['PiercingLine'] = (
        (result_df['IsBullish']) &
        (result_df['IsBearish'].shift(1)) &
        (result_df['open'] < result_df['close'].shift(1)) &
        (result_df['close'] > (result_df['open'].shift(1) + result_df['close'].shift(1)) / 2) &
        (result_df['close'] < result_df['open'].shift(1))
    )
    
    # 외부 하락형 (Dark Cloud Cover)
    result_df['DarkCloudCover'] = (
        (result_df['IsBearish']) &
        (result_df['IsBullish'].shift(1)) &
        (result_df['open'] > result_df['close'].shift(1)) &
        (result_df['close'] < (result_df['open'].shift(1) + result_df['close'].shift(1)) / 2) &
        (result_df['close'] > result_df['open'].shift(1))
    )
    
    # 3개의 캔들 패턴
    
    # 모닝스타 (Morning Star)
    result_df['MorningStar'] = (
        (result_df['IsBearish'].shift(2)) &
        (result_df['BodySize'].shift(1) < result_df['BodySize'].shift(2) * 0.5) &
        (result_df['IsBullish']) &
        (result_df['close'] > (result_df['open'].shift(2) + result_df['close'].shift(2)) / 2)
    )
    
    # 이브닝스타 (Evening Star)
    result_df['EveningStar'] = (
        (result_df['IsBullish'].shift(2)) &
        (result_df['BodySize'].shift(1) < result_df['BodySize'].shift(2) * 0.5) &
        (result_df['IsBearish']) &
        (result_df['close'] < (result_df['open'].shift(2) + result_df['close'].shift(2)) / 2)
    )
    
    # 상승 3군 (Three White Soldiers)
    result_df['ThreeWhiteSoldiers'] = (
        (result_df['IsBullish']) &
        (result_df['IsBullish'].shift(1)) &
        (result_df['IsBullish'].shift(2)) &
        (result_df['open'] > result_df['open'].shift(1)) &
        (result_df['open'].shift(1) > result_df['open'].shift(2)) &
        (result_df['close'] > result_df['close'].shift(1)) &
        (result_df['close'].shift(1) > result_df['close'].shift(2))
    )
    
    # 하락 3군 (Three Black Crows)
    result_df['ThreeBlackCrows'] = (
        (result_df['IsBearish']) &
        (result_df['IsBearish'].shift(1)) &
        (result_df['IsBearish'].shift(2)) &
        (result_df['open'] < result_df['open'].shift(1)) &
        (result_df['open'].shift(1) < result_df['open'].shift(2)) &
        (result_df['close'] < result_df['close'].shift(1)) &
        (result_df['close'].shift(1) < result_df['close'].shift(2))
    )
    
    return result_df

# 엘리엇 파동 지표 계산 (간단한 구현)
def calculate_elliott_wave_indicators(df):
    """
    엘리엇 파동 분석을 위한 간단한 지표 계산
    
    Args:
        df: OHLCV 데이터프레임
        
    Returns:
        엘리엇 파동 지표가 추가된 데이터프레임
    """
    result_df = df.copy()
    
    # 추세 방향 확인을 위한 지표
    result_df['EMA5'] = result_df['close'].ewm(span=5, adjust=False).mean()
    result_df['EMA13'] = result_df['close'].ewm(span=13, adjust=False).mean()
    
    # 상승/하락 추세 확인
    result_df['UpTrend'] = result_df['EMA5'] > result_df['EMA13']
    result_df['DownTrend'] = result_df['EMA5'] < result_df['EMA13']
    
    # 추세 강도 (ATR 기반)
    result_df['ATR14'] = calculate_atr(result_df, 14)
    result_df['ATR14_Pct'] = result_df['ATR14'] / result_df['close'] * 100
    
    # Zigzag 패턴 감지 (단순화된 버전)
    # 이전 20일 내에서 고점과 저점 찾기
    lookback = 20
    
    for i in range(lookback, len(result_df)):
        window = result_df.iloc[i-lookback:i]
        
        # 이전 고점/저점
        prev_high_idx = window['high'].idxmax()
        prev_low_idx = window['low'].idxmin()
        
        current_idx = result_df.index[i]
        
        # 새로운 고점/저점 확인
        result_df.loc[current_idx, 'IsPotentialTop'] = (
            result_df.loc[current_idx, 'high'] == window['high'].max() and
            result_df.loc[current_idx, 'high'] > result_df.loc[current_idx-1, 'high'] and
            result_df.loc[current_idx, 'high'] > result_df.loc[current_idx+1 if i+1 < len(result_df) else current_idx, 'high']
        )
        
        result_df.loc[current_idx, 'IsPotentialBottom'] = (
            result_df.loc[current_idx, 'low'] == window['low'].min() and
            result_df.loc[current_idx, 'low'] < result_df.loc[current_idx-1, 'low'] and
            result_df.loc[current_idx, 'low'] < result_df.loc[current_idx+1 if i+1 < len(result_df) else current_idx, 'low']
        )
    
    return result_df

# 볼륨 프로파일(Volume Profile) 계산
def calculate_volume_profile(df, price_bins=10):
    """
    볼륨 프로파일(Volume Profile) 계산
    
    Args:
        df: 거래량 포함된 OHLCV 데이터프레임
        price_bins: 가격 구간 수
        
    Returns:
        가격 구간별 거래량 데이터프레임
    """
    if 'volume' not in df.columns:
        print("거래량 데이터가 없어 볼륨 프로파일을 계산할 수 없습니다.")
        return pd.DataFrame()
    
    price_range = (df['high'].max() - df['low'].min())
    bin_size = price_range / price_bins
    
    # 각 봉의 거래량을 가격 구간에 분배
    volume_by_price = {}
    
    for idx, row in df.iterrows():
        # 해당 봉의 가격 범위
        candle_range = row['high'] - row['low']
        # 해당 봉이 걸쳐있는 가격 구간들 계산
        lower_bin = int((row['low'] - df['low'].min()) / bin_size)
        upper_bin = int((row['high'] - df['low'].min()) / bin_size)
        
        # 각 구간에 거래량 분배
        for bin_idx in range(lower_bin, upper_bin + 1):
            if bin_idx >= price_bins:
                bin_idx = price_bins - 1
            elif bin_idx < 0:
                bin_idx = 0
                
            bin_price = df['low'].min() + (bin_idx * bin_size + bin_size/2)
            
            # 빈에 거래량 할당 (단순화를 위해 균등 분배)
            if candle_range > 0:
                # 캔들 크기에 비례하여 각 구간에 거래량 분배
                segment_size = bin_size
                if bin_idx == lower_bin:
                    # 최하단 구간은 low부터 다음 구간 경계까지
                    segment_size = min(bin_size, (lower_bin + 1) * bin_size + df['low'].min() - row['low'])
                elif bin_idx == upper_bin:
                    # 최상단 구간은 이전 구간 경계부터 high까지
                    segment_size = min(bin_size, row['high'] - upper_bin * bin_size - df['low'].min())
                
                volume_contribution = (segment_size / candle_range) * row['volume']
            else:
                # 시가=종가인 경우, 해당 가격에 모든 거래량 할당
                volume_contribution = row['volume'] if row['low'] <= bin_price <= row['high'] else 0
            
            if bin_price in volume_by_price:
                volume_by_price[bin_price] += volume_contribution
            else:
                volume_by_price[bin_price] = volume_contribution
    
    # 결과를 데이터프레임으로 변환
    volume_profile_df = pd.DataFrame({
        'price': list(volume_by_price.keys()),
        'volume': list(volume_by_price.values())
    }).sort_values('price')
    return volume_profile_df

# 피보나치 수준 계산 함수
def calculate_fibonacci_levels(df, period=20):
    """
    특정 기간 내의 최고가와 최저가를 기준으로 피보나치 수준 계산
    
    Args:
        df: OHLCV 데이터프레임
        period: 분석 기간
        
    Returns:
        피보나치 수준 딕셔너리
    """
    # 해당 기간의 최고가와 최저가 찾기
    recent_df = df.tail(period)
    high = recent_df['high'].max()
    low = recent_df['low'].min()
    
    # 피보나치 비율
    ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    
    # 피보나치 수준 계산 (하락 트렌드)
    fib_down = {}
    for ratio in ratios:
        level = high - (high - low) * ratio
        fib_down[f'Fib_{ratio:.3f}'] = level
    
    # 피보나치 수준 계산 (상승 트렌드)
    fib_up = {}
    for ratio in ratios:
        level = low + (high - low) * ratio
        fib_up[f'Fib_{ratio:.3f}'] = level
    
    # 피보나치 확장 수준 (1.272, 1.618, 2.0, 2.618)
    extension_ratios = [1.272, 1.618, 2.0, 2.618]
    extension_levels = {}
    
    for ratio in extension_ratios:
        # 상승 확장
        ext_up = low + (high - low) * ratio
        # 하락 확장
        ext_down = high - (high - low) * ratio
        
        extension_levels[f'FibExt_Up_{ratio:.3f}'] = ext_up
        extension_levels[f'FibExt_Down_{ratio:.3f}'] = ext_down
    
    result = {
        'trend_direction': 'up' if df['close'].iloc[-1] > df['close'].iloc[-period] else 'down',
        'high': high,
        'low': low,
        'fib_up': fib_up,
        'fib_down': fib_down,
        'extension': extension_levels
    }
    
    return result

# 피벗 포인트 계산 함수
def calculate_pivot_points(df, method='standard'):
    """
    여러 방식의 피벗 포인트 계산
    
    Args:
        df: OHLCV 데이터프레임
        method: 계산 방식 ('standard', 'fibonacci', 'woodie', 'camarilla', 'demark')
        
    Returns:
        피벗 포인트 딕셔너리
    """
    # 최근 데이터 가져오기 (일반적으로 전날 데이터 사용)
    prev_high = df['high'].iloc[-2]
    prev_low = df['low'].iloc[-2]
    prev_close = df['close'].iloc[-2]
    prev_open = df['open'].iloc[-2]
    
    result = {}
    
    if method == 'standard':
        # 스탠다드 피벗 포인트
        pivot = (prev_high + prev_low + prev_close) / 3
        
        s1 = (2 * pivot) - prev_high
        s2 = pivot - (prev_high - prev_low)
        s3 = s2 - (prev_high - prev_low)
        
        r1 = (2 * pivot) - prev_low
        r2 = pivot + (prev_high - prev_low)
        r3 = r2 + (prev_high - prev_low)
        
        result = {
            'P': pivot,
            'S1': s1, 'S2': s2, 'S3': s3,
            'R1': r1, 'R2': r2, 'R3': r3
        }
        
    elif method == 'fibonacci':
        # 피보나치 피벗 포인트
        pivot = (prev_high + prev_low + prev_close) / 3
        
        s1 = pivot - 0.382 * (prev_high - prev_low)
        s2 = pivot - 0.618 * (prev_high - prev_low)
        s3 = pivot - 1.0 * (prev_high - prev_low)
        
        r1 = pivot + 0.382 * (prev_high - prev_low)
        r2 = pivot + 0.618 * (prev_high - prev_low)
        r3 = pivot + 1.0 * (prev_high - prev_low)
        
        result = {
            'P': pivot,
            'S1': s1, 'S2': s2, 'S3': s3,
            'R1': r1, 'R2': r2, 'R3': r3
        }
        
    elif method == 'woodie':
        # 우디 피벗 포인트
        pivot = (prev_high + prev_low + 2 * prev_close) / 4
        
        s1 = (2 * pivot) - prev_high
        s2 = pivot - (prev_high - prev_low)
        s3 = s2 - (prev_high - prev_low)
        
        r1 = (2 * pivot) - prev_low
        r2 = pivot + (prev_high - prev_low)
        r3 = r2 + (prev_high - prev_low)
        
        result = {
            'P': pivot,
            'S1': s1, 'S2': s2, 'S3': s3,
            'R1': r1, 'R2': r2, 'R3': r3
        }
        
    elif method == 'camarilla':
        # 카마릴라 피벗 포인트
        pivot = (prev_high + prev_low + prev_close) / 3
        
        range_val = prev_high - prev_low
        
        s1 = prev_close - (range_val * 1.1 / 12)
        s2 = prev_close - (range_val * 1.1 / 6)
        s3 = prev_close - (range_val * 1.1 / 4)
        s4 = prev_close - (range_val * 1.1 / 2)
        
        r1 = prev_close + (range_val * 1.1 / 12)
        r2 = prev_close + (range_val * 1.1 / 6)
        r3 = prev_close + (range_val * 1.1 / 4)
        r4 = prev_close + (range_val * 1.1 / 2)
        
        result = {
            'P': pivot,
            'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4,
            'R1': r1, 'R2': r2, 'R3': r3, 'R4': r4
        }
        
    elif method == 'demark':
        # 디마크 피벗 포인트
        if prev_close < prev_open:
            x = prev_high + (2 * prev_low) + prev_close
        elif prev_close > prev_open:
            x = (2 * prev_high) + prev_low + prev_close
        else:
            x = prev_high + prev_low + (2 * prev_close)
            
        pivot = x / 4
        
        s1 = x / 2 - prev_high
        r1 = x / 2 - prev_low
        
        result = {
            'P': pivot,
            'S1': s1,
            'R1': r1
        }
    
    return result

def add_technical_indicators(df):
    """
    모든 기술적 지표를 계산하고 데이터프레임에 추가
    
    Args:
        df: OHLCV 데이터프레임
        
    Returns:
        기술적 지표가 추가된 데이터프레임
    """
    df_with_indicators = df.copy()
    
    # 1. 기본 이동평균선
    df_with_indicators['MA5'] = df_with_indicators['close'].rolling(window=5).mean()
    df_with_indicators['MA20'] = df_with_indicators['close'].rolling(window=20).mean()
    df_with_indicators['MA60'] = df_with_indicators['close'].rolling(window=60).mean()
    
    # 2. 볼린저 밴드
    std_dev = df_with_indicators['close'].rolling(window=20).std()
    df_with_indicators['BB_Middle'] = df_with_indicators['MA20']
    df_with_indicators['BB_Upper'] = df_with_indicators['BB_Middle'] + (std_dev * 2)
    df_with_indicators['BB_Lower'] = df_with_indicators['BB_Middle'] - (std_dev * 2)
    
    # 3. RSI
    delta = df_with_indicators['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_with_indicators['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. MACD
    exp1 = df_with_indicators['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_with_indicators['close'].ewm(span=26, adjust=False).mean()
    df_with_indicators['MACD'] = exp1 - exp2
    df_with_indicators['MACD_Signal'] = df_with_indicators['MACD'].ewm(span=9, adjust=False).mean()
    df_with_indicators['MACD_Histogram'] = df_with_indicators['MACD'] - df_with_indicators['MACD_Signal']
    
    # 5. 스토캐스틱
    high_14 = df_with_indicators['high'].rolling(window=14).max()
    low_14 = df_with_indicators['low'].rolling(window=14).min()
    df_with_indicators['Stoch_K'] = 100 * ((df_with_indicators['close'] - low_14) / (high_14 - low_14))
    df_with_indicators['Stoch_D'] = df_with_indicators['Stoch_K'].rolling(window=3).mean()
    
    # 6. VWAP (거래량 가중 평균 가격)
    if 'volume' in df.columns:
        df_with_indicators['VWAP'] = calculate_vwap(df)
    
    # 7. 슈퍼트렌드
    supertrend_data = calculate_supertrend(df)
    df_with_indicators['Supertrend'] = supertrend_data['Supertrend']
    df_with_indicators['Supertrend_Direction'] = supertrend_data['Direction']
    
    # 8. 켈트너 채널
    keltner_data = calculate_keltner_channel(df)
    df_with_indicators['KC_Middle'] = keltner_data['middle_line']
    df_with_indicators['KC_Upper'] = keltner_data['upper_line']
    df_with_indicators['KC_Lower'] = keltner_data['lower_line']
    
    # 9. 캔들스틱 패턴
    pattern_data = detect_candlestick_patterns(df)
    pattern_columns = [col for col in pattern_data.columns if col not in df.columns]
    for col in pattern_columns:
        df_with_indicators[col] = pattern_data[col]
    
    # 10. 엘리엇 파동 지표
    elliott_data = calculate_elliott_wave_indicators(df)
    elliott_columns = [col for col in elliott_data.columns if col not in df_with_indicators.columns]
    for col in elliott_columns:
        df_with_indicators[col] = elliott_data[col]
    
    # 11. 피보나치 수준
    fib_data = calculate_fibonacci_levels(df)
    df_with_indicators['Fib_Trend'] = fib_data['trend_direction']
    
    # 12. 피벗 포인트
    pivot_data = calculate_pivot_points(df)
    for key, value in pivot_data.items():
        df_with_indicators[f'Pivot_{key}'] = value
    
    # NaN 값 처리 (선택적)
    # df_with_indicators = df_with_indicators.fillna(method='bfill')
    
    return df_with_indicators

# 데이터 로드
def load_stock_data(file_path='dataframe.csv'):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

# 데이터를 전처리하고 기술적 지표를 추가하는 메인 함수

if __name__ == "__main__":
    # 데이터 로드
    df = load_stock_data('dataframe.csv')
    # 기술적 지표 추가
    df_processed = add_technical_indicators(df)

    print(f"데이터 처리 완료: {len(df_processed)} 행, {df_processed.columns.size} 열")
    print(f"추가된 기술적 지표: {[col for col in df_processed.columns if col not in ['date', 'open', 'high', 'low', 'close', 'volume']]}")
    df_processed.to_csv("dataframe_indicators.csv", index=True)

    print(df_processed)