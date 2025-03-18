"""
주가 기술 분석 모듈

이 모듈은 주가 데이터에 대한 기술적 분석을 수행합니다.
"""

import pandas as pd
import numpy as np
import re
import traceback
from typing import List, Dict, Optional, Union

# 상수 정의
DEFAULT_LOOKBACK_PERIOD = 30
DEFAULT_PATTERN_LOOKBACK = 5


class TechnicalAnalyzer:
    """주가 데이터에 대한 기술적 분석을 수행하는 클래스"""
    
    def __init__(self, df):
        """
        TechnicalAnalyzer 클래스를 초기화합니다.
        
        Args:
            df (DataFrame): 분석할 주가 데이터
        """
        self.df = df
        self.current = df.iloc[-1] if not df.empty else None
        self.prev = df.iloc[-2] if len(df) > 1 else None
    
    def calculate_trend_strength(self):
        """
        현재 추세의 강도와 방향을 계산합니다.
        
        Returns:
            str: 추세 강도 설명 (Strong Uptrend, Moderate Downtrend 등)
        """
        if self.df.empty or self.current is None:
            return "Insufficient data"
        
        # 상승 추세 신호 카운트
        uptrend_signals = self._count_uptrend_signals()
        
        # 하락 추세 신호 카운트
        downtrend_signals = self._count_downtrend_signals()
        
        # 추세 강도 판단
        return self._determine_trend_strength(uptrend_signals, downtrend_signals)
    
    def _count_uptrend_signals(self):
        """상승 추세 신호 개수를 계산합니다."""
        return sum([
            self.current.get('UpTrend', False),
            self.current.get('close', 0) > self.current.get('MA20', 0),
            self.current.get('MA5', 0) > self.current.get('MA20', 0),
            self.current.get('RSI', 0) > 50,
            self.current.get('MACD', 0) > self.current.get('MACD_Signal', 0),
            self.current.get('Supertrend_Direction', 0) == 1
        ])
    
    def _count_downtrend_signals(self):
        """하락 추세 신호 개수를 계산합니다."""
        return sum([
            self.current.get('DownTrend', False),
            self.current.get('close', 0) < self.current.get('MA20', 0),
            self.current.get('MA5', 0) < self.current.get('MA20', 0),
            self.current.get('RSI', 0) < 50,
            self.current.get('MACD', 0) < self.current.get('MACD_Signal', 0),
            self.current.get('Supertrend_Direction', 0) == -1
        ])
    
    def _determine_trend_strength(self, uptrend_signals, downtrend_signals):
        """신호 개수를 기반으로 추세 강도를 판단합니다."""
        if uptrend_signals >= 5:
            return "Strong Uptrend"
        elif uptrend_signals >= 4:
            return "Moderate Uptrend"
        elif uptrend_signals >= 3:
            return "Weak Uptrend"
        elif downtrend_signals >= 5:
            return "Strong Downtrend"
        elif downtrend_signals >= 4:
            return "Moderate Downtrend"
        elif downtrend_signals >= 3:
            return "Weak Downtrend"
        else:
            return "Sideways/Neutral"
    
    def identify_support_resistance(self, lookback=20):
        """
        주요 지지선 및 저항선 수준을 식별합니다.
        
        Args:
            lookback (int): 분석할 과거 데이터 기간
            
        Returns:
            dict: 지지선 및 저항선 목록
        """
        if self.df.empty or len(self.df) < lookback:
            return {"support": [], "resistance": []}
        
        # 최근 데이터 가져오기
        recent_df = self.df.iloc[-lookback:]
        
        # 지지선 및 저항선 추출
        support_levels = []
        resistance_levels = []
        
        # 피봇 포인트 추가
        self._add_pivot_points(support_levels, resistance_levels)
        
        # 최근 저점을 지지선으로 추가
        self._add_local_minima_maxima(recent_df, support_levels, resistance_levels)
        
        # 볼린저 밴드 추가
        self._add_bollinger_bands(support_levels, resistance_levels)
        
        # 레벨 정리 및 정렬
        support_levels, resistance_levels = self._process_levels(support_levels, resistance_levels)
        
        return {
            "support": support_levels,
            "resistance": resistance_levels
        }
    
    def _add_pivot_points(self, support_levels, resistance_levels):
        """피봇 포인트를 지지선/저항선에 추가"""
        for level in ['Pivot_S1', 'Pivot_S2', 'Pivot_S3']:
            if level in self.current and not pd.isna(self.current[level]):
                support_levels.append(self.current[level])
        
        for level in ['Pivot_R1', 'Pivot_R2', 'Pivot_R3']:
            if level in self.current and not pd.isna(self.current[level]):
                resistance_levels.append(self.current[level])
    
    def _add_local_minima_maxima(self, recent_df, support_levels, resistance_levels):
        """최근 로컬 고점/저점을 지지선/저항선에 추가"""
        self._find_local_minima(recent_df, support_levels)
        self._find_local_maxima(recent_df, resistance_levels)
    
    def _find_local_minima(self, df, support_levels):
        """로컬 최소값을 찾아 지지선에 추가"""
        if 'low' in df:
            for i in range(1, len(df)-1):
                if df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i+1]:
                    support_levels.append(df['low'].iloc[i])
    
    def _find_local_maxima(self, df, resistance_levels):
        """로컬 최대값을 찾아 저항선에 추가"""
        if 'high' in df:
            for i in range(1, len(df)-1):
                if df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i+1]:
                    resistance_levels.append(df['high'].iloc[i])
    
    def _add_bollinger_bands(self, support_levels, resistance_levels):
        """볼린저 밴드를 지지선/저항선에 추가"""
        if 'BB_Lower' in self.current and not pd.isna(self.current['BB_Lower']):
            support_levels.append(self.current['BB_Lower'])
        
        if 'BB_Upper' in self.current and not pd.isna(self.current['BB_Upper']):
            resistance_levels.append(self.current['BB_Upper'])
    
    def _process_levels(self, support_levels, resistance_levels):
        """지지선/저항선 처리 및 필터링"""
        # 반올림 및 정렬
        support_levels = sorted(set([round(level, 2) for level in support_levels if not pd.isna(level)]))
        resistance_levels = sorted(set([round(level, 2) for level in resistance_levels if not pd.isna(level)]))
        
        # 현재 가격 기준으로 필터링
        current_price = self.current.get('close', 0)
        support_levels = [level for level in support_levels if level < current_price]
        resistance_levels = [level for level in resistance_levels if level > current_price]
        
        return support_levels, resistance_levels
    
    def detect_patterns(self, lookback=DEFAULT_PATTERN_LOOKBACK):
        """
        최근 캔들스틱 및 차트 패턴을 감지합니다.
        
        Args:
            lookback (int): 분석할 과거 데이터 기간
            
        Returns:
            list: 감지된 패턴 목록
        """
        if self.df.empty or len(self.df) < lookback:
            return []
        
        recent_df = self.df.iloc[-lookback:]
        patterns = []
        
        # 캔들스틱 패턴 확인
        self._check_candlestick_patterns(recent_df, patterns)
        
        # 잠재적 고점과 저점 확인
        self._check_potential_tops_bottoms(recent_df, patterns)
        
        return patterns
    
    def _check_candlestick_patterns(self, df, patterns):
        """캔들스틱 패턴을 확인하고 패턴 목록에 추가"""
        pattern_columns = [
            'Doji', 'Hammer', 'InvertedHammer', 'ShootingStar', 'Marubozu', 
            'BullishEngulfing', 'BearishEngulfing', 'PiercingLine', 
            'DarkCloudCover', 'MorningStar', 'EveningStar', 
            'ThreeWhiteSoldiers', 'ThreeBlackCrows'
        ]
        
        for col in pattern_columns:
            if col in df.columns:
                pattern_rows = df[df[col] == True]
                for idx, _ in pattern_rows.iterrows():
                    date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
                    # 가독성을 위한 패턴 이름 형식 지정
                    pattern_name = ' '.join(re.findall('[A-Z][^A-Z]*', col))
                    patterns.append(f"{pattern_name} on {date_str}")
    
    def _check_potential_tops_bottoms(self, recent_df, patterns):
        """잠재적 고점/저점 확인 및 패턴에 추가"""
        if 'IsPotentialTop' in recent_df.columns:
            top_rows = recent_df[recent_df['IsPotentialTop'] == True]
            for idx, _ in top_rows.iterrows():
                date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
                patterns.append(f"Potential Top on {date_str}")
        
        if 'IsPotentialBottom' in recent_df.columns:
            bottom_rows = recent_df[recent_df['IsPotentialBottom'] == True]
            for idx, _ in bottom_rows.iterrows():
                date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
                patterns.append(f"Potential Bottom on {date_str}")
    
    def analyze_indicators(self):
        """
        기술적 지표의 현재 상태를 분석합니다.
        
        Returns:
            dict: 기술적 지표 분석 결과
        """
        if self.df.empty or self.current is None:
            return {}
        
        indicator_analysis = {}
        
        # 지표별 분석 수행
        self._analyze_rsi(indicator_analysis)
        self._analyze_macd(indicator_analysis)
        self._analyze_stochastic(indicator_analysis)
        self._analyze_bollinger_bands(indicator_analysis)
        self._analyze_supertrend(indicator_analysis)
        self._analyze_atr(indicator_analysis)
        
        return indicator_analysis
    
    def _analyze_rsi(self, indicator_analysis):
        """RSI 지표 분석"""
        if 'RSI' in self.current:
            rsi_value = self.current['RSI']
            if not pd.isna(rsi_value):
                if rsi_value > 70:
                    rsi_status = "Overbought"
                elif rsi_value < 30:
                    rsi_status = "Oversold"
                elif rsi_value > 50:
                    rsi_status = "Bullish"
                else:
                    rsi_status = "Bearish"
                
                indicator_analysis["RSI"] = {
                    "value": round(rsi_value, 2), 
                    "status": rsi_status
                }
    
    def _analyze_macd(self, indicator_analysis):
        """MACD 지표 분석"""
        macd_keys = ['MACD', 'MACD_Signal', 'MACD_Histogram']
        if all(k in self.current for k in macd_keys):
            macd = self.current['MACD']
            signal = self.current['MACD_Signal']
            histogram = self.current['MACD_Histogram']
            
            if not any(pd.isna(x) for x in [macd, signal, histogram]):
                macd_status = self._determine_macd_status(macd, signal, histogram)
                
                indicator_analysis["MACD"] = {
                    "value": round(macd, 4),
                    "signal": round(signal, 4),
                    "histogram": round(histogram, 4),
                    "status": macd_status
                }
    
    def _determine_macd_status(self, macd, signal, histogram):
        """MACD 상태 결정"""
        if macd > signal and histogram > 0:
            return "Bullish"
        elif macd < signal and histogram < 0:
            return "Bearish"
        elif macd > signal and histogram < 0:
            return "Bullish Weakening"
        elif macd < signal and histogram > 0:
            return "Bearish Weakening"
        else:
            return "Neutral"
    
    def _analyze_stochastic(self, indicator_analysis):
        """스토캐스틱 지표 분석"""
        if all(k in self.current for k in ['Stoch_K', 'Stoch_D']):
            k = self.current['Stoch_K']
            d = self.current['Stoch_D']
            
            if not any(pd.isna(x) for x in [k, d]):
                stoch_status = self._determine_stochastic_status(k, d)
                
                indicator_analysis["Stochastic"] = {
                    "K": round(k, 2),
                    "D": round(d, 2),
                    "status": stoch_status
                }
    
    def _determine_stochastic_status(self, k, d):
        """스토캐스틱 상태 결정"""
        if k > 80 and d > 80:
            return "Overbought"
        elif k < 20 and d < 20:
            return "Oversold"
        elif k > d:
            return "Bullish Crossover"
        elif k < d:
            return "Bearish Crossover"
        else:
            return "Neutral"
    
    def _analyze_bollinger_bands(self, indicator_analysis):
        """볼린저 밴드 지표 분석"""
        bb_keys = ['close', 'BB_Upper', 'BB_Middle', 'BB_Lower']
        if all(k in self.current for k in bb_keys):
            close = self.current['close']
            upper = self.current['BB_Upper']
            middle = self.current['BB_Middle']
            lower = self.current['BB_Lower']
            
            if not any(pd.isna(x) for x in [close, upper, middle, lower]):
                width = (upper - lower) / middle  # 밴드폭
                bb_status = self._determine_bb_status(close, upper, middle, lower)
                
                indicator_analysis["Bollinger_Bands"] = {
                    "status": bb_status,
                    "bandwidth": round(width, 4)
                }
    
    def _determine_bb_status(self, close, upper, middle, lower):
        """볼린저 밴드 상태 결정"""
        if close > upper:
            return "Above upper band (overbought)"
        elif close < lower:
            return "Below lower band (oversold)"
        elif close > middle:
            return "In upper half (bullish)"
        else:
            return "In lower half (bearish)"
    
    def _analyze_supertrend(self, indicator_analysis):
        """수퍼트렌드 지표 분석"""
        if 'Supertrend_Direction' in self.current:
            direction = self.current['Supertrend_Direction']
            
            if not pd.isna(direction):
                supertrend_status = "Uptrend" if direction == 1 else "Downtrend"
                
                indicator_analysis["Supertrend"] = {
                    "direction": int(direction),
                    "status": supertrend_status
                }
    
    def _analyze_atr(self, indicator_analysis):
        """ATR 지표 분석"""
        if 'ATR14' in self.current:
            atr = self.current['ATR14']
            atr_pct = self.current.get('ATR14_Pct', None)
            
            if not pd.isna(atr):
                indicator_analysis["ATR"] = {
                    "value": round(atr, 2),
                    "percentage": round(atr_pct, 2) if atr_pct and not pd.isna(atr_pct) else None
                }
    
    def generate_trading_signals(self):
        """
        기술적 분석을 기반으로 잠재적 매매 신호를 생성합니다.
        
        Returns:
            list: 생성된 매매 신호 목록
        """
        if self.df.empty or len(self.df) < 3 or self.current is None or self.prev is None:
            return []
        
        signals = []
        
        # 신호 유형별 분석
        self._check_ma_crossover_signals(signals)
        self._check_macd_signals(signals)
        self._check_rsi_signals(signals)
        self._check_bollinger_band_signals(signals)
        self._check_supertrend_signals(signals)
        self._check_stochastic_signals(signals)
        self._check_pattern_signals(signals)
        
        return signals
    
    def _check_ma_crossover_signals(self, signals):
        """이동평균 교차 신호 확인"""
        if all(k in self.df.columns for k in ['MA5', 'MA20']):
            if (self.df['MA5'].iloc[-2] <= self.df['MA20'].iloc[-2] and 
                self.df['MA5'].iloc[-1] > self.df['MA20'].iloc[-1]):
                signals.append("MA5 crossed above MA20 (Golden Cross - Short term)")
            elif (self.df['MA5'].iloc[-2] >= self.df['MA20'].iloc[-2] and 
                  self.df['MA5'].iloc[-1] < self.df['MA20'].iloc[-1]):
                signals.append("MA5 crossed below MA20 (Death Cross - Short term)")
    
    def _check_macd_signals(self, signals):
        """MACD 신호 확인"""
        if all(k in self.df.columns for k in ['MACD', 'MACD_Signal']):
            if (self.df['MACD'].iloc[-2] <= self.df['MACD_Signal'].iloc[-2] and 
                self.df['MACD'].iloc[-1] > self.df['MACD_Signal'].iloc[-1]):
                signals.append("MACD crossed above Signal line (Bullish)")
            elif (self.df['MACD'].iloc[-2] >= self.df['MACD_Signal'].iloc[-2] and 
                  self.df['MACD'].iloc[-1] < self.df['MACD_Signal'].iloc[-1]):
                signals.append("MACD crossed below Signal line (Bearish)")
    
    def _check_rsi_signals(self, signals):
        """RSI 신호 확인"""
        if 'RSI' in self.df.columns:
            self._check_rsi_overbought_oversold(signals)
            self._check_rsi_reversal(signals)
    
    def _check_rsi_overbought_oversold(self, signals):
        """RSI 과매수/과매도 신호 확인"""
        if (self.df['RSI'].iloc[-2] < 30 and self.df['RSI'].iloc[-1] >= 30):
            signals.append("RSI moved up from oversold territory (Bullish)")
        elif (self.df['RSI'].iloc[-2] > 70 and self.df['RSI'].iloc[-1] <= 70):
            signals.append("RSI moved down from overbought territory (Bearish)")
    
    def _check_rsi_reversal(self, signals):
        """RSI 반전 신호 확인"""
        if len(self.df) >= 3:
            if (self.df['RSI'].iloc[-3] > self.df['RSI'].iloc[-2] and 
                self.df['RSI'].iloc[-1] > self.df['RSI'].iloc[-2] and 
                self.df['RSI'].iloc[-1] < 70):
                signals.append("RSI positive reversal (Bullish)")
            elif (self.df['RSI'].iloc[-3] < self.df['RSI'].iloc[-2] and 
                  self.df['RSI'].iloc[-1] < self.df['RSI'].iloc[-2] and 
                  self.df['RSI'].iloc[-1] > 30):
                signals.append("RSI negative reversal (Bearish)")
    
    def _check_bollinger_band_signals(self, signals):
        """볼린저 밴드 신호 확인"""
        if all(k in self.df.columns for k in ['close', 'BB_Upper', 'BB_Lower']):
            self._check_bb_bounce_signals(signals)
            self._check_bb_double_pattern(signals)
    
    def _check_bb_bounce_signals(self, signals):
        """볼린저 밴드 바운스 신호 확인"""
        if (self.df['close'].iloc[-2] <= self.df['BB_Lower'].iloc[-2] and 
            self.df['close'].iloc[-1] > self.df['BB_Lower'].iloc[-1]):
            signals.append("Price bounced off lower Bollinger Band (Bullish)")
        elif (self.df['close'].iloc[-2] >= self.df['BB_Upper'].iloc[-2] and 
              self.df['close'].iloc[-1] < self.df['BB_Upper'].iloc[-1]):
            signals.append("Price rejected at upper Bollinger Band (Bearish)")
    
    def _check_bb_double_pattern(self, signals):
        """볼린저 밴드 더블 패턴 확인"""
        if len(self.df) >= 3:
            if (self.df['close'].iloc[-3] < self.df['BB_Lower'].iloc[-3] and 
                self.df['close'].iloc[-2] < self.df['BB_Lower'].iloc[-2] and 
                self.df['close'].iloc[-1] > self.df['BB_Lower'].iloc[-1]):
                signals.append("Double bottom at lower Bollinger Band (Strong Bullish)")
            elif (self.df['close'].iloc[-3] > self.df['BB_Upper'].iloc[-3] and 
                  self.df['close'].iloc[-2] > self.df['BB_Upper'].iloc[-2] and 
                  self.df['close'].iloc[-1] < self.df['BB_Upper'].iloc[-1]):
                signals.append("Double top at upper Bollinger Band (Strong Bearish)")
    
    def _check_supertrend_signals(self, signals):
        """수퍼트렌드 신호 확인"""
        if 'Supertrend_Direction' in self.df.columns:
            if (self.df['Supertrend_Direction'].iloc[-2] == -1 and 
                self.df['Supertrend_Direction'].iloc[-1] == 1):
                signals.append("Supertrend changed to up direction (Bullish)")
            elif (self.df['Supertrend_Direction'].iloc[-2] == 1 and 
                  self.df['Supertrend_Direction'].iloc[-1] == -1):
                signals.append("Supertrend changed to down direction (Bearish)")
    
    def _check_stochastic_signals(self, signals):
        """스토캐스틱 신호 확인"""
        if all(k in self.df.columns for k in ['Stoch_K', 'Stoch_D']):
            self._check_stochastic_crossover(signals)
            self._check_stochastic_extremes(signals)
    
    def _check_stochastic_crossover(self, signals):
        """스토캐스틱 교차 신호 확인"""
        if (self.df['Stoch_K'].iloc[-2] <= self.df['Stoch_D'].iloc[-2] and 
            self.df['Stoch_K'].iloc[-1] > self.df['Stoch_D'].iloc[-1]):
            signals.append("Stochastic %K crossed above %D (Bullish)")
        elif (self.df['Stoch_K'].iloc[-2] >= self.df['Stoch_D'].iloc[-2] and 
              self.df['Stoch_K'].iloc[-1] < self.df['Stoch_D'].iloc[-1]):
            signals.append("Stochastic %K crossed below %D (Bearish)")
    
    def _check_stochastic_extremes(self, signals):
        """스토캐스틱 극단치 확인"""
        if (self.df['Stoch_K'].iloc[-1] < 20 and 
            self.df['Stoch_D'].iloc[-1] < 20):
            signals.append("Stochastic in oversold territory (Potential bullish reversal)")
        elif (self.df['Stoch_K'].iloc[-1] > 80 and 
              self.df['Stoch_D'].iloc[-1] > 80):
            signals.append("Stochastic in overbought territory (Potential bearish reversal)")
    
    def _check_pattern_signals(self, signals):
        """패턴 기반 신호 확인"""
        pattern_columns = [
            ('BullishEngulfing', "Bullish Engulfing pattern detected (Bullish)"),
            ('BearishEngulfing', "Bearish Engulfing pattern detected (Bearish)"),
            ('Hammer', "Hammer pattern detected (Bullish)"),
            ('ShootingStar', "Shooting Star pattern detected (Bearish)"),
            ('MorningStar', "Morning Star pattern detected (Strong Bullish)"),
            ('EveningStar', "Evening Star pattern detected (Strong Bearish)"),
            ('ThreeWhiteSoldiers', "Three White Soldiers pattern detected (Strong Bullish)"),
            ('ThreeBlackCrows', "Three Black Crows pattern detected (Strong Bearish)")
        ]
        
        for col, message in pattern_columns:
            if col in self.df.columns and self.df[col].iloc[-1]:
                signals.append(message)
        
        # 잠재적 고점/저점 신호
        self._check_potential_top_bottom_signals(signals)
    
    def _check_potential_top_bottom_signals(self, signals):
        """잠재적 고점/저점 신호 확인"""
        if 'IsPotentialTop' in self.df.columns and self.df['IsPotentialTop'].iloc[-1]:
            signals.append("Potential top formation detected (Bearish)")
        if 'IsPotentialBottom' in self.df.columns and self.df['IsPotentialBottom'].iloc[-1]:
            signals.append("Potential bottom formation detected (Bullish)")
    
    def calculate_risk_level(self):
        """
        다양한 요소를 기반으로 현재 위험 수준을 계산합니다.
        
        Returns:
            str: 위험 수준 (Low, Medium, High)
        """
        if self.df.empty or self.current is None:
            return "Unknown"
        
        risk_factors = []
        
        # 추세 기반 위험
        self._assess_trend_risk(risk_factors)
        
        # 변동성 기반 위험 (ATR)
        self._assess_volatility_risk(risk_factors)
        
        # RSI 기반 위험
        self._assess_rsi_risk(risk_factors)
        
        # 볼린저 밴드 기반 위험
        self._assess_bollinger_risk(risk_factors)
        
        # 평균 위험 점수 계산 및 범주화
        return self._categorize_risk(risk_factors)
    
    def _assess_trend_risk(self, risk_factors):
        """추세 기반 위험 평가"""
        trend = self.calculate_trend_strength()
        if "Strong" in trend:
            if "Uptrend" in trend:
                risk_factors.append(1)  # 강한 상승세에서 낮은 위험
            else:
                risk_factors.append(3)  # 강한 하락세에서 높은 위험
        elif "Moderate" in trend:
            risk_factors.append(2)  # 중간 추세에서 중간 위험
        else:
            risk_factors.append(2.5)  # 약한/횡보 추세에서 약간 높은 위험
    
    def _assess_volatility_risk(self, risk_factors):
        """변동성 기반 위험 평가 (ATR)"""
        if 'ATR14_Pct' in self.current and not pd.isna(self.current['ATR14_Pct']):
            atr_pct = self.current['ATR14_Pct']
            if atr_pct > 3:
                risk_factors.append(3)  # 높은 변동성
            elif atr_pct > 2:
                risk_factors.append(2.5)  # 평균 이상 변동성
            elif atr_pct > 1:
                risk_factors.append(2)  # 평균 변동성
            else:
                risk_factors.append(1)  # 낮은 변동성
    
    def _assess_rsi_risk(self, risk_factors):
        """RSI 기반 위험 평가"""
        if 'RSI' in self.current and not pd.isna(self.current['RSI']):
            rsi = self.current['RSI']
            if rsi > 70 or rsi < 30:
                risk_factors.append(3)  # 과매수/과매도 조건에서 높은 위험
            elif 40 <= rsi <= 60:
                risk_factors.append(1.5)  # 중립 RSI에서 낮은 위험
            else:
                risk_factors.append(2)  # 중간 위험
    
    def _assess_bollinger_risk(self, risk_factors):
        """볼린저 밴드 기반 위험 평가"""
        bb_keys = ['close', 'BB_Upper', 'BB_Lower', 'BB_Middle']
        if all(k in self.current for k in bb_keys):
            if not any(pd.isna(self.current[k]) for k in bb_keys):
                self._assess_bandwidth_risk(risk_factors)
                self._assess_price_position_risk(risk_factors)
    
    def _assess_bandwidth_risk(self, risk_factors):
        """밴드폭 기반 위험 평가"""
        close = self.current['close']
        upper = self.current['BB_Upper']
        lower = self.current['BB_Lower']
        middle = self.current['BB_Middle']
        
        # 밴드폭 계산
        bandwidth = (upper - lower) / middle
        
        if bandwidth > 0.1:
            risk_factors.append(3)  # 넓은 밴드는 높은 변동성과 위험을 나타냄
        elif bandwidth > 0.05:
            risk_factors.append(2)  # 중간 너비 밴드
        else:
            risk_factors.append(1)  # 좁은 밴드는 낮은 변동성과 위험을 나타냄
    
    def _assess_price_position_risk(self, risk_factors):
        """가격 위치 기반 위험 평가"""
        close = self.current['close']
        upper = self.current['BB_Upper']
        lower = self.current['BB_Lower']
        middle = self.current['BB_Middle']
        
        if close > upper or close < lower:
            risk_factors.append(3)  # 밴드 외부에서 높은 위험
        elif abs(close - middle) / middle < 0.01:
            risk_factors.append(1.5)  # 중간 밴드 근처에서 낮은 위험
        else:
            risk_factors.append(2)  # 밴드 내에서 중간 위험
    
    def _categorize_risk(self, risk_factors):
        """위험 점수를 범주형으로 변환"""
        avg_risk = sum(risk_factors) / len(risk_factors) if risk_factors else 2
        
        if avg_risk < 1.67:
            return "Low"
        elif avg_risk < 2.33:
            return "Medium"
        else:
            return "High"


def prepare_analysis_data(df):
    """
    LLM 처리를 위한 기술적 분석 데이터 준비
    
    Args:
        df (DataFrame): 분석할 주가 데이터
        
    Returns:
        dict or None: 분석 데이터 또는 오류 발생 시 None
    """
    if df is None or df.empty:
        return None
    
    try:
        analyzer = TechnicalAnalyzer(df)
        
        # 현재 가격 가져오기 및 최근 변화 계산
        current_price = df['close'].iloc[-1] if 'close' in df.columns else None
        
        # 기간별 가격 변화 계산
        price_changes = calculate_price_changes(df)
        
        # 모든 기술적 분석 구성 요소 수집
        trend = analyzer.calculate_trend_strength()
        sr_levels = analyzer.identify_support_resistance()
        indicators = analyzer.analyze_indicators()
        patterns = analyzer.detect_patterns()
        signals = analyzer.generate_trading_signals()
        risk_level = analyzer.calculate_risk_level()
        
        # 기술적 지표의 서식이 지정된 문자열 표현 생성
        indicator_str = format_indicators_to_string(indicators)
        
        # 분석 데이터 준비
        analysis_data = prepare_formatted_data(
            current_price, price_changes, trend, sr_levels, 
            indicator_str, patterns, signals, risk_level
        )
        
        return analysis_data
        
    except Exception as e:
        print(f"분석 데이터 준비 오류: {e}")
        traceback.print_exc()
        return None


def calculate_price_changes(df):
    """기간별 가격 변화 계산"""
    price_changes = {}
    if 'close' in df.columns:
        if len(df) >= 2:
            price_changes["1_day"] = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100
        if len(df) >= 6:
            price_changes["5_day"] = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100
        if len(df) >= 21:
            price_changes["20_day"] = (df['close'].iloc[-1] / df['close'].iloc[-21] - 1) * 100
    return price_changes


def format_indicators_to_string(indicators):
    """지표 딕셔너리를 서식이 지정된 문자열로 변환"""
    indicator_str = ""
    for indicator, data in indicators.items():
        indicator_str += f"- {indicator}: "
        if isinstance(data, dict):
            indicator_str += ", ".join([f"{k}: {v}" for k, v in data.items()])
        else:
            indicator_str += str(data)
        indicator_str += "\n"
    return indicator_str


def prepare_formatted_data(current_price, price_changes, trend, sr_levels, 
                           indicator_str, patterns, signals, risk_level):
    """분석 데이터를 형식화된 딕셔너리로 준비"""
    # 시장 상태 평가 추가
    market_state = evaluate_market_state(trend, signals, risk_level)
    
    return {
        "current_price": round(current_price, 2) if current_price is not None else None,
        "price_change": {
            "1_day": round(price_changes.get("1_day", 0), 2) if "1_day" in price_changes else None,
            "5_day": round(price_changes.get("5_day", 0), 2) if "5_day" in price_changes else None,
            "20_day": round(price_changes.get("20_day", 0), 2) if "20_day" in price_changes else None
        },
        "price_change_1d": round(price_changes.get("1_day", 0), 2) if "1_day" in price_changes else None,
        "price_change_5d": round(price_changes.get("5_day", 0), 2) if "5_day" in price_changes else None,
        "price_change_20d": round(price_changes.get("20_day", 0), 2) if "20_day" in price_changes else None,
        "trend": trend,
        "support_levels": sr_levels["support"],
        "resistance_levels": sr_levels["resistance"],
        "technical_indicators": indicator_str.strip() if indicator_str else "사용 가능한 기술적 지표 데이터 없음",
        "recent_patterns": patterns,
        "trading_signals": signals,
        "risk_level": risk_level,
        # 추가된 상세 정보
        "market_state": market_state,
        "signal_strength": calculate_signal_strength(signals),
        "pattern_reliability": evaluate_pattern_reliability(patterns),
        "trend_metadata": extract_trend_metadata(trend),
        "support_resistance_metadata": {
            "nearest_support": find_nearest_level(sr_levels["support"], current_price, "below"),
            "nearest_resistance": find_nearest_level(sr_levels["resistance"], current_price, "above"),
            "support_strength": evaluate_level_strength(sr_levels["support"]),
            "resistance_strength": evaluate_level_strength(sr_levels["resistance"])
        },
        "signal_conflicts": identify_signal_conflicts(signals, trend),
        "recommended_timeframe": suggest_trading_timeframe(signals, trend, risk_level)
    }


def evaluate_market_state(trend, signals, risk_level):
    """추세, 신호, 위험 수준을 바탕으로 시장 상태 평가"""
    # 상승 추세 신호 및 하락 추세 신호 개수
    bullish_signals = sum(1 for signal in signals if any(term in signal.lower() for term in ["bullish", "uptrend", "bottom", "support", "golden"]))
    bearish_signals = sum(1 for signal in signals if any(term in signal.lower() for term in ["bearish", "downtrend", "top", "resistance", "death"]))
    
    # 기본 상태 설정
    state = "Neutral"
    confidence = "Medium"
    
    # 추세 기반 상태
    if "Strong Uptrend" in trend:
        state = "Bullish"
        confidence = "High"
    elif "Moderate Uptrend" in trend:
        state = "Bullish"
        confidence = "Medium"
    elif "Weak Uptrend" in trend:
        state = "Bullish"
        confidence = "Low"
    elif "Strong Downtrend" in trend:
        state = "Bearish"
        confidence = "High"
    elif "Moderate Downtrend" in trend:
        state = "Bearish"
        confidence = "Medium"
    elif "Weak Downtrend" in trend:
        state = "Bearish"
        confidence = "Low"
    
    # 신호 기반 확신도 조정
    signal_difference = bullish_signals - bearish_signals
    if signal_difference >= 3 and state == "Bullish":
        confidence = "High"
    elif signal_difference <= -3 and state == "Bearish":
        confidence = "High"
    elif signal_difference >= 2 and state == "Neutral":
        state = "Bullish"
        confidence = "Medium"
    elif signal_difference <= -2 and state == "Neutral":
        state = "Bearish"
        confidence = "Medium"
    
    # 위험 수준 추가
    volatility = "Normal"
    if risk_level == "High":
        volatility = "High"
    elif risk_level == "Low":
        volatility = "Low"
    
    return {
        "state": state,
        "confidence": confidence,
        "volatility": volatility,
        "bullish_signals_count": bullish_signals,
        "bearish_signals_count": bearish_signals
    }


def calculate_signal_strength(signals):
    """신호 강도 계산"""
    if not signals:
        return {
            "strength": "Neutral",
            "bullish_score": 0,
            "bearish_score": 0
        }
    
    bullish_score = 0
    bearish_score = 0
    
    # 신호에 가중치 부여
    for signal in signals:
        signal_lower = signal.lower()
        
        # 강한 신호에 높은 가중치 부여
        if "strong bullish" in signal_lower:
            bullish_score += 3
        elif "strong bearish" in signal_lower:
            bearish_score += 3
        elif "bullish" in signal_lower:
            bullish_score += 2
        elif "bearish" in signal_lower:
            bearish_score += 2
        elif any(term in signal_lower for term in ["golden cross", "bottom", "oversold", "support"]):
            bullish_score += 1
        elif any(term in signal_lower for term in ["death cross", "top", "overbought", "resistance"]):
            bearish_score += 1
    
    # 전체 신호 강도 계산
    net_score = bullish_score - bearish_score
    
    if net_score >= 5:
        strength = "Strong Bullish"
    elif net_score >= 2:
        strength = "Moderate Bullish"
    elif net_score <= -5:
        strength = "Strong Bearish"
    elif net_score <= -2:
        strength = "Moderate Bearish"
    else:
        strength = "Neutral"
    
    return {
        "strength": strength,
        "bullish_score": bullish_score,
        "bearish_score": bearish_score,
        "net_score": net_score
    }


def evaluate_pattern_reliability(patterns):
    """패턴 신뢰도 평가"""
    if not patterns:
        return {
            "reliability": "No patterns detected",
            "confidence": "N/A"
        }
    
    # 고신뢰도 패턴
    high_reliability_patterns = [
        "Morning Star", "Evening Star", "Three White Soldiers", "Three Black Crows",
        "Bullish Engulfing", "Bearish Engulfing"
    ]
    
    # 중간 신뢰도 패턴
    medium_reliability_patterns = [
        "Hammer", "Shooting Star", "Doji", "Piercing Line", "Dark Cloud Cover"
    ]
    
    # 패턴 신뢰도 카운트
    high_reliability_count = sum(1 for pattern in patterns if any(p in pattern for p in high_reliability_patterns))
    medium_reliability_count = sum(1 for pattern in patterns if any(p in pattern for p in medium_reliability_patterns))
    
    # 신뢰도 점수 계산 (0-10)
    total_patterns = len(patterns)
    if total_patterns == 0:
        reliability_score = 0
    else:
        reliability_score = (high_reliability_count * 10 + medium_reliability_count * 5) / total_patterns
    
    # 신뢰도 수준 결정
    if reliability_score >= 7:
        reliability = "High"
    elif reliability_score >= 4:
        reliability = "Medium"
    else:
        reliability = "Low"
    
    return {
        "reliability": reliability,
        "confidence": f"{min(round(reliability_score, 1), 10)}/10",
        "high_reliability_patterns": high_reliability_count,
        "medium_reliability_patterns": medium_reliability_count,
        "total_patterns": total_patterns
    }


def extract_trend_metadata(trend):
    """추세 정보 추출"""
    direction = "Neutral"
    if "Uptrend" in trend:
        direction = "Uptrend"
    elif "Downtrend" in trend:
        direction = "Downtrend"
    
    strength = "Neutral"
    if "Strong" in trend:
        strength = "Strong"
    elif "Moderate" in trend:
        strength = "Moderate"
    elif "Weak" in trend:
        strength = "Weak"
    
    momentum_score = 0
    if direction == "Uptrend":
        if strength == "Strong":
            momentum_score = 3
        elif strength == "Moderate":
            momentum_score = 2
        elif strength == "Weak":
            momentum_score = 1
    elif direction == "Downtrend":
        if strength == "Strong":
            momentum_score = -3
        elif strength == "Moderate":
            momentum_score = -2
        elif strength == "Weak":
            momentum_score = -1
    
    return {
        "direction": direction,
        "strength": strength,
        "momentum_score": momentum_score,
        "raw_trend": trend
    }


def find_nearest_level(levels, current_price, direction="below"):
    """현재 가격에서 가장 가까운 지지/저항 레벨 찾기"""
    if not levels:
        return None
    
    if direction == "below":
        # 현재 가격보다 낮은 레벨 중 가장 높은 레벨 (가장 가까운 지지선)
        below_levels = [level for level in levels if level < current_price]
        return max(below_levels) if below_levels else min(levels)
    else:
        # 현재 가격보다 높은 레벨 중 가장 낮은 레벨 (가장 가까운 저항선)
        above_levels = [level for level in levels if level > current_price]
        return min(above_levels) if above_levels else max(levels)


def evaluate_level_strength(levels):
    """지지/저항 레벨 강도 평가"""
    if not levels or len(levels) < 2:
        return "Unknown"
    
    # 레벨 간 거리 계산
    level_gaps = [abs(levels[i] - levels[i-1]) for i in range(1, len(levels))]
    avg_gap = sum(level_gaps) / len(level_gaps) if level_gaps else 0
    
    # 레벨 조밀도에 따른 강도 평가
    if avg_gap < 0.02 * levels[0]:  # 레벨간 거리가 가격의 2% 미만이면 강함
        return "Strong"
    elif avg_gap < 0.05 * levels[0]:  # 레벨간 거리가 가격의 5% 미만이면 중간
        return "Medium"
    else:
        return "Weak"


def identify_signal_conflicts(signals, trend):
    """신호 간 충돌 확인"""
    if not signals:
        return {
            "has_conflicts": False,
            "conflicts": []
        }
    
    # 상승/하락 추세 방향
    trend_direction = "Neutral"
    if "Uptrend" in trend:
        trend_direction = "Bullish"
    elif "Downtrend" in trend:
        trend_direction = "Bearish"
    
    # 신호 분류
    bullish_signals = [s for s in signals if "Bullish" in s or "bullish" in s.lower()]
    bearish_signals = [s for s in signals if "Bearish" in s or "bearish" in s.lower()]
    
    conflicts = []
    
    # 추세와 신호 간 충돌
    if trend_direction == "Bullish" and len(bearish_signals) > len(bullish_signals):
        conflicts.append({
            "type": "Trend-Signal Conflict",
            "description": f"Bullish trend with more bearish signals ({len(bearish_signals)}) than bullish signals ({len(bullish_signals)})"
        })
    elif trend_direction == "Bearish" and len(bullish_signals) > len(bearish_signals):
        conflicts.append({
            "type": "Trend-Signal Conflict",
            "description": f"Bearish trend with more bullish signals ({len(bullish_signals)}) than bearish signals ({len(bearish_signals)})"
        })
    
    # 신호 간 충돌 (상승/하락 신호가 모두 있으면)
    if bullish_signals and bearish_signals:
        conflicts.append({
            "type": "Mixed Signals",
            "description": f"Both bullish ({len(bullish_signals)}) and bearish ({len(bearish_signals)}) signals present",
            "bullish_signals": bullish_signals[:3],  # 최대 3개만 표시
            "bearish_signals": bearish_signals[:3]   # 최대 3개만 표시
        })
    
    return {
        "has_conflicts": len(conflicts) > 0,
        "conflicts": conflicts,
        "dominant_direction": "Bullish" if len(bullish_signals) > len(bearish_signals) else "Bearish" if len(bearish_signals) > len(bullish_signals) else "Neutral"
    }


def suggest_trading_timeframe(signals, trend, risk_level):
    """적합한 거래 시간대 제안"""
    # 신호 및 추세 강도를 기반으로 한 기본 시간대
    if "Strong" in trend:
        base_timeframe = "Medium-term (1-2 weeks)"
    elif "Moderate" in trend:
        base_timeframe = "Short-term (3-5 days)"
    else:
        base_timeframe = "Very short-term (1-2 days)"
    
    # 패턴 기반 시간대 조정
    pattern_signals = [s for s in signals if any(pattern in s for pattern in ["Star", "Soldiers", "Crows", "Engulfing"])]
    if pattern_signals:
        if "Strong" in trend:
            base_timeframe = "Medium to Long-term (2-4 weeks)"
        else:
            base_timeframe = "Medium-term (1-2 weeks)"
    
    # 위험 수준에 따른 조정
    timeframe_adjusted = base_timeframe
    if risk_level == "High":
        # 높은 위험에서는 더 짧은 시간대 제안
        if "Medium" in base_timeframe or "Long" in base_timeframe:
            timeframe_adjusted = "Short-term (3-5 days)"
    elif risk_level == "Low":
        # 낮은 위험에서는 더 긴 시간대 제안
        if "Short" in base_timeframe and not "Very" in base_timeframe:
            timeframe_adjusted = "Medium-term (1-2 weeks)"
        elif "Very short" in base_timeframe:
            timeframe_adjusted = "Short-term (3-5 days)"
    
    return {
        "suggested_timeframe": timeframe_adjusted,
        "base_timeframe": base_timeframe,
        "risk_adjusted": timeframe_adjusted != base_timeframe
    }