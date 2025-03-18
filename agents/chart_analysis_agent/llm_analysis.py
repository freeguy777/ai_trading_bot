"""
LangChain 기반 주가 분석 모듈

이 모듈은 LangChain을 사용하여 주가 데이터에 대한 
인공지능 기반 분석 및 예측을 제공합니다.
"""

import os
import traceback
from typing import List, Dict, Optional, Union
from dotenv import load_dotenv

# LangChain 및 OpenAI 관련 라이브러리
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import SequentialChain, LLMChain
from pydantic import BaseModel, Field

# 상수 정의
DEFAULT_MODEL = "gemini-2.5-flash-preview-04-17"
DEFAULT_TEMPERATURE = 0.2


# ---------------------- Pydantic 모델 정의 ----------------------

class TechnicalIndicatorAnalysis(BaseModel):
    """기술적 지표 분석 결과를 위한 Pydantic 모델"""
    rsi: Dict = Field(default={}, description="RSI 분석 (값 및 상태 포함)")
    macd: Dict = Field(default={}, description="MACD 분석 (값, 신호 및 상태 포함)")
    stochastic: Dict = Field(default={}, description="스토캐스틱 오실레이터 분석")
    bollinger_bands: Dict = Field(default={}, description="볼린저 밴드 분석")
    supertrend: Dict = Field(default={}, description="수퍼트렌드 분석")
    atr: Dict = Field(default={}, description="평균 실제 범위 분석")


class SupportResistance(BaseModel):
    """지지선 및 저항선 정보를 위한 Pydantic 모델"""
    support_levels: List[float] = Field(default=[], description="주요 지지 수준")
    resistance_levels: List[float] = Field(default=[], description="주요 저항 수준")


class StockAnalysisSummary(BaseModel):
    """주식 분석 요약을 위한 Pydantic 모델"""
    current_position: str = Field(..., description="현재 시장 포지션 요약")
    key_levels: str = Field(..., description="주시해야 할 주요 지지/저항 수준")
    signals_patterns: str = Field(..., description="중요한 신호 및 패턴 요약")
    strategy: str = Field(..., description="권장 거래 전략")


class TradeRecommendation(BaseModel):
    """거래 추천을 위한 Pydantic 모델"""
    action: str = Field(..., description="권장 조치 (매수, 매도, 보유)")
    entry_price: Optional[float] = Field(None, description="제안된 진입 가격")
    stop_loss: Optional[float] = Field(None, description="제안된 손절 수준")
    take_profit: Optional[float] = Field(None, description="제안된 목표 수익 수준")
    risk_reward_ratio: Optional[float] = Field(None, description="계산된 위험-수익 비율")
    confidence: str = Field(..., description="신뢰 수준 (낮음, 중간, 높음)")


class StockAnalysis(BaseModel):
    """전체 주식 분석 결과를 위한 Pydantic 모델"""
    current_price: float = Field(..., description="현재 주가")
    price_changes: Dict = Field(..., description="최근 가격 변동 (백분율)")
    trend: str = Field(..., description="현재 추세 방향 및 강도")
    support_resistance: SupportResistance = Field(..., description="지지 및 저항 수준")
    technical_indicators: Dict = Field(..., description="기술적 지표 분석")
    recent_patterns: List[str] = Field(default=[], description="최근 감지된 차트 패턴")
    trading_signals: List[str] = Field(default=[], description="잠재적 거래 신호")
    risk_level: str = Field(..., description="현재 위험 수준 (낮음, 중간, 높음)")
    short_term_outlook: str = Field(..., description="단기 시장 전망")
    medium_term_outlook: str = Field(..., description="중기 시장 전망")
    summary: StockAnalysisSummary = Field(..., description="분석 간결 요약")
    trade_recommendation: TradeRecommendation = Field(..., description="특정 거래 추천")


# ---------------------- LangChain 프롬프트 템플릿 ----------------------

STOCK_ANALYSIS_TEMPLATE = """
당신은 정량적 분석 및 거래 전략에 전문 지식을 갖춘 전문 주식 시장 기술 분석가입니다.

다음 기술 데이터를 분석하고 종합적인 분석을 제공하세요. 기술적 지표, 차트 패턴 및 잠재적 거래 기회에 중점을 둡니다.

기술적 분석 데이터:

현재 가격: {current_price}

최근 가격 변화:
- 1일: {price_change_1d}%
- 5일: {price_change_5d}% (가능한 경우)
- 20일: {price_change_20d}% (가능한 경우)

현재 추세: {trend}

지지 수준: {support_levels}
저항 수준: {resistance_levels}

기술적 지표:
{technical_indicators}

감지된 최근 패턴:
{recent_patterns}

거래 신호:
{trading_signals}

위험 수준: {risk_level}

이 기술적 분석을 기반으로 아래 지정된 형식으로 종합적인 분석을 제공하세요.

{format_instructions}
"""

SIMPLE_ANALYSIS_TEMPLATE = """
당신은 정량적 분석 및 거래 전략에 전문 지식을 갖춘 전문 주식 시장 기술 분석가입니다.

다음 기술 데이터를 분석하고 종합적인 분석을 제공하세요. 기술적 지표, 차트 패턴 및 잠재적 거래 기회에 중점을 둡니다.

기술적 분석 데이터:

현재 가격: {current_price}

최근 가격 변화:
- 1일: {price_change_1d}%
- 5일: {price_change_5d}% (가능한 경우)
- 20일: {price_change_20d}% (가능한 경우)

현재 추세: {trend}

지지 수준: {support_levels}
저항 수준: {resistance_levels}

기술적 지표:
{technical_indicators}

감지된 최근 패턴:
{recent_patterns}

거래 신호:
{trading_signals}

위험 수준: {risk_level}

이 기술적 분석을 기반으로 다음을 제공하세요:
1. 현재 시장 포지션 요약
2. 주시해야 할 주요 지지 및 저항 수준
3. 기술적 지표 분석 및 그것이 시사하는 바
4. 감지된 패턴이나 신호 해석
5. 단기 전망 (1-5일)
6. 중기 전망 (1-3주)
7. 위험 평가 및 포지션 관리 권장 사항
8. 전반적인 결론 및 거래 전략 권장 사항

적절한 제목과 구조로 명확하고 전문적인 형식으로 분석을 제공하세요.
"""

EXECUTIVE_SUMMARY_TEMPLATE = """
아래 상세 주식 분석을 기반으로 전문 트레이더가 투자 결정을 내릴 수 있도록 종합적인 기술 분석 요약을 제공하세요. 
포괄적이고 정보가 풍부하며 실행 가능한 분석을 제공하세요.

{detailed_analysis}

다음과 같은 구조로 상세한 분석 요약을 작성하세요:

## 1. 시장 포지션 및 가격 동향
- 현재 가격과 주요 이동평균선(MA5, MA20, MA60 등) 대비 위치
- 최근 가격 변동성 및 추세 강도 분석
- 상승/하락/횡보 추세의 명확한 특성과 지속 기간
- 거래량 분석 및 추세 확인 여부

## 2. 주요 기술적 지표 분석
- RSI: 현재 수치, 과매수/과매도 상태, 다이버전스 존재 여부
- MACD: 히스토그램 방향, 신호선 교차, 모멘텀 강도
- 스토캐스틱: 과매수/과매도 여부, %K/%D 교차 상황
- 볼린저 밴드: 현재 밴드 폭, 가격 위치, 스퀴즈/확장 상태
- 기타 관련 지표의 종합적 해석

## 3. 지지/저항 구간 상세 분석
- 중요 지지 수준(최소 3개)과 각 수준별 강도 평가
- 주요 저항 수준(최소 3개)과 돌파 가능성 평가
- 심리적 가격대 및 과거 반응 구간 분석
- 피보나치 수준 또는 피벗 포인트 관련 분석

## 4. 차트 패턴 및 신호 상세 분석
- 현재 진행 중인 차트 패턴(헤드앤숄더, 삼각형, 쐐기형 등) 분석
- 캔들스틱 패턴의 신뢰도 및 과거 성공률 평가
- 강세/약세 신호들 사이의 충돌 여부와 우선순위 평가
- 신호 강도 및 확률적 성공 가능성 분석

## 5. 시나리오 분석
- 강세 시나리오: 촉발 요인, 목표 가격, 확률 평가
- 약세 시나리오: 촉발 요인, 하락 폭, 확률 평가
- 횡보 시나리오: 지속 가능성 및 돌파 방향 예측
- 각 시나리오별 거래 계획 및 대응 전략

## 6. 리스크 평가 및 관리
- 현재 위험/보상 비율 정량적 평가
- 손절/목표가 상세 설정 근거
- 최대 허용 리스크 및 포지션 사이징 제안
- 리스크 관리를 위한 트레일링 스탑 또는 부분 익절 전략

## 7. 실행 계획
- 권장 조치(매수/매도/관망)와 명확한 진입 조건
- 최적 진입 구간 및 가격 레벨
- 정밀한 손절 및 목표 가격(복수의 목표 포함)
- 포지션 관리 및 스케일링 전략

## 8. 종합 평가
- 단기(1-3일), 중기(1-2주), 장기(1-3개월) 전망
- 기술적 신뢰도 점수(1-10)
- 최종 권장사항 및 주의사항
- 추가 모니터링이 필요한 핵심 지표 또는 이벤트

이 요약은 전문 트레이더가 즉시 활용할 수 있도록 정확하고 구체적이며 실행 가능한 정보를 제공해야 합니다. 
주관적 해석보다는 객관적인 데이터와 지표에 기반한 분석을 제공하세요.
"""


# ---------------------- LangChain 구성 요소 ----------------------

def create_stock_analysis_chain(llm):
    """
    상세한 주식 분석을 위한 LangChain 체인 생성
    
    Args:
        llm: 사용할 언어 모델 인스턴스
        
    Returns:
        Chain: 주식 분석을 위한 LangChain 체인
    """
    # 출력 파서 생성
    parser = PydanticOutputParser(pydantic_object=StockAnalysis)
    
    # 파서 지침이 포함된 프롬프트 생성
    stock_analysis_prompt = PromptTemplate(
        template=STOCK_ANALYSIS_TEMPLATE,
        input_variables=[
            "current_price", "price_change_1d", "price_change_5d", "price_change_20d", 
            "trend", "support_levels", "resistance_levels", "technical_indicators", 
            "recent_patterns", "trading_signals", "risk_level"
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # 체인 생성
    stock_analysis_chain = stock_analysis_prompt | llm | parser
    return stock_analysis_chain


def create_simple_analysis_chain(llm):
    """
    구조화된 출력이 필요하지 않을 때 텍스트 기반 분석을 위한 간단한 체인 생성
    
    Args:
        llm: 사용할 언어 모델 인스턴스
        
    Returns:
        Chain: 텍스트 기반 분석을 위한 LangChain 체인
    """
    simple_prompt = PromptTemplate(
        template=SIMPLE_ANALYSIS_TEMPLATE,
        input_variables=[
            "current_price", "price_change_1d", "price_change_5d", "price_change_20d", 
            "trend", "support_levels", "resistance_levels", "technical_indicators", 
            "recent_patterns", "trading_signals", "risk_level"
        ]
    )
    
    simple_chain = simple_prompt | llm | StrOutputParser()
    return simple_chain


def create_executive_summary_chain(llm):
    """
    간결한 임원 요약을 생성하기 위한 체인 생성
    
    Args:
        llm: 사용할 언어 모델 인스턴스
        
    Returns:
        Chain: 임원 요약을 위한 LangChain 체인
    """
    summary_prompt = PromptTemplate(
        template=EXECUTIVE_SUMMARY_TEMPLATE,
        input_variables=["detailed_analysis"]
    )
    
    summary_chain = summary_prompt | llm | StrOutputParser()
    return summary_chain


# ---------------------- LLM 기반 분석 ----------------------

def setup_api_key(api_key):
    """API 키 설정 및 유효성 검사"""
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        return True
    elif not os.getenv("GEMINI_API_KEY"):
        load_dotenv(dotenv_path="config/.env")  # 상대 경로 사용
        return os.getenv("GEMINI_API_KEY") is not None
    return True


def perform_analysis(llm, analysis_data, structured_output):
    """분석 체인 생성 및 실행"""
    result = {}
    
    if structured_output:
        # 구조화된 출력
        analysis_chain = create_stock_analysis_chain(llm)
        analysis_result = analysis_chain.invoke(analysis_data)
        
        # Pydantic 모델 -> dict 변환
        result["analysis_result"] = analysis_result.dict()
        
        # 임원 요약 생성 (상세 정보 강화)
        summary_chain = create_executive_summary_chain(llm)
        
        # 상세 분석 데이터 보강
        enhanced_data = {
            "detailed_analysis": str(analysis_result),
            # 원본 기술적 지표 데이터 추가
            "raw_indicators": analysis_data["technical_indicators"],
            "recent_patterns_raw": analysis_data["recent_patterns"],
            "trading_signals_raw": analysis_data["trading_signals"],
            "risk_level_raw": analysis_data["risk_level"],
            "support_levels_raw": analysis_data["support_levels"],
            "resistance_levels_raw": analysis_data["resistance_levels"],
            "price_changes": {
                "1d": analysis_data["price_change_1d"],
                "5d": analysis_data["price_change_5d"],
                "20d": analysis_data["price_change_20d"],
            }
        }
        
        result["executive_summary"] = summary_chain.invoke({"detailed_analysis": str(analysis_result)})
        
        # 원본 분석 데이터도 결과에 추가
        result["technical_indicators"] = analysis_data["technical_indicators"]
        result["price_changes"] = {
            "1d": analysis_data["price_change_1d"],
            "5d": analysis_data["price_change_5d"],
            "20d": analysis_data["price_change_20d"],
        }
        result["support_resistance"] = {
            "support": analysis_data["support_levels"],
            "resistance": analysis_data["resistance_levels"]
        }
        result["trend"] = analysis_data["trend"]
        result["signals"] = analysis_data["trading_signals"]
    else:
        # 텍스트 기반 출력
        simple_chain = create_simple_analysis_chain(llm)
        analysis_text = simple_chain.invoke(analysis_data)
        result["analysis_text"] = analysis_text
    
    return result


def analyze_with_llm(analysis_data, api_key=None, model=DEFAULT_MODEL, structured_output=True):
    """
    LangChain을 사용하여 기술 분석 데이터에 대한 LLM 분석 수행
    
    Args:
        analysis_data (dict): 기술 분석에서 생성된 분석 데이터
        api_key (str, optional): GOOGLE API 키
        model (str, optional): 사용할 GOOGLE 모델
        structured_output (bool, optional): 구조화된 출력 사용 여부
        
    Returns:
        dict: 분석 결과를 포함하는 딕셔너리
    """
    try:
        # API 키 설정
        if not setup_api_key(api_key):
            return {"error": "GOOGLE API 키가 필요합니다"}
        
        # 분석 데이터 유효성 검증
        if analysis_data is None:
            return {"error": "유효한 분석 데이터가 필요합니다"}
        
        # LLM 설정
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=DEFAULT_TEMPERATURE,
            api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # 분석 수행
        result = perform_analysis(llm, analysis_data, structured_output)
        
        # 분석 데이터를 결과에 추가
        result["analysis_data"] = analysis_data
        
        return result
        
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}