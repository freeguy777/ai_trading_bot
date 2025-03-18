import os
import sys
import logging
import json
from dotenv import load_dotenv

#####절대 경로 임포트 사용
current_dir = os.path.dirname(os.path.abspath(__file__))
agents_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(agents_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from agents.chart_analysis_agent.get_chart import KisConfig, analyze_chart
from agents.chart_analysis_agent.get_technical_indicators import add_technical_indicators

# 분리된 데이터 분석 및 LLM 모듈 임포트
from agents.chart_analysis_agent.technical_analysis import prepare_analysis_data
from agents.chart_analysis_agent.llm_analysis import analyze_with_llm

load_dotenv(dotenv_path="config/.env")  # 상대 경로 사용

def print_technical_analysis(analysis_data):
    """기술적 데이터 분석 결과 출력 함수"""
    if analysis_data:
        print(f"현재 가격: {analysis_data['current_price']}")
        print(f"추세: {analysis_data['trend']}")
        print(f"위험 수준: {analysis_data['risk_level']}")
        print(f"감지된 패턴: {', '.join(analysis_data['recent_patterns']) if analysis_data['recent_patterns'] else '없음'}")
        print(f"거래 신호: {', '.join(analysis_data['trading_signals'][:3]) if analysis_data['trading_signals'] else '없음'}")
    else:
        print("기술적 데이터 분석 중 오류가 발생했습니다.")

def print_ai_analysis_results(result, analysis_data):
    """AI 분석 결과 출력 함수"""    
    if "executive_summary" in result:
        print("\n===== AI 분석 요약 =====")
        print(result["executive_summary"])
    
    # 추가 기술 분석 메타데이터 출력
    if "analysis_data" in result and "market_state" in analysis_data:
        print("\n===== 추가 기술 분석 정보 =====")
        print(f"시장 상태: {analysis_data['market_state']['state']} (확신도: {analysis_data['market_state']['confidence']})")
        print(f"신호 강도: {analysis_data['signal_strength']['strength']} (상승: {analysis_data['signal_strength']['bullish_score']}, 하락: {analysis_data['signal_strength']['bearish_score']})")
        print(f"추세 정보: {analysis_data['trend_metadata']['direction']} ({analysis_data['trend_metadata']['strength']})")
        print(f"권장 거래 시간대: {analysis_data['recommended_timeframe']['suggested_timeframe']}")
        
        if analysis_data['signal_conflicts']['has_conflicts']:
            print("\n[주의] 신호 충돌 감지:")
            for conflict in analysis_data['signal_conflicts']['conflicts']:
                print(f"- {conflict['description']}")
    
    if "error" in result:
        print("\n오류 발생:", result["error"])

def api_key_init():
    api_key = os.getenv("KIS_API_KEY")
    api_secret = os.getenv("KIS_API_SECRET")
    account_no_full = os.getenv("KIS_INVEST_ACCOUNT_NO_FULL")
    config = KisConfig(api_key=api_key, api_secret=api_secret, account_no_full=account_no_full)

    return config

def main(ticker): 
    logging.info(os.getcwd())
    config = api_key_init() 
    exchange = "나스닥"

    #print("\n===== 일봉 차트 =====")
    df_indicator = analyze_chart(ticker, exchange, config)
    #print(df_indicator.head(3).to_string())

    #print("\n===== 보조 지표 추가 차트 =====")
    df_technical_indicators = add_technical_indicators(df_indicator)
    #print(df_technical_indicators.head(3).to_string())

    #print("\n===== 기술적 데이터 분석 =====")
    analysis_data = prepare_analysis_data(df_technical_indicators)
    #print_technical_analysis(analysis_data)

    #print("\n===== AI 분석 결과 생성 =====")
    result = analyze_with_llm(analysis_data=analysis_data, api_key=os.getenv("GEMINI_API_KEY"), model="gemini-2.5-flash-preview-04-17", structured_output=True)
    #print_ai_analysis_results(result, analysis_data)

    return result["executive_summary"]

if __name__ == "__main__":
    main("AAPL")  # 삼성전자 티커