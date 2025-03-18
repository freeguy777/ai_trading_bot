import os
import pandas as pd
from typing import List, Dict

# LangChain Google Gemini imports
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

# 0. ----- 헬퍼 --------------------------------------------------

def _news_to_df(news: List[Dict]) -> pd.DataFrame:
    """list[dict] → pivot(sentiment_score) DataFrame"""
    df = (pd.DataFrame(news)
            .assign(date=lambda d: pd.to_datetime(d["published_at"]).dt.date)
            .pivot_table(values="score",
                         index="date",
                         columns="ticker",
                         aggfunc="mean")
            .add_prefix("NEWS_"))
    return df

# 1. ----- 핵심 함수 --------------------------------------------

def run_analysis(feat: pd.DataFrame,
                 news: List[Dict]) -> Dict[str, pd.DataFrame]:
    """최소 분석: 상관 + 간단 예측 + 리스크 라벨 (이벤트 미포함)"""
    # ① 날짜 index 정렬
    feat = feat.set_index(pd.to_datetime(feat["date"]))
    feat = feat.drop(columns="date")

    # ② 뉴스 매핑
    feat_daily = feat.resample("D").ffill()
    combined = (
        feat_daily
        .join(_news_to_df(news), how="left")
        .fillna(0)
    )

    # ③ 상관관계 (pearson)
    corr = combined.corr().round(2)

    # ④ 퀵‑예측: 마지막 5일 simple moving average
    forecast = (
        combined.tail(5)
                .mean()
                .to_frame(name="M5_forecast")
                .T
    )

    # ⑤ 리스크 라벨: 20일 ATR/가격 > 0.02 → 'high'
    atr = feat["SPY"].pct_change().rolling(20).std()
    risk = pd.DataFrame({
        "volatility": ["high" if atr.iloc[-1] > 0.02 else "normal"]
    })

    return {"correlation": corr, "forecast": forecast, "risk": risk}

# 2. ----- LLM 분석 통합 (Google Gemini) ------------------------

system_template = """
당신은 월스트리트에서 수십 년간 경험을 쌓은 최고 수준의 퀀트 전략가이자 포트폴리오 매니저입니다.
제공된 다양한 데이터(뉴스 감성, 지정학적 이벤트, 시계열 데이터, 통계 분석 결과)를 단순 요약하는 것을 넘어, 데이터 포인트들을 유기적으로 연결하고 그 이면에 숨겨진 의미와 패턴을 통찰력 있게 분석해야 합니다.

**주요 임무:**
1.  **데이터 통합 분석:** 개별 데이터 소스의 정보를 통합하여 거시경제 및 시장 상황에 대한 종합적인 그림을 그립니다. 단편적인 정보가 아닌, 전체적인 맥락 속에서 각 데이터의 의미를 해석하십시오.
2.  **인과관계 및 영향 분석:** 뉴스 감성, 지정학적 이벤트가 특정 자산(예: SPY, TLT 등) 및 거시 지표(예: GDP, CPI)에 미치는 단기적/장기적 영향을 논리적으로 추론합니다. 상관관계 분석 결과를 바탕으로 관계의 강도와 잠재적 동인을 설명하십시오.
3.  **미래 예측 및 시나리오:** 제시된 예측 데이터(M5_forecast)의 신뢰도를 평가하고, 이를 바탕으로 가능한 시장 시나리오를 제시합니다. 리스크 분석 결과를 고려하여 각 시나리오의 발생 가능성과 잠재적 파급 효과를 논하십시오.
4.  **투자 전략 제시:** 분석된 인사이트를 바탕으로, 명확한 논리적 근거를 갖춘 투자 의견(예: 매수, 매도, 중립) 또는 자산 배분 전략을 제시합니다. 잠재적 위험 요인과 이에 대한 대응 방안도 함께 고려해야 합니다.
5.  **비판적 사고:** 제공된 데이터의 한계점이나 잠재적 편향성을 인지하고, 분석 결과 해석 시 이를 명시적으로 고려합니다. 확신에 찬 어조보다는, 논리적 추론과 데이터에 기반한 균형 잡힌 시각을 유지하십시오.

결과는 전문 투자 보고서 형식으로, 명확하고 간결하며 실행 가능한 인사이트를 담아 작성해야 합니다.
"""

human_template = """
**입력 데이터:**

▶ **뉴스 데이터 (Ticker | Sentiment | Score | Headline):**
{news}

▶ **지정학 데이터 (Country | Event Type | Title | Date | URL):**
{geopolitical}

▶ **시장 및 거시 데이터 (전처리 완료, 날짜별 시계열):**
{preprocessed}

▶ **상관관계 분석 결과 (Correlation Matrix):**
{correlation}

▶ **단기 예측 결과 (5일 이동평균 기반):**
{forecast}

▶ **리스크 분석 결과 (SPY 변동성 기반):**
{risk}

**요청 보고서 형식:**

위 6가지 입력 데이터를 심층적으로 분석하여, 아래 항목 순서대로 투자 보고서를 작성해 주십시오. 각 항목에서는 반드시 논리적 근거와 데이터 분석 결과를 명확히 제시해야 합니다.

1.  **종합 요약 (Executive Summary):** 현재 시장 상황에 대한 핵심 진단과 주요 분석 결과, 그리고 최종 투자 스탠스를 간결하게 요약합니다.

2.  **뉴스 감성 영향 분석:**
    *   주요 뉴스들의 전반적인 감성 톤(긍정/부정/중립)과 그 강도를 평가합니다.
    *   특정 뉴스(긍정/부정 스코어가 높은)가 관련 자산(Ticker 기준)의 단기 가격 변동 또는 시장 심리에 미칠 수 있는 잠재적 영향을 분석합니다. (예: 특정 기업 뉴스 -> 해당 ETF 영향)
    *   감성 데이터와 실제 시장 움직임 간의 일치/불일치 여부를 논합니다.

3.  **지정학적 이벤트 리스크 평가:**
    *   주요 지정학적 이벤트의 성격과 잠재적 파급력을 분석합니다.
    *   이 이벤트들이 특정 국가 경제, 글로벌 공급망, 원자재 가격(USO 등), 또는 안전자산 선호도(GLD, TLT 등)에 미칠 수 있는 영향을 구체적으로 명시합니다.
    *   이벤트 발생 시 예상되는 시장 반응 시나리오를 제시합니다.

4.  **상관관계 분석 및 해석:**
    *   제시된 상관관계 매트릭스에서 주목할 만한 강한 양(+) 또는 음(-)의 상관관계를 식별합니다. (예: SPY와 TLT의 관계, GDP와 특정 ETF의 관계 등)
    *   이러한 상관관계의 경제적 또는 시장 논리적 배경을 설명합니다. (예: 경기 확장 시 주식과 채권의 관계 변화)
    *   포트폴리오 다변화 관점에서 이 상관관계가 가지는 함의를 분석합니다.

5.  **시장 예측 및 리스크 평가:**
    *   제공된 단기 예측(5일 이동평균)의 의미를 해석하고, 이것이 현재 시장 추세를 반영하는지 평가합니다. (단, 이 예측의 한계점을 명확히 인지)
    *   단기 예측 결과와 리스크 분석 결과(변동성)를 종합하여 향후 시장의 잠재적 방향성 및 위험 수준을 진단합니다.
    *   예측 및 리스크 분석 결과를 바탕으로 단기적인 투자 전략 조정 필요성을 논합니다.

6.  **투자 결론 및 전략 제언:**
    *   위 모든 분석(뉴스 감성, 지정학, 상관관계, 예측, 리스크)을 종합하여 현재 시장 상황에 대한 최종적인 판단을 내립니다.
    *   구체적인 투자 스탠스(예: 미국 주식 비중 확대/축소, 채권 투자 전략, 특정 섹터/자산 선호 등)를 명확한 논리와 함께 제시합니다.
    *   제시된 전략의 잠재적 위험 요인과 모니터링해야 할 주요 지표들을 언급합니다.
"""

prompt = PromptTemplate(
    template=system_template + "\n" + human_template,
    input_variables=["news", "geopolitical", "preprocessed", "correlation", "forecast", "risk"]
)

# 2.5 ----- LLM 보고서 생성 함수 ------------------------------------
def generate_investment_report(
    feat: pd.DataFrame,
    scored_news: List[Dict],
    events: List[Dict],
    cfg: Dict
) -> str:
    """LLM을 사용하여 투자 분석 보고서를 생성합니다."""
    gemini_api_key = cfg.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Configuration missing GEMINI_API_KEY")

    # LLM 초기화 (Gemini API 키 사용)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17",
        temperature=0,
        api_key=gemini_api_key
    )

    # 기본 분석 실행 (보고서 생성에 필요)
    analysis_out = run_analysis(feat.copy(), scored_news) # 원본 feat 변경 방지

    # 문자열 포매팅
    news_str = "\n".join([
        f"{art['ticker']} | {art.get('sentiment', 'N/A')} | {art.get('score', 0.0):+.2f} | {art['title']}"
        for art in scored_news
    ])
    events_str = "\n".join([
        f"{e['country']}, {e['event']}, {e['title']}, {e.get('date', 'N/A')}, {e['url']}"
        for e in events
    ])
    preprocessed_str = feat.to_markdown(index=True)
    correlation_str = analysis_out['correlation'].to_markdown()
    forecast_str = analysis_out['forecast'].to_markdown()
    risk_str = analysis_out['risk'].to_markdown()

    # LLM 실행 (RunnableSequence와 invoke 사용)
    input_data = {
        "news": news_str,
        "geopolitical": events_str,
        "preprocessed": preprocessed_str,
        "correlation": correlation_str,
        "forecast": forecast_str,
        "risk": risk_str
    }
    # .run() 대신 .invoke() 사용하고 결과에서 .content 추출
    response = (prompt | llm).invoke(input_data)
    report = response.content

    return report

# 3. ----- CLI 용 (테스트용) ----------------------------------------
if __name__ == "__main__":
    import os
    import sys
    #####절대 경로 임포트 사용
    current_dir = os.path.dirname(os.path.abspath(__file__))
    agents_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(agents_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from macro_economic_agent.config_loader import ConfigLoader
    from macro_economic_agent.ingestion.news_fetcher import NewsFetcher
    from macro_economic_agent.processing.sentiment_analyzer import SentimentAnalyzer
    from macro_economic_agent.ingestion.market_data import get_price_series_yf,get_macro_series_fred
    from macro_economic_agent.ingestion.geopolitical_events import GeoPoliticalEventsFetcher
    from macro_economic_agent.processing.preprocessor import DataPreprocessor

    # 데이터 로딩
    cfg = ConfigLoader.load()
    news  = NewsFetcher(["SPY", "TLT", "UUP"]).fetch_news(limit_per_ticker=3)
    # API 키를 여기서 미리 확인해도 좋음
    gemini_api_key = cfg.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Configuration missing GEMINI_API_KEY in test block")
    scored_news = SentimentAnalyzer(gemini_api_key).analyze_sentiment(news)
    price = get_price_series_yf(["SPY", "TLT", "GLD", "USO", "UUP"], start="2024-01-01")
    macro = get_macro_series_fred(["GDP", "CPIAUCSL", "FEDFUNDS"])
    events = GeoPoliticalEventsFetcher(cfg).fetch_events()
    proc  = DataPreprocessor(
        log_cols=["SPY", "TLT", "CPIAUCSL"],
        zscore_cols=["GDP", "FEDFUNDS"],
    )
    feat = proc.transform(price, macro)

    # 보고서 생성 함수 호출 및 출력 (테스트)
    report = generate_investment_report(feat, scored_news, events, cfg)
    print("--- Generated Report (Test) ---")
    print(report)
