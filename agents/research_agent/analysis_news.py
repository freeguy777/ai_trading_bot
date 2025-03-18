import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
from datetime import datetime
import base64
from io import BytesIO
import os
import google.generativeai as genai
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv(dotenv_path="config/.env")  # 상대 경로 사용

# Google Gemini API 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def analyze_news_sentiment(news_articles):
    """
    뉴스 기사의 감성을 분석합니다.
    
    Args:
        news_articles (list): 뉴스 기사 목록
        
    Returns:
        tuple: (감성이 포함된 뉴스 기사, 감성 요약)
    """
    # VADER 감성 분석을 위한 어휘 사전 다운로드 (아직 다운로드되지 않은 경우)
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
    
    # VADER 감성 분석기 초기화
    sid = SentimentIntensityAnalyzer()
    
    # 날짜별 감성 추적 준비
    sentiment_by_date = {}
    
    # 각 기사의 감성 분석
    for article in news_articles:
        # 감성 분석을 위해 제목과 스니펫 조합
        text = article.get("title", "") + " " + article.get("summary", "")
        
        # VADER 감성 점수 획득
        vader_sentiment = sid.polarity_scores(text)
        
        # TextBlob을 사용한 추가 감성 분석
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # 기사에 감성 점수 추가
        article["sentiment"] = {
            "vader": vader_sentiment,
            "textblob": {
                "polarity": textblob_polarity,
                "subjectivity": textblob_subjectivity
            },
            # 두 감성 분석 방법의 가중 평균 (VADER 70%, TextBlob 30%)
            "compound": vader_sentiment["compound"] * 0.7 + textblob_polarity * 0.3
        }
        
        # 날짜별 감성 추적
        date = article.get("date", "Unknown")
        if date not in sentiment_by_date:
            sentiment_by_date[date] = []
        sentiment_by_date[date].append(article["sentiment"]["compound"])
    
    # 종합 감성 계산
    if news_articles:
        # 가중치 감성 계산 (최신 뉴스가 더 높은 가중치를 가짐)
        total_weight = 0
        weighted_compound_sum = 0
        
        for i, article in enumerate(news_articles):
            # 최신성에 기반한 가중치 계산 (기사가 날짜별로 정렬되어 있다고 가정)
            weight = 1 / (i + 1)**0.5  # 제곱근 감소
            total_weight += weight
            weighted_compound_sum += article["sentiment"]["compound"] * weight
        
        avg_compound = weighted_compound_sum / total_weight if total_weight > 0 else 0
        simple_avg_compound = sum(article["sentiment"]["compound"] for article in news_articles) / len(news_articles)
        
        positive_articles = sum(1 for article in news_articles if article["sentiment"]["compound"] > 0.05)
        negative_articles = sum(1 for article in news_articles if article["sentiment"]["compound"] < -0.05)
        neutral_articles = len(news_articles) - positive_articles - negative_articles
    else:
        avg_compound = 0
        simple_avg_compound = 0
        positive_articles = 0
        negative_articles = 0
        neutral_articles = 0
    
    # 날짜별 평균 감성 계산
    avg_sentiment_by_date = {
        date: sum(scores) / len(scores) for date, scores in sentiment_by_date.items()
    }
    
    # 트렌드 분석을 위한 날짜 정렬
    sorted_dates = sorted(avg_sentiment_by_date.keys())
    sentiment_trend = [avg_sentiment_by_date[date] for date in sorted_dates]
    
    # 감성이 개선되고 있는지 또는 악화되고 있는지 결정
    sentiment_direction = "stable"
    if len(sentiment_trend) > 1:
        # 트렌드 결정을 위한 간단한 선형 회귀
        x = list(range(len(sentiment_trend)))
        y = sentiment_trend
        
        # 최소 제곱법을 사용한 기울기 계산
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x_i * y_i for x_i, y_i in zip(x, y))
        sum_xx = sum(x_i * x_i for x_i in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x) if (n * sum_xx - sum_x * sum_x) != 0 else 0
        
        if slope > 0.01:
            sentiment_direction = "improving"
        elif slope < -0.01:
            sentiment_direction = "deteriorating"
    
    sentiment_summary = {
        "avg_compound": avg_compound,
        "simple_avg_compound": simple_avg_compound,
        "positive_articles": positive_articles,
        "negative_articles": negative_articles,
        "neutral_articles": neutral_articles,
        "total_articles": len(news_articles),
        "sentiment_distribution": {
            "positive": positive_articles / len(news_articles) if news_articles else 0,
            "negative": negative_articles / len(news_articles) if news_articles else 0,
            "neutral": neutral_articles / len(news_articles) if news_articles else 0
        },
        "sentiment_by_date": avg_sentiment_by_date,
        "sentiment_trend": {
            "dates": sorted_dates,
            "values": sentiment_trend,
            "direction": sentiment_direction
        }
    }
    
    return news_articles, sentiment_summary

def generate_sentiment_trend_chart_data(sentiment_summary, ticker):
    """
    시간 경과에 따른 감성 트렌드 데이터를 반환합니다.
    차트를 생성하지만 이미지로 저장하지 않습니다.
    
    Args:
        sentiment_summary (dict): 감성 분석 요약
        ticker (str): 주식 티커 기호
        
    Returns:
        dict: 차트 데이터 (날짜 및 감성 값)
    """
    if not sentiment_summary["sentiment_trend"]["dates"]:
        return None
    
    # 차트 데이터 반환
    return {
        "dates": sentiment_summary["sentiment_trend"]["dates"],
        "values": sentiment_summary["sentiment_trend"]["values"]
    }

def analyze_stock_impact(sentiment_summary, finance_data):
    """
    뉴스 감성이 주가에 미치는 잠재적 영향을 분석합니다.
    Alpha Vantage 데이터(회사 개요)에 맞게 수정됨. 실시간 가격 정보 제외.

    Args:
        sentiment_summary (dict): 감성 분석 요약
        finance_data (dict): Alpha Vantage 회사 개요 데이터
        
    Returns:
        dict: 잠재적 주가 영향 분석 (가격 정보 제외)
    """

    # 감성에 기반한 잠재적 영향 결정 (기존 로직 유지)
    impact = "neutral"
    confidence = 0.5
    sentiment_score = sentiment_summary["avg_compound"]

    if sentiment_score > 0.25:
        impact = "매우 긍정적"
        confidence = min(0.5 + sentiment_score, 0.95)
    elif sentiment_score > 0.05:
        impact = "긍정적"
        confidence = 0.5 + (sentiment_score - 0.05) * 2
    elif sentiment_score < -0.25:
        impact = "매우 부정적"
        confidence = min(0.5 - sentiment_score, 0.95)
    elif sentiment_score < -0.05:
        impact = "부정적"
        confidence = 0.5 + (abs(sentiment_score) - 0.05) * 2
    else:
        impact = "중립적"
        confidence = 0.5

    if "knowledge_graph" in finance_data and "financial_data" in finance_data["knowledge_graph"]:
        financial_data = finance_data["knowledge_graph"]["financial_data"]
        stock_price = financial_data.get("price", None)
        price_change = financial_data.get("price_change", None)
        price_change_percentage = financial_data.get("price_change_percentage", None)
        
    # 감성 분포 분석
    sentiment_ratio = None
    if sentiment_summary["positive_articles"] + sentiment_summary["negative_articles"] > 0:
        sentiment_ratio = sentiment_summary["positive_articles"] / (sentiment_summary["positive_articles"] + sentiment_summary["negative_articles"])
    
    # 감성 트렌드 고려
    if sentiment_summary["sentiment_trend"]["direction"] == "improving":
        trend_impact = " 감성 추세가 시간이 지남에 따라 개선되고 있어, 앞으로 더 긍정적인 전망을 시사할 수 있습니다."
    elif sentiment_summary["sentiment_trend"]["direction"] == "deteriorating":
        trend_impact = " 감성 추세가 시간이 지남에 따라 악화되고 있어, 앞으로 도전과제가 있을 수 있음을 시사합니다."
    else:
        trend_impact = " 감성 추세는 시간이 지남에 따라 안정적으로 유지되고 있습니다."
    
    impact_analysis = {
        "sentiment_impact": impact,
        "confidence": confidence,
        "positive_to_negative_ratio": sentiment_ratio,
        "sentiment_trend_direction": sentiment_summary["sentiment_trend"]["direction"],
        "analysis_summary": f"Based on the analysis of {sentiment_summary['total_articles']} news articles, the sentiment is predominantly {impact} with a confidence of {confidence:.2f}.{trend_impact}" # 수정됨
    }

    return impact_analysis

def summarize_news_by_sentiment(news_articles, ticker, gemini_key=None):
    """
    감성 카테고리별로 뉴스를 요약합니다.
    Google의 Gemini-2.5-flash 모델을 사용합니다.
    
    Args:
        news_articles (list): 감성이 포함된 뉴스 기사 목록
        ticker (str): 주식 티커 기호
        
    Returns:
        dict: 감성 카테고리별 요약
    """
    try:
        # Gemini API 키가 설정되어 있지 않으면 간단한 요약 반환
        if not GEMINI_API_KEY:
            print("Gemini API 키가 설정되어 있지 않습니다. 기본 요약이 제공됩니다.")
            return {
                "positive": "Gemini API 키가 제공되지 않았습니다. 요약을 생성할 수 없습니다.",
                "neutral": "Gemini API 키가 제공되지 않았습니다. 요약을 생성할 수 없습니다.",
                "negative": "Gemini API 키가 제공되지 않았습니다. 요약을 생성할 수 없습니다."
            }
        
        # 감성 카테고리별로 뉴스 분류
        positive_articles = [a for a in news_articles if a["sentiment"]["compound"] > 0.05]
        neutral_articles = [a for a in news_articles if -0.05 <= a["sentiment"]["compound"] <= 0.05]
        negative_articles = [a for a in news_articles if a["sentiment"]["compound"] < -0.05]
        
        # Gemini 모델 설정
        model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
        
        summaries = {}
        
        # 긍정적 뉴스 요약
        if positive_articles:
            positive_texts = "\n".join([f"Title: {a.get('title', '')}\nSummary: {a.get('summary', '')}" # snippet -> summary 수정됨
                                       for a in positive_articles[:10]])
            
            prompt = f"""다음 {ticker} 주식에 대한 긍정적인 뉴스 기사들을 요약해주세요:
            
            {positive_texts}
            
            주요 긍정적인 포인트, 트렌드 및 주가에 미칠 수 있는 잠재적 영향에 대해 간결하게 요약해주세요. 
            요약은 약 150단어 정도로 작성하고, 이 기사들에서 언급된 가장 중요한 긍정적인 측면에 초점을 맞춰주세요.
            한국어로 응답해주세요.
            """
            
            response = model.generate_content(prompt)
            summaries["positive"] = response.text
        else:
            summaries["positive"] = f"분석 기간 동안 {ticker}에 대한 긍정적인 뉴스 기사가 발견되지 않았습니다."
        
        # 중립적 뉴스 요약
        if neutral_articles:
            neutral_texts = "\n".join([f"Title: {a.get('title', '')}\nSummary: {a.get('summary', '')}" # snippet -> summary 수정됨
                                      for a in neutral_articles[:10]])
            
            prompt = f"""다음 {ticker} 주식에 대한 중립적인 뉴스 기사들을 요약해주세요:
            
            {neutral_texts}
            
            주요 중립적인 포인트, 사실, 그리고 언급된 맥락 정보에 대해 간결하게 요약해주세요.
            요약은 약 150단어 정도로 작성하고, 투자자들에게 관련될 수 있는 객관적인 정보에 초점을 맞춰주세요.
            한국어로 응답해주세요.
            """
            
            response = model.generate_content(prompt)
            summaries["neutral"] = response.text
        else:
            summaries["neutral"] = f"분석 기간 동안 {ticker}에 대한 중립적인 뉴스 기사가 발견되지 않았습니다."
        
        # 부정적 뉴스 요약
        if negative_articles:
            negative_texts = "\n".join([f"Title: {a.get('title', '')}\nSummary: {a.get('summary', '')}" # snippet -> summary 수정됨
                                       for a in negative_articles[:10]])
            
            prompt = f"""다음 {ticker} 주식에 대한 부정적인 뉴스 기사들을 요약해주세요:
            
            {negative_texts}
            
            주요 우려사항, 도전과제, 그리고 주가에 미칠 수 있는 잠재적 부정적 영향에 대해 간결하게 요약해주세요.
            요약은 약 150단어 정도로 작성하고, 이 기사들에서 언급된 가장 중요한 위험이나 문제에 초점을 맞춰주세요.
            한국어로 응답해주세요.
            """
            
            response = model.generate_content(prompt)
            summaries["negative"] = response.text
        else:
            summaries["negative"] = f"분석 기간 동안 {ticker}에 대한 부정적인 뉴스 기사가 발견되지 않았습니다."
        
        return summaries
    
    except Exception as e:
        print(f"Gemini API를 사용한 요약 생성 중 오류 발생: {e}")
        return {
            "positive": f"요약 생성 중 오류 발생: {e}",
            "neutral": f"요약 생성 중 오류 발생: {e}",
            "negative": f"요약 생성 중 오류 발생: {e}"
        }

def generate_financial_report(ticker, news_articles, sentiment_summary, impact_analysis, finance_data, start_date, end_date, gemini_key=None):
    """
    주식에 대한 종합적인 뉴스 분석 보고서를 생성합니다.
    
    Args:
        ticker (str): 주식 티커 기호
        news_articles (list): 감성이 포함된 뉴스 기사 목록
        sentiment_summary (dict): 감성 분석 요약
        impact_analysis (dict): 잠재적 주가 영향 분석
        finance_data (dict): 주식에 대한 재무 데이터 (사용되지 않음)
        start_date (str): 분석 기간의 시작 날짜
        end_date (str): 분석 기간의 종료 날짜
        
    Returns:
        str: 마크다운 형식의 재무 보고서
    """
    # 회사 이름 추출
    company_name = finance_data.get("Name", ticker) # 수정됨
    if "knowledge_graph" in finance_data:
        company_name = finance_data["knowledge_graph"].get("title", ticker)
    
    # 감성 트렌드 데이터 가져오기 (차트 이미지 저장하지 않음)
    chart_data = generate_sentiment_trend_chart_data(sentiment_summary, ticker)
    
    # 감성별 뉴스 요약 생성
    print("뉴스 감성별 요약 생성 중...")
    news_summaries = summarize_news_by_sentiment(news_articles, ticker)
    
    # 긍정/부정 비율 계산
    pos_neg_ratio = impact_analysis.get('positive_to_negative_ratio')
    ratio_display = f"{pos_neg_ratio:.2f}" if pos_neg_ratio is not None else "N/A"
    
    # 보고서 생성
    report = f"""# News Analysis Report: {company_name} ({ticker})

## 분석 기간
- **시작일:** {start_date}
- **종료일:** {end_date}

## 뉴스 감성 분석
- **분석된 총 기사:** {sentiment_summary['total_articles']}
- **긍정적 기사:** {sentiment_summary['positive_articles']} ({sentiment_summary['sentiment_distribution']['positive']:.2%})
- **중립적 기사:** {sentiment_summary['neutral_articles']} ({sentiment_summary['sentiment_distribution']['neutral']:.2%})
- **부정적 기사:** {sentiment_summary['negative_articles']} ({sentiment_summary['sentiment_distribution']['negative']:.2%})
- **평균 감성 점수:** {sentiment_summary['simple_avg_compound']:.4f}
- **가중 감성 점수 (최근 뉴스 우선):** {sentiment_summary['avg_compound']:.4f}
- **감성 추세:** {sentiment_summary['sentiment_trend']['direction']}

"""
    
    # 감성 트렌드 차트 섹션은 제거됨 (이미지 파일 저장 없음)
    
    # 잠재적 시장 영향 섹션 추가
    report += f"""## 잠재적 시장 영향 (감성 기반)
- **예상 영향:** {impact_analysis['sentiment_impact']}
- **신뢰도:** {impact_analysis['confidence']:.2%}
- **긍정 대 부정 비율:** {ratio_display}
- **감성 추세 방향:** {impact_analysis['sentiment_trend_direction']}

## 분석 요약
{impact_analysis['analysis_summary']}

## 감성별 뉴스 요약

### 긍정적 뉴스 요약
{news_summaries.get('positive', '긍정적 뉴스 요약이 없습니다.')}

### 중립적 뉴스 요약
{news_summaries.get('neutral', '중립적 뉴스 요약이 없습니다.')}

### 부정적 뉴스 요약
{news_summaries.get('negative', '부정적 뉴스 요약이 없습니다.')}

## 최근 뉴스 헤드라인
"""

    # 뉴스 헤드라인 섹션 수정
    for i, article in enumerate(news_articles[:10]):
        sentiment = "긍정적" if article["sentiment"]["compound"] > 0.05 else "부정적" if article["sentiment"]["compound"] < -0.05 else "중립적"
        date = article.get("date", "날짜 미상")
        report += f"{i+1}. **{article['title']}** - {date} - *{sentiment}* (점수: {article['sentiment']['compound']:.2f})\n   출처: {article.get('source', '미상')}\n   [더 읽기]({article.get('url', '#')})\n\n"

    # 면책 조항 추가
    report += """

## Disclaimer
This report was automatically generated based on news sentiment analysis and should not be considered as financial advice. Any predictions or other information about possible outcomes are hypothetical in nature, do not guarantee accuracy or completeness, do not reflect actual investment results, and are not guarantees of future results. Always conduct your own research and consult with a financial advisor before making investment decisions.
"""
    
    return report