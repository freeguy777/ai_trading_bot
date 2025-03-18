import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from settings import COLORS
import os, sys
from dotenv import load_dotenv

# --- 경로 설정 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

from kis_trader.main import get_account_profit_rate, get_average_price, get_current_price

load_dotenv("config/.env")
ticker = os.getenv("TICKER")

def render_dashboard_charts(report):
    """대시보드 페이지의 차트 렌더링"""
    col1, col2 = st.columns(2)
    
    with col1:
        # 보유 정보 카드
        st.subheader("보유 정보")
        render_holdings_card(report)
    
    with col2:
        # 핵심투자논거 카드
        st.subheader("핵심투자논거")
        thesis = report.get('investment_thesis', '')
        st.markdown(f"<div class='thesis-card'>{thesis}</div>", unsafe_allow_html=True)

def render_holdings_card(report):
    """보유 정보 시각화"""
    current_price = get_current_price(ticker, exchange_name="나스닥")
    current_price = f"{current_price:,.2f}"
    return_pct = get_account_profit_rate()
    avg_price = get_average_price(ticker, exchange_name='나스닥')

    if avg_price == -1:                    # KIS 래퍼가 “없음”일 때 -1 반환
        avg_price_disp = "보유수량 없음"
    else:
        avg_price_disp = f"{avg_price:,.2f}"   # 두 자리 반올림·콤마 포함

    # 수익률에 따른 색상 결정
    #return_value = report.get('current_return_pct', 0)
    return_color = COLORS["positive"] if return_pct >= 0 else COLORS["negative"]
    return_prefix = "+" if return_pct > 0 else ""
    
    holding_html = f"""
    <div class="holdings-container">
        <div class="holding-item">
            <div class="holding-label">현재가</div>
            <div class="holding-value">{current_price}</div>
        </div>
        <div class="holding-item">
            <div class="holding-label">수익률</div>
            <div class="holding-value" style="color:{return_color}">
                {return_prefix}{return_pct}
            </div>
        </div>
        <div class="holding-item">
            <div class="holding-label">평균단가</div>
            <div class="holding-value">{avg_price_disp}</div>
        </div>
    </div>
    """
    st.markdown(holding_html, unsafe_allow_html=True)

def render_department_analysis(report):
    """부서별 분석 시각화"""
    # 4개 부서 분석 리포트 표시
    col1, col2 = st.columns(2)
    
    # 차트 분석
    with col1:
        chart_content = report.get('chart_report', '')
        st.markdown("### 📈 차트 분석")
        st.markdown(f"<div class='department-card'>{chart_content}</div>", 
                   unsafe_allow_html=True)
    
    # 리서치 분석
    with col2:
        research_content = report.get('research_report', '')
        st.markdown("### 🔍 리서치 분석")
        st.markdown(f"<div class='department-card'>{research_content}</div>", 
                   unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    # 재무 분석
    with col3:
        financial_content = report.get('financial_report', '')
        st.markdown("### 💰 재무 분석")
        st.markdown(f"<div class='department-card'>{financial_content}</div>", 
                   unsafe_allow_html=True)
    
    # 거시경제 분석
    with col4:
        macro_content = report.get('macro_report', '')
        st.markdown("### 🌐 거시경제 분석")
        st.markdown(f"<div class='department-card'>{macro_content}</div>", 
                   unsafe_allow_html=True)

def render_detailed_analysis(report):
    """상세 분석 시각화"""
    # 가치평가 섹션
    st.subheader("가치평가 분석")
    col1, col2 = st.columns(2)
    
    with col1:
        valuation = report.get('valuation_assessment', '')
        st.markdown("#### 적정가치평가")
        st.markdown(f"<div class='analysis-card'>{valuation}</div>", unsafe_allow_html=True)
    
    with col2:
        # 상승잠재력 vs 하방리스크 비교
        upside = report.get('upside_potential', '')
        downside = report.get('downside_risks', '')
        
        st.markdown("#### 상승잠재력 vs 하방리스크")
        st.markdown(
            f"<div class='comparison-container'>"
            f"<div class='comparison-item positive'>"
            f"<div class='comparison-header'>상승잠재력 ▲</div>"
            f"<div class='comparison-content'>{upside}</div>"
            f"</div>"
            f"<div class='comparison-item negative'>"
            f"<div class='comparison-header'>하방리스크 ▼</div>"
            f"<div class='comparison-content'>{downside}</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    # 촉매 분석 섹션
    st.subheader("촉매 분석")
    render_catalysts_timeline(report)
    
    # 전략 섹션
    st.subheader("투자 전략")
    col3, col4 = st.columns(2)
    
    with col3:
        horizon = report.get('investment_horizon', '')
        st.markdown("#### 투자기간")
        st.markdown(f"<div class='strategy-card'>{horizon}</div>", unsafe_allow_html=True)
    
    with col4:
        exit_strategy = report.get('exit_strategy', '')
        st.markdown("#### 투자철회조건")
        st.markdown(f"<div class='strategy-card warning'>{exit_strategy}</div>", 
                   unsafe_allow_html=True)

def render_catalysts_timeline(report):
    """촉매 타임라인 시각화"""
    # 3개 기간의 촉매 표시
    col1, col2, col3 = st.columns(3)
    
    # 단기 촉매
    with col1:
        short_term = report.get('catalysts_short_term', [])
        if not isinstance(short_term, list):
            short_term = [short_term]
        
        st.markdown("#### 단기 촉매")
        for item in short_term:
            st.markdown(f"<div class='catalyst-item short-term'>● {item}</div>", 
                       unsafe_allow_html=True)
    
    # 중기 촉매
    with col2:
        mid_term = report.get('catalysts_mid_term', [])
        if not isinstance(mid_term, list):
            mid_term = [mid_term]
        
        st.markdown("#### 중기 촉매")
        for item in mid_term:
            st.markdown(f"<div class='catalyst-item mid-term'>● {item}</div>", 
                       unsafe_allow_html=True)
    
    # 장기 촉매
    with col3:
        long_term = report.get('catalysts_long_term', [])
        if not isinstance(long_term, list):
            long_term = [long_term]
        
        st.markdown("#### 장기 촉매")
        for item in long_term:
            st.markdown(f"<div class='catalyst-item long-term'>● {item}</div>", 
                       unsafe_allow_html=True)

def render_full_report(report):
    """종합 보고서 시각화"""
    # 전체 분석 요약
    detail_report = report.get('detail_report', '')
    
    # 내용을 단락으로 분리
    paragraphs = detail_report.split('\n\n')
    
    # 전체 분석 요약 표시
    st.markdown(
        f"<div class='full-report-container'>"
        f"<div class='full-report-header'>종합 투자 분석 리포트</div>"
        f"<div class='full-report-date'>{report.get('display_date', '')}</div>"
        f"<div class='full-report-content'>"
    , unsafe_allow_html=True)
    
    for p in paragraphs:
        if p.strip():
            st.markdown(f"<p>{p}</p>", unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # 모니터링 지표 표시
    st.subheader("핵심 모니터링 지표")
    metrics = report.get('key_monitoring_metrics', [])
    if not isinstance(metrics, list):
        metrics = [metrics]
    
    metrics_html = "<div class='monitoring-metrics'><ul>"
    for metric in metrics:
        metrics_html += f"<li>{metric}</li>"
    metrics_html += "</ul></div>"
    
    st.markdown(metrics_html, unsafe_allow_html=True)