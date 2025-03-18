import streamlit as st
import pandas as pd
from settings import COLORS, DECISION_STYLES, CREDIBILITY_LEVELS

def render_header():
    """앱 헤더 렌더링"""
    st.markdown(
        "<div class='header-container'>"
        "<h1>📊 투자 분석 대시보드</h1>"
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("<hr>", unsafe_allow_html=True)

def render_sidebar(df):
    """사이드바 렌더링 및 리포트 선택"""
    with st.sidebar:
        st.title("분석 리포트 목록")
        
        # 데이터가 없는 경우 처리
        if df.empty:
            st.warning("저장된 리포트가 없습니다.")
            return None
        
        # 검색 기능
        search_term = st.text_input("리포트 검색", "", placeholder="검색어 입력...")
        
        # 필터링 기능
        decision_options = df['decision'].unique().tolist()
        decision_filter = st.multiselect(
            "투자결정 필터",
            options=decision_options,
            default=[]
        )
        
        # 검색 및 필터 적용
        filtered_df = df.copy()
        if search_term:
            # 여러 컬럼에서 검색
            search_cols = ['detail_report', 'investment_thesis', 'decision']
            mask = False
            for col in search_cols:
                if col in filtered_df.columns:
                    mask = mask | filtered_df[col].str.contains(search_term, case=False, na=False)
            filtered_df = filtered_df[mask]
        
        if decision_filter:
            filtered_df = filtered_df[filtered_df['decision'].isin(decision_filter)]
        
        # 날짜별 리포트 목록 표시
        if len(filtered_df) == 0:
            st.warning("검색 결과가 없습니다.")
            return None
        
        date_options = filtered_df['run_date'].unique()
        # 날짜 선택 컴포넌트 (스타일된 라디오 버튼)
        selected_date_index = st.radio(
            "날짜 선택",
            options=range(len(date_options)),
            format_func=lambda i: format_date_with_decision(
                date_options[i], 
                filtered_df[filtered_df['run_date'] == date_options[i]].iloc[0]['decision']
            )
        )
        
        return date_options[selected_date_index]
def format_date_with_decision(date, decision):
    """날짜와 투자결정을 함께 포맷팅 (HTML 없이)"""
    date_str = date.strftime("%Y-%m-%dT%H:%M:%S")
    
    decision_icons = {
        "매수": "🟢 ",
        "매도": "🔴 ",
        "보유": "🟠 ",
        # 다른 결정 추가
    }
    
    icon = decision_icons.get(decision, "• ")
    return f"{date_str} [{icon}{decision}]"

def render_metrics_cards(report, columns):
    """핵심 지표 카드 렌더링"""
    # 투자결정 카드
    with columns[0]:
        decision = report.get('decision', '')
        decision_style = DECISION_STYLES.get(decision, {"color": "#6B7280", "icon": "•"})
        
        st.markdown(
            f"<div class='metric-card' style='border-left: 4px solid {decision_style['color']};'>"
            f"<div class='metric-title'>투자결정</div>"
            f"<div class='metric-value' style='color:{decision_style['color']};'>"
            f"{decision_style['icon']} {decision}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    # 신뢰도 카드
    with columns[1]:
        credibility = report.get('credibility', '')
        cred_style = CREDIBILITY_LEVELS.get(credibility, {"stars": 0, "color": "#6B7280"})
        
        stars = "★" * cred_style["stars"] + "☆" * (5 - cred_style["stars"])
        
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>신뢰도</div>"
            f"<div class='metric-value' style='color:{cred_style['color']};'>"
            f"{stars} <span class='credibility-text'>{credibility}</span></div>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    # 포지션비중 카드
    with columns[2]:
        allocation = report.get('allocation_suggestion', 0)
        
        # 포지션 비중에 따른 색상 결정
        if allocation > 70:
            color = COLORS["positive"]
        elif allocation > 30:
            color = COLORS["neutral"]
        else:
            color = COLORS["negative"]
        
        progress_bar = create_progress_bar(allocation, color)
        
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>포지션비중</div>"
            f"<div class='metric-value'>{progress_bar} {allocation}%</div>"
            f"</div>",
            unsafe_allow_html=True
        )

def create_progress_bar(value, color):
    """커스텀 진행 막대 생성"""
    filled = int(value / 10)
    empty = 10 - filled
    
    return (
        f"<div class='progress-bar'>"
        f"<span class='filled' style='width:{value}%; background-color:{color};'></span>"
        f"</div>"
    )

def render_info_card(title, content, icon="ℹ️", color=None):
    """정보 카드 렌더링"""
    color_style = f"style='color:{color};'" if color else ""
    
    st.markdown(
        f"<div class='info-card'>"
        f"<div class='info-card-header' {color_style}>"
        f"<span class='info-card-icon'>{icon}</span> {title}"
        f"</div>"
        f"<div class='info-card-content'>{content}</div>"
        f"</div>",
        unsafe_allow_html=True
    )