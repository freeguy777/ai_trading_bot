import streamlit as st
import pandas as pd
import os  # os 모듈 추가
from pathlib import Path  # pathlib 추가
import locale # locale 모듈 임포트
from data_manager import load_data, process_report
from ui_components import render_header, render_sidebar, render_metrics_cards
from visualizations import (
    render_dashboard_charts, 
    render_department_analysis, 
    render_detailed_analysis, 
    render_full_report
)
from settings import APP_TITLE, APP_ICON

def init_korean_locale():
    """
    ko_KR 로캘을 시도하고 실패하면 시스템 기본값으로 폴백.
    컨테이너에서 쓸데없는 예외를 계속 발생시키지 않는다.
    """
    _CANDIDATES = (
        "ko_KR.UTF-8",   # 일반 Linux
        "ko_KR.utf8",    # 일부 배포판
        "Korean_Korea.949",  # Windows
    )

    for loc in _CANDIDATES:
        try:
            locale.setlocale(locale.LC_ALL, loc)
            # 서브프로세스를 위해 환경변수도 같이 세팅
            os.environ["LANG"] = loc
            os.environ["LC_ALL"] = loc
            return   # 첫 성공 시 바로 종료
        except locale.Error:
            continue

    # 전부 실패하면 영어 로캘로 통일하고 한 번만 경고
    locale.setlocale(locale.LC_ALL, "C.UTF-8")
    st.warning(
        "컨테이너에 한국어 로캘이 없네요. 날짜·숫자 형식이 영어 기본값으로 표시됩니다."
    )

def main():
    # 앱 설정
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 로캘 설정 (한국어, UTF-8)
    init_korean_locale()
            
    # 커스텀 CSS 적용
    # 스크립트 파일의 위치를 기준으로 CSS 파일 경로 계산
    script_dir = Path(__file__).parent
    css_file_path = script_dir / "static/css/style.css"
    
    if css_file_path.is_file():
        with open(css_file_path, encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS 파일을 찾을 수 없습니다: {css_file_path}")
    
    # 헤더 렌더링
    render_header()
    
    # 데이터 로드
    reports_df = load_data()
    
    # 사이드바 렌더링 및 선택된 날짜 받기
    selected_date = render_sidebar(reports_df)
    
    # 선택된 리포트가 있을 경우
    if selected_date is not None and not reports_df.empty:
        # 선택된 리포트 데이터 가져오기
        selected_report = reports_df[reports_df['run_date'] == selected_date].iloc[0]
        processed_report = process_report(selected_report)
        
        # 탭 메뉴 생성
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 대시보드", 
            "🏢 부서별 분석", 
            "🔍 상세 분석", 
            "📝 종합 보고서"
        ])
        
        # 탭 1: 대시보드
        with tab1:
            render_dashboard_tab(processed_report)
        
        # 탭 2: 부서별 분석
        with tab2:
            render_department_tab(processed_report)
        
        # 탭 3: 상세 분석
        with tab3:
            render_detailed_tab(processed_report)
        
        # 탭 4: 종합 보고서
        with tab4:
            render_report_tab(processed_report)
    else:
        st.info("왼쪽 사이드바에서 분석 리포트를 선택해 주세요.")

def render_dashboard_tab(report):
    """대시보드 탭 렌더링"""
    ticker = report.get('ticker', '–')
    st.header(f"📊 {ticker} 투자 분석 대시보드")

    # 핵심 지표 카드 렌더링
    metrics_cols = st.columns(3)
    render_metrics_cards(report, metrics_cols)

    # 대시보드 차트 렌더링
    render_dashboard_charts(report)

def render_department_tab(report):
    """부서별 분석 탭 렌더링"""
    st.header("부서별 분석 리포트")
    render_department_analysis(report)

def render_detailed_tab(report):
    """상세 분석 탭 렌더링"""
    st.header("상세 투자 분석")
    render_detailed_analysis(report)

def render_report_tab(report):
    """종합 보고서 탭 렌더링"""
    st.header("종합 분석 보고서")
    render_full_report(report)

if __name__ == "__main__":
    main()