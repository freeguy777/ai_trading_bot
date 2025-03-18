import sys
import pandas as pd
import os
from datetime import datetime
from dataclasses import asdict

def load_data():
    """데이터베이스에서 모든 리포트를 로드하고 처리"""
    # --- 경로 설정 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # sys.path에 project_root를 최우선 순위로 추가 (다시 추가)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    from memory_SQL.main import  load_reports
    
    # 모든 리포트 로드
    reports = load_reports()
    
    # 데이터프레임으로 변환
    df = pd.DataFrame([asdict(report) for report in reports])
    
    # 데이터 전처리
    if not df.empty:
        # 날짜 타입 변환
        df['run_date'] = pd.to_datetime(df['run_date'], format='ISO8601')
        # 리스트 데이터 처리
        list_columns = ['catalysts_short_term', 'catalysts_mid_term', 
                       'catalysts_long_term', 'key_monitoring_metrics']
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(parse_list_field)
        
        # 최신순 정렬
        df = df.sort_values('run_date', ascending=False)
    
    return df

def parse_list_field(val):
    """문자열로 저장된 리스트를 파싱"""
    if isinstance(val, list):
        return val
    if not isinstance(val, str):
        return []
    try:
        if val.strip().startswith('['):
            return eval(val)
        # 단일 문자열인 경우
        return [val]
    except:
        return []

def process_report(report):
    """리포트 데이터를 시각화 용도로 가공"""
    processed = report.copy()
    
    # 날짜 포맷팅
    if isinstance(processed['run_date'], pd.Timestamp):
        processed['display_date'] = processed['run_date'].strftime("%Y-%m-%dT%H:%M:%S") 
    elif isinstance(processed['run_date'], str):
        # ISO 형식 문자열을 datetime으로 파싱한 후 포맷팅
        from datetime import datetime
        dt = datetime.fromisoformat(processed['run_date'])
        processed['display_date'] = dt.strftime("%Y-%m-%dT%H:%M:%S")
    
    # 나머지 코드는 동일
    # 포지션 비중을 0-100 스케일로 정규화
    if 'allocation_suggestion' in processed:
        processed['allocation_percentage'] = min(max(processed['allocation_suggestion'], 0), 100)
    
    # 금액 포맷팅
    money_fields = ['current_price', 'average_purchase_price']
    for field in money_fields:
        if field in processed:
            processed[f'{field}_formatted'] = format_currency_dollor(processed[field])
    
    # 퍼센트 포맷팅
    if 'current_return_pct' in processed:
        processed['current_return_formatted'] = format_percentage(processed['current_return_pct'])
    
    return processed

def format_currency(value):
    """금액을 한글 단위로 포맷팅"""
    if value >= 100000000:  # 1억 이상
        return f"{value/100000000:.1f}억원"
    elif value >= 10000:  # 1만 이상
        return f"{value/10000:.1f}만원"
    else:
        return f"{value:,}원"

def format_currency_dollor(value: float | int) -> str:
    """달러 금액을 사람 읽기 좋은 단축 표기로 변환

    - 1 billion 이상  → $ 1.2 B
    - 1 million 이상  → $ 345.6 M
    - 1 thousand 이상 → $ 7.8 K
    - 그 미만          → $ 1,234
    """
    abs_val = abs(value)        # 음수 지원
    sign    = "-" if value < 0 else ""

    if abs_val >= 1_000_000_000:        # 10억 달러 이상
        formatted = f"{abs_val/1_000_000_000:.2f} B"
    elif abs_val >= 1_000_000:          # 100만 달러 이상
        formatted = f"{abs_val/1_000_000:.2f} M"
    elif abs_val >= 1_000:              # 1천 달러 이상
        formatted = f"{abs_val/1_000:.2f} K"
    else:                               # 1천 달러 미만
        formatted = f"{abs_val:,.2f}"

    return f"{sign}${formatted}"

def format_percentage(value):
    """퍼센트 포맷팅"""
    return f"{value:.1f}%"