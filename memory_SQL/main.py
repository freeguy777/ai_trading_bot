"""
db_manager.py
간단·무미건조 SQLite 래퍼
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import json

###############################################################################
# 기본 설정
###############################################################################

_DB_PATH = Path(__file__).with_name("trading_reports.sqlite3")
_TABLE_NAME = "daily_reports"


@contextmanager
def _get_conn() -> Iterable[sqlite3.Connection]:
    """SQLite 연결을 컨텍스트 매니저로 얻는다."""
    with sqlite3.connect(_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row  # dict-style row
        yield conn


###############################################################################
# 데이터 스키마
###############################################################################

@dataclass(slots=True, frozen=True)
class TradingReport:
    run_date: str = field(default_factory=lambda: datetime.now().isoformat(timespec='seconds'))
    ticker: str = ""
    # --- 각 부서 취합 결과 ---
    chart_report: str = ""                            # 차트 분석 보고서
    research_report: str = ""                         # 리서치 분석 보고서
    financial_report: str = ""                        # 재무 분석 보고서
    macro_report: str = ""                            # 거시경제 분석 보고서
    # --- 요약 필드 ---
    decision: str = ""                                # 투자결정 (alias="투자결정")
    credibility: str = ""                             # 신뢰도 (alias="신뢰도")
    allocation_suggestion: float = 0                  # 포지션비중 / 투자비중제안 (alias="포지션비중")
    # --- 10가지 상세 근거 (모두 최소 3문장 요구) ---
    investment_thesis: str = ""                       # 핵심투자논거 (alias="핵심투자논거")
    valuation_assessment: str = ""                    # 적정가치평가 (alias="적정가치평가")
    catalysts_short_term: List[str] = field(default_factory=list)              
    catalysts_mid_term: List[str] = field(default_factory=list)                
    catalysts_long_term: List[str] = field(default_factory=list)               
    upside_potential: str = ""                        # 상승잠재력 (alias="상승잠재력")
    downside_risks: str = ""                          # 하방리스크 (alias="하방리스크")
    contrarian_view: str = ""                         # 차별화관점 (alias="차별화관점")
    investment_horizon: str = ""                      # 투자기간 (alias="투자기간")
    key_monitoring_metrics: List[str] = field(default_factory=list)            
    exit_strategy: str = ""                           # 투자철회조건 (alias="투자철회조건")
    # --- 전체 서술형 보고서 ---
    detail_report: str = ""                           # 전체분석요약 / 상세보고서 (alias="전체분석요약")
    # --- 현재 보유 정보 ---
    current_price: int = 0                            # 현재 주식 보유 금액
    current_return_pct: int = 0                       # 현재 수익률 (%)
    average_purchase_price: int = 0                   # 평단가
###############################################################################
# 테이블 초기화 (필요 시 자동 실행)
###############################################################################

def _init_db() -> None:
    """테이블이 없으면 만든다."""
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {_TABLE_NAME} (
        run_date TEXT,
        ticker TEXT,  
        chart_report TEXT,
        research_report TEXT,
        financial_report TEXT,
        macro_report TEXT,
        decision TEXT,
        credibility TEXT,
        allocation_suggestion INTEGER,
        investment_thesis TEXT,
        valuation_assessment TEXT,
        catalysts_short_term TEXT,       
        catalysts_mid_term TEXT,         
        catalysts_long_term TEXT,        
        upside_potential TEXT,
        downside_risks TEXT,
        contrarian_view TEXT,
        investment_horizon TEXT,
        key_monitoring_metrics TEXT,     
        exit_strategy TEXT,
        detail_report TEXT,
        current_price INTEGER,
        current_return_pct INTEGER,      
        average_purchase_price INTEGER  
    );
    """
    with _get_conn() as conn:
        conn.execute(create_sql)          # 1) 테이블이 아예 없으면 생성
        # 2) 컬럼 유무 점검 → 없으면 ALTER
        need_cols = {"ticker": "TEXT"}    # 필요한 컬럼: 자료형 매핑
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({_TABLE_NAME});")}
        for col, typ in need_cols.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE {_TABLE_NAME} ADD COLUMN {col} {typ};")
        conn.commit()


###############################################################################
# 외부에 공개되는 함수
###############################################################################

def save_report(report: TradingReport) -> None:
    """TradingReport 인스턴스를 DB에 저장. 리스트는 JSON 문자열로 변환."""
    _init_db()
    report_dict = asdict(report)
    cols = ", ".join(report_dict.keys())
    placeholders = ", ".join("?" for _ in report_dict)
    sql = f"INSERT INTO {_TABLE_NAME} ({cols}) VALUES ({placeholders})"

    # 리스트 필드를 JSON 문자열로 변환 (다시 추가)
    values_to_insert = []
    list_fields = ['catalysts_short_term', 'catalysts_mid_term', 
                   'catalysts_long_term', 'key_monitoring_metrics']
    for key, value in report_dict.items():
        if key in list_fields and isinstance(value, list):
            values_to_insert.append(json.dumps(value, ensure_ascii=False))
        else:
            values_to_insert.append(value)

    with _get_conn() as conn:
        conn.execute(sql, tuple(values_to_insert))
        conn.commit()


def load_reports(
    *,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: Optional[int] = None
) -> List[TradingReport]:
    """
    조건에 맞는 레코드들을 불러온다. JSON 문자열은 리스트로 변환.
    - since: ISO 형식 날짜(YYYY-MM-DD) 이상
    - until: ISO 형식 날짜(YYYY-MM-DD) 이하
    - limit: 최대 반환 개수
    """
    _init_db()
    where_clauses: List[str] = []
    params: List[str | int] = []

    if since:
        where_clauses.append("run_date >= ?")
        params.append(since)
    if until:
        where_clauses.append("run_date <= ?")
        params.append(until)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    limit_sql = f"LIMIT {limit}" if limit else ""

    sql = f"SELECT * FROM {_TABLE_NAME} {where_sql} ORDER BY run_date DESC {limit_sql}"

    results = []
    list_fields = ['catalysts_short_term', 'catalysts_mid_term', 
                   'catalysts_long_term', 'key_monitoring_metrics']

    with _get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
        for row in rows:
            row_dict = dict(row) # sqlite3.Row를 dict로 변환
            # JSON 문자열 필드를 리스트로 변환 (다시 추가)
            for field_name in list_fields:
                if field_name in row_dict and isinstance(row_dict[field_name], str):
                    try:
                        row_dict[field_name] = json.loads(row_dict[field_name])
                    except json.JSONDecodeError:
                        # 파싱 실패 시 빈 리스트
                        row_dict[field_name] = [] 
            results.append(TradingReport(**row_dict))
    return results


###############################################################################
# 사용 예시 (직접 실행했을 때만)
###############################################################################

if __name__ == "__main__":
    # 저장 예시
    sample = TradingReport(
        ticker="QQQ",
        # 각 부서 취합 결과
        chart_report="MACD 골든크로스 발생, RSI 과매수 진입",
        research_report="애널리스트 3명 Buy 의견, 목표가 상향 조정 뉴스",
        financial_report="EPS +15% YoY, 매출 성장률 둔화",
        macro_report="미국 GDP 예상 상향, 금리 인상 가능성 잔존",
        # 요약 필드
        decision="보유",
        credibility="Medium",
        allocation_suggestion=15.0, # float으로 명시
        # 상세 근거
        investment_thesis="강력한 브랜드와 생태계 기반 안정적 성장 기대",
        valuation_assessment="현재 주가는 PER 밴드 상단, 약간 고평가",
        catalysts_short_term="신제품 출시 기대감",
        catalysts_mid_term="서비스 부문 성장 지속",
        catalysts_long_term="자율주행 및 AR/VR 신시장 진출 가능성",
        upside_potential="서비스 매출 가속화 시 추가 상승 가능",
        downside_risks="중국 시장 의존도, 반독점 규제 강화",
        contrarian_view="성장 둔화 우려로 단기 조정 가능성",
        investment_horizon="1년 이상 중장기",
        key_monitoring_metrics="iPhone 판매량, 서비스 매출 성장률, 마진율",
        exit_strategy="PER 40배 초과 또는 핵심 성장 동력 훼손 시 매도 고려",
        # 전체 서술형 보고서
        detail_report="현재 보유 비중 유지하며 시장 상황 모니터링 필요. 단기적 변동성 확대 가능성 있으나 장기 성장성은 유효.",
        current_price=10_000_000,
        current_return_pct=12.0,
        average_purchase_price=10500,
    )
    save_report(sample)

    # 조회 예시
    for rec in load_reports(limit=5):
        print(rec)
