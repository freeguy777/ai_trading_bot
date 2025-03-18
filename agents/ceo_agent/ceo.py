# agents/ceo_agent/ceo.py
import time
import os
import sys
import datetime
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from math import floor
from dotenv import load_dotenv

# --- 경로 설정 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
sys.path.append(os.path.dirname(project_root)) # Optional depending on exact structure and how you run

try:
    # 각 부서의 분석 함수 임포트
    from agents.chart_analysis_agent.main import main as get_chart_report
    from agents.research_agent.main import analyze_stock as get_research_report
    from agents.financial_health_agent.main import run_financial_analysis as get_financial_report
    from agents.macro_economic_agent.main import run_macro_economic_analysis as get_macro_report
    from kis_trader.main import get_account_profit_rate, get_average_price, get_present_balance, execute_kis_order, get_current_price, get_holding_amount, get_order_sell_quantity, get_total_cash_usd
    from memory_SQL.main import TradingReport, save_report
    from final_analysis import perform_final_analysis

except ImportError as e:
    print(f"Error importing analysis modules: {e}")
    print("Please ensure the project structure is correct and all necessary __init__.py files exist.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1) # 오류 발생 시 종료

class CEO_Agent:
    def __init__(self):
        print("CEO Agent Initialized.")
        # init_db() 호출 제거
    

    
    def request_analysis_and_decide(self, ticker: str):
        """
        CEO로서 각 부서에 분석을 요청하고, 취합된 보고서를 바탕으로 최종 결정을 내립니다.
        """
        print(f"\n===== CEO: {ticker}에 대한 분석을 지시합니다. ({datetime.now()}) =====")

        # 1. 각 부서에 분석 요청
        chart_report     = self._dispatch("차트 분석 부서",      get_chart_report,     ticker,        ticker=ticker)
        research_report  = self._dispatch("리서치 분석 부서",    get_research_report,  ticker, 5, False, ticker=ticker)
        financial_report = self._dispatch("재무제표 분석 부서",  get_financial_report, ticker,        ticker=ticker)
        macro_report     = self._dispatch("매크로 분석 부서",    get_macro_report,                   ticker=ticker)

        # 5. 최종 분석 부서(LLM)에 종합 분석 및 결정 요청
        print("📀 CEO: 수신된 보고서들을 종합하여 최종 투자 판단을 요청합니다.")
        final_decision_data = {}
        try:
            holding_shares  = get_holding_amount(ticker, exchange_name="나스닥")  #보유 수량
            average_price = get_average_price(ticker, exchange_name="나스닥")          #평단가
            final_decision_data = perform_final_analysis(chart_report, research_report, financial_report, macro_report, ticker, holding_shares, average_price)
            if "오류" in final_decision_data:
                raise ValueError(final_decision_data["오류"])
        except Exception as e:
            print(f"CEO: 최종 분석 중 오류 발생: {e}")
            error_message = (
                f"## 최종 분석 오류\n종합 분석 중 예외 발생: {e}\n\n"
                f"### 차트 보고서:\n{chart_report}\n\n"
                f"### 리서치 보고서:\n{research_report}"
            )
            final_decision_data = {"투자결정": "Error","신뢰도": "N/A","포지션비중": "N/A","전체분석요약": error_message,}
        
        # 6. 매매 진행
        print(f"CEO: 매매 진행 바랍니다.")
        order_qty = self.calculate_order_quantity(final_decision_data.get('포지션비중', 'N/A'), ticker)
        if order_qty > 0 :
            success = execute_kis_order(ticker, order_qty, "시장가", "매수", "나스닥")
            success =1
            print(f"{ticker} : {order_qty}주 매매 결과: {'매수 성공' if success else '실패'}")
        elif order_qty < 0 :
            success = execute_kis_order(ticker, abs(order_qty), "시장가", "매도", "나스닥")
            success =1
            print(f"{ticker} : {order_qty}주 매매 결과: {'매도 성공' if success else '실패'}")
        elif order_qty == 0 :
            print(f"{ticker}매매 결과: 보유")
        
        time.sleep(15)  #매수 진행 시간

        # 7. SQL 저장
        print(f"💿 CEO: 분석결과 저장(SQL)")
        sample = TradingReport(
            run_date=datetime.now().isoformat(),
            ticker=ticker,
            # 각 부서 취합 결과
            chart_report=chart_report,
            research_report=research_report,
            financial_report=financial_report,
            macro_report=macro_report,
            # 요약 필드
            decision = final_decision_data.get('투자결정', 'N/A'),
            credibility = final_decision_data.get('신뢰도', 'N/A'),
            allocation_suggestion = final_decision_data.get('포지션비중', 'N/A'),
            # 11가지 상세 근거
            investment_thesis = final_decision_data.get('핵심투자논거','N/A'),
            valuation_assessment = final_decision_data.get('적정가치평가','N/A'),
            catalysts_short_term = final_decision_data.get('단기촉매','N/A'), 
            catalysts_mid_term = final_decision_data.get('중기촉매','N/A'), 
            catalysts_long_term = final_decision_data.get('장기촉매','N/A'),
            upside_potential = final_decision_data.get('상승잠재력','N/A'),
            downside_risks = final_decision_data.get('하방리스크','N/A'),
            contrarian_view = final_decision_data.get('차별화관점','N/A'),
            investment_horizon = final_decision_data.get('투자기간','N/A'),
            key_monitoring_metrics = final_decision_data.get('핵심모니터링지표','N/A'),
            exit_strategy = final_decision_data.get('투자철회조건','N/A'),                
            # 전체 서술형 보고서
            detail_report = final_decision_data.get('전체분석요약', '보고서 내용 없음'),
            current_price = get_current_price(ticker),                                                          # 현재 주식 금액
            current_return_pct = get_account_profit_rate(),                                                     # 현재 계좌 수익률
            average_purchase_price = get_average_price(ticker, exchange_name="나스닥"),                         # 평단가
        )
        save_report(sample)


        # 8. SQL 출력
        # print(f"CEO: 매매 결과(SQL)")
        # for rec in load_reports(limit=3):
        #     print(rec)

        # 결과 출력
        self.print_investment_decision(ticker, final_decision_data)

        return final_decision_data
    
    def calculate_order_quantity(self, allocation: float, ticker: str) -> int:
        """
        allocation : float
        ticker     : str  
        Returns  : int 
        """
        pct = allocation/100
        total_cash_withholdings_usd    = float(get_total_cash_usd(exchange_name="나스닥"))
        price           = float(get_current_price(ticker, "나스닥"))
        holding_shares  = int(get_holding_amount(ticker, exchange_name="나스닥"))
        sellable_shares = int(get_order_sell_quantity(ticker, exchange_name="나스닥"))

        # ② 목표 주식 수
        target_shares  = floor(total_cash_withholdings_usd * pct / price)

        # ③ 실제 주문 수량(양수=매수, 음수=매도)
        order_qty = target_shares - holding_shares
        if order_qty > 0:       # 매수
            return order_qty
        elif order_qty < 0:     #매도
            return -min(abs(order_qty), sellable_shares)

        return 0

    def _dispatch(self, dept_name: str, func, *args, ticker: str) -> str:
        print(f"📀 CEO: {dept_name}, {ticker} 분석 보고 바랍니다.")
        try:
            report = func(*args)
            print(f"💿 CEO: {dept_name} 보고서 수신 완료.")
        except Exception as e:
            report = f"## {ticker} {dept_name} 오류\n분석 중 예외 발생: {e}"
        return report
    
    def print_investment_decision(self, ticker, final_decision_data):
        """
        최종 투자 판단 결과를 출력합니다.
        """
        print("\n===================================================")
        print(f"    📈 최종 투자 판단 결과 ({ticker})    ")
        print("===================================================")
        print(f"▶ 투자 결정      : {final_decision_data.get('투자결정', 'N/A')}")
        print(f"▶ 신뢰도         : {final_decision_data.get('신뢰도', 'N/A')}")
        print(f"▶ 투자 비중 제안 : {final_decision_data.get('포지션비중', 'N/A')}")
        print("---------------------------------------------------")
        print("▶ 상세 보고서:")
        print(final_decision_data.get('전체분석요약', '보고서 내용 없음'))
        print("---------------------------------------------------")
        print(f"▶ 핵심 투자 논거: {final_decision_data.get('핵심투자논거', 'N/A')}")
        print(f"▶ 적정 가치 평가: {final_decision_data.get('적정가치평가', 'N/A')}")
        print("▶ 촉매 요인:")
        print(f" - 단기: {', '.join(final_decision_data.get('단기촉매', ['N/A']))}")
        print(f" - 중기: {', '.join(final_decision_data.get('중기촉매', ['N/A']))}")
        print(f" - 장기: {', '.join(final_decision_data.get('장기촉매', ['N/A']))}")
        print(f"▶ 상승 잠재력: {final_decision_data.get('상승잠재력', 'N/A')}")
        print(f"▶ 하방 리스크: {final_decision_data.get('하방리스크', 'N/A')}")
        print(f"▶ 차별화된 관점: {final_decision_data.get('차별화관점', 'N/A')}")
        print(f"▶ 투자 기간: {final_decision_data.get('투자기간', 'N/A')}")
        print(f"▶ 핵심 모니터링 지표: {', '.join(final_decision_data.get('핵심모니터링지표', ['N/A']))}")
        print(f"▶ 투자 철회 조건: {final_decision_data.get('투자철회조건', 'N/A')}")
        print("===================================================\n")

        print("\n===================================================")
        print(f"   🏧 계좌 정보 ({ticker})    ")
        print("===================================================")
        print(f"[현재가] {ticker}: {get_current_price(ticker, exchange_name='나스닥')}")
        print(f"[전체 수익률] {get_account_profit_rate()}%")
        print(f"[평단가] {ticker}: {get_average_price(ticker, exchange_name='나스닥')}")
        print(f"[가용 예수금] {get_present_balance(exchange_name='나스닥')}")
        print("--- 잔고 조회 테스트 종료 ---")

def _trade(ticker: str = "QQQ"):
    """Wrapper so APScheduler can call a clean function."""
    ceo.request_analysis_and_decide(ticker)

# --- 실행 코드 ---
if __name__ == "__main__":
    load_dotenv("config/.env")
    ticker = os.getenv("TICKER")
    ceo = CEO_Agent()
    ticker = "NVDA"
    # ── Scheduler setup ──────────────────────────────────────────────────────
    seoul   = ZoneInfo("Asia/Seoul")          # your wall-clock
    eastern = ZoneInfo("America/New_York")    # NYSE clock (DST aware)
    sched   = BlockingScheduler(timezone=seoul)
    
    # 테스트 끝나면 주석처리 
    # sched.add_job(
    #     _trade,
    #     trigger="date",                        # DateTrigger
    #     run_date=datetime.now(seoul) + timedelta(seconds=3),
    #     id="one_off_smoke_test",
    #     kwargs={"ticker": ticker},
    # )

    # 30 min after US market open ⇒ 10:00 ET, 월–금만
    sched.add_job(
        _trade,
        trigger=CronTrigger(
            day_of_week="mon-fri",      
            hour=10,
            minute=0,
            timezone=eastern,
        ),
        id="open_plus_30",
        kwargs={"ticker": ticker},
    )

    # 30 min before close ⇒ 15:30 ET, 월–금만
    '''
    sched.add_job(
        _trade,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=15,
            minute=30,
            timezone=eastern,
        ),
        id="close_minus_30",
        kwargs={"ticker": ticker},
    )
    '''

    print(
        "Scheduler armed: 10:00 & 15:30 America/New_York "
        "(= 23:00 & 04:30 KST during DST, 00:00 & 05:30 in winter)."
    )
    sched.start()


#작업환경 맞추기 : pipenv install --dev
#코드 실행 : nohup python3 -u agents/ceo_agent/ceo.py > output.log 2>&1 &
#streamlit 실행 : nohup python3 -m streamlit run ./make_streamlit_clode/main.py > streamlit.log 2>&1 &
#실행 확인 : ps ax | grep .py
#종료하기 ex. kill -9 13586