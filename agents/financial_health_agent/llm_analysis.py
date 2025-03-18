# llm_analysis.py
import json
from typing import Dict, List, Any
import os
# 사용할 LLM 라이브러리 임포트
import google.generativeai as genai

# --- format_financial_data_for_llm 함수는 이전과 동일 ---
def format_financial_data_for_llm(financial_data: Dict[str, List[Dict[str, Any]]], ticker: str) -> str:
    """
    가져온 재무 데이터를 LLM 프롬프트에 적합한 문자열 형식으로 변환합니다.
    LLM이 더 쉽게 읽을 수 있도록 주요 지표를 추출합니다.
    """
    output_str = f"--- {ticker} 재무 데이터 (최근 {len(next(iter(financial_data.values()), []))}년) ---\n\n"

    # LLM에게 전달할 주요 지표 정의 (필요에 따라 수정)
    key_metrics = {
        "income": ['date', 'symbol', 'reportedCurrency', 'revenue', 'costOfRevenue', 'grossProfit', 'grossProfitRatio',
                   'researchAndDevelopmentExpenses', 'generalAndAdministrativeExpenses', 'sellingAndMarketingExpenses',
                   'sellingGeneralAndAdministrativeExpenses', 'otherExpenses', 'operatingExpenses',
                   'costAndExpenses', 'interestIncome', 'interestExpense', 'depreciationAndAmortization',
                   'ebitda', 'ebitdaratio', 'operatingIncome', 'operatingIncomeRatio', 'totalOtherIncomeExpensesNet',
                   'incomeBeforeTax', 'incomeBeforeTaxRatio', 'incomeTaxExpense', 'netIncome', 'netIncomeRatio',
                   'eps', 'epsdiluted', 'weightedAverageShsOut', 'weightedAverageShsOutDil'],
        "balance": ['date', 'symbol', 'reportedCurrency', 'cashAndCashEquivalents', 'shortTermInvestments',
                    'cashAndShortTermInvestments', 'netReceivables', 'inventory', 'otherCurrentAssets',
                    'totalCurrentAssets', 'propertyPlantEquipmentNet', 'goodwill', 'intangibleAssets',
                    'goodwillAndIntangibleAssets', 'longTermInvestments', 'taxAssets', 'otherNonCurrentAssets',
                    'totalNonCurrentAssets', 'otherAssets', 'totalAssets', 'accountPayables', 'shortTermDebt',
                    'taxPayables', 'deferredRevenue', 'otherCurrentLiabilities', 'totalCurrentLiabilities',
                    'longTermDebt', 'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent',
                    'otherNonCurrentLiabilities', 'totalNonCurrentLiabilities', 'otherLiabilities',
                    'capitalLeaseObligations', 'totalLiabilities', 'preferredStock', 'commonStock',
                    'retainedEarnings', 'accumulatedOtherComprehensiveIncomeLoss', 'othertotalStockholdersEquity',
                    'totalStockholdersEquity', 'totalEquity', 'totalLiabilitiesAndStockholdersEquity',
                    'minorityInterest', 'totalLiabilitiesAndTotalEquity', 'totalInvestments',
                    'totalDebt', 'netDebt'],
        "cashflow": ['date', 'symbol', 'reportedCurrency', 'netIncome', 'depreciationAndAmortization',
                     'deferredIncomeTax', 'stockBasedCompensation', 'changeInWorkingCapital',
                     'accountsReceivables', 'inventory', 'accountsPayables', 'otherWorkingCapital',
                     'otherNonCashItems', 'netCashProvidedByOperatingActivities', 'investmentsInPropertyPlantAndEquipment',
                     'acquisitionsNet', 'purchasesOfInvestments', 'salesMaturitiesOfInvestments', 'otherInvestingActivites',
                     'netCashUsedForInvestingActivites', 'debtRepayment', 'commonStockIssued',
                     'commonStockRepurchased', 'dividendsPaid', 'otherFinancingActivites',
                     'netCashUsedProvidedByFinancingActivities', 'effectOfForexChangesOnCash', 'netChangeInCash',
                     'cashAtEndOfPeriod', 'cashAtBeginningOfPeriod', 'operatingCashFlow', 'capitalExpenditure',
                     'freeCashFlow']
    }


    for statement_type, statements_list in financial_data.items():
        output_str += f"## {statement_type.capitalize()} Statement ##\n"
        if not statements_list:
            output_str += "사용 가능한 데이터가 없습니다.\n\n"
            continue

        # 해당 재무제표에서 추출할 지표 목록 (key_metrics에 없으면 모든 키 사용)
        # metrics_to_extract = key_metrics.get(statement_type, list(statements_list[0].keys()) if statements_list else [])
        metrics_to_extract = key_metrics.get(statement_type, []) # 정의된 것만 사용하도록 변경

        # 추출할 지표가 없으면 건너뛰기
        if not metrics_to_extract:
            output_str += "분석할 주요 지표가 정의되지 않았습니다.\n\n"
            continue

        # 실제 데이터에 존재하는 지표만 필터링
        if statements_list:
            available_keys = set(statements_list[0].keys())
            metrics_to_extract = [m for m in metrics_to_extract if m in available_keys]

        # 헤더 출력 (열 정렬을 위해 패딩 추가)
        header = " | ".join([f"{metric:<25}" for metric in metrics_to_extract])
        output_str += header + "\n"
        output_str += "-" * (len(header) + len(metrics_to_extract) * 3 -1) + "\n" # 구분선 길이 조정

        # 각 연도의 데이터 출력 (최신순 정렬)
        for statement in sorted(statements_list, key=lambda x: x.get('date', ''), reverse=True):
            row_values = []
            for metric in metrics_to_extract:
                value = statement.get(metric) # 데이터가 없을 경우 None 반환됨
                # None 값 처리 및 숫자 포맷팅
                if value is None:
                    value_str = 'N/A'
                elif isinstance(value, (int, float)) and abs(value) > 1e6: # 백만 이상인 경우
                     value_str = f"{value:,.0f}" # 천 단위 쉼표, 소수점 없이
                elif isinstance(value, float):
                    value_str = f"{value:.2f}" # 소수점 2자리까지 표시 (비율 등에 적합)
                else:
                     value_str = str(value)
                row_values.append(f"{value_str:<25}") # 왼쪽 정렬 및 패딩
            output_str += " | ".join(row_values) + "\n"
        output_str += "\n"

    return output_str
# --- format_financial_data_for_llm 함수 끝 ---


def analyze_financials_with_llm(financial_data: Dict[str, List[Dict[str, Any]]], ticker: str) -> str:
    """
    제공된 재무 데이터를 Google Gemini를 사용하여 **심층적으로** 분석합니다.

    Args:
        financial_data (Dict[str, List[Dict[str, Any]]]): 재무제표 데이터 딕셔너리.
        ticker (str): 주식 티커 심볼.

    Returns:
        str: Gemini가 생성한 심층 분석 리포트 또는 오류 메시지.
    """
    print(f"{ticker} 재무 데이터 Gemini 심층 분석 시작...")

    # 1. Gemini가 이해하기 쉽도록 데이터 포맷팅
    formatted_data = format_financial_data_for_llm(financial_data, ticker)
    # print("\n--- Gemini 전송 데이터 ---") # 선택사항: 디버깅용 포맷된 데이터 출력
    # print(formatted_data)
    # print("--- Gemini 전송 데이터 끝 ---\n")

    # 2. 프롬프트 **대폭 강화**
    # 전문성과 통찰력을 요구하는 질문 추가
    prompt = f"""
**페르소나:** 당신은 월스트리트에서 활동하는 매우 경험 많은 주식 분석가이자 포트폴리오 매니저입니다. 당신의 강점은 재무제표의 숫자 이면에 숨겨진 의미를 파악하고, 기업의 실제 건강 상태와 잠재적 위험을 날카롭게 진단하는 것입니다. 특히 대차대조표 분석을 통해 기업의 자금 조달 방식, 부채의 질, 유동성 압박 등을 파악하는 데 능숙합니다. "무슨 돈으로?" 라는 질문을 항상 염두에 두고 분석합니다.

**미션:** 다음은 티커 {ticker} 회사의 최근 3년간 연간 재무 데이터입니다. 이 데이터를 **매우 비판적이고 심층적으로 분석**하여 전문 투자자를 위한 보고서를 작성하십시오. 단순한 요약을 넘어, 통찰력 있는 해석과 잠재적 위험 신호를 명확히 제시해야 합니다.

{formatted_data}

**분석 지침 (다음 질문들에 구체적으로 답변하며 보고서 형식으로 작성):**

1.  **사업 및 재무 상태 요약:**
    *   이 회사의 주요 사업은 무엇으로 추정됩니까? (재무제표만으로 추론)
    *   최근 3년간의 전반적인 재무 성과(매출, 이익) 추세는 어떻습니까? 성장의 동력 혹은 부진의 원인은 무엇으로 보입니까?

2.  **수익성 분석 (추세 및 질적 평가):**
    *   매출 총이익률과 영업 이익률의 변화 추세는 어떻습니까? 이익률 변화의 원인은 무엇일 가능성이 높습니까? (예: 비용 구조 변화, 경쟁 심화, 가격 정책 변화 등)
    *   순이익과 영업 현금 흐름 간의 관계는 어떻습니까? 큰 차이가 있다면 그 이유는 무엇일까요? (예: 감가상각비, 운전 자본 변동 등) 이익의 질은 양호합니까?

3.  **대차대조표 심층 분석 (가장 중요):**
    *   **자산 구성 및 질:** 총자산의 성장 추세는 어떻습니까? 자산 증가는 주로 어떤 항목(유동/비유동, 유형/무형)에 의해 주도되었습니까? 특히 무형자산(Goodwill, Intangible Assets)의 비중과 변화에 주목하십시오. 이것이 시사하는 바는 무엇입니까?
    *   **부채 구조 및 위험:** 총부채와 순부채(Total Debt - Cash)의 변화 추세는 어떻습니까? 부채 증가는 주로 단기차입금과 장기차입금 중 어디서 발생했습니까? **단기 부채의 급격한 증가는 유동성 위험 신호일 수 있습니다. 이 점을 면밀히 분석하십시오.** 부채 비율(예: Debt-to-Equity)은 어떤 수준이며, 산업 평균과 비교할 때 (가정) 위험 수준은 어떻습니까?
    *   **자본 조달 방식 ("무슨 돈으로?"):** 회사는 성장을 위한 자금을 어떻게 조달하고 있습니까? (영업 현금 흐름, 유상증자(commonStock 변화), 부채 증가, 자산 매각 등). 자금 조달 방식의 건전성은 어떻게 평가합니까? 자기자본(Equity)의 변화 요인(순이익 누적(Retained Earnings), 자본금 변동 등)을 분석하십시오.
    *   **유동성 평가:** 유동비율(Current Ratio)과 당좌비율(Quick Ratio, 계산 가능 시)은 어떻습니까? (유동자산 / 유동부채, (유동자산-재고자산)/유동부채). 1년 내 갚아야 할 유동부채를 상환할 능력이 충분합니까? 운전 자본(Working Capital: 유동자산 - 유동부채)의 변화 추세는 긍정적입니까? **유동 부채가 유동 자산을 크게 초과하는 경우, 이는 심각한 위험 신호입니다.**

4.  **현금 흐름 분석:**
    *   영업 활동 현금 흐름(Operating Cash Flow)은 꾸준히 창출되고 있습니까? 순이익 대비 OCF 비율은 어떻습니까?
    *   투자 활동 현금 흐름(Investing Cash Flow)은 주로 어디에 사용되고 있습니까? (예: 설비 투자(Capital Expenditure), 기업 인수(Acquisitions)). 투자 규모는 적절합니까? 지속적인 대규모 투자가 있다면, 자금 조달 방식과 연계하여 분석하십시오.
    *   재무 활동 현금 흐름(Financing Cash Flow)은 무엇을 말해줍니까? (예: 부채 상환/차입, 자사주 매입(Common Stock Repurchased), 배당(Dividends Paid)). 회사의 재무 전략을 엿볼 수 있습니까?
    *   잉여 현금 흐름(Free Cash Flow: OCF - CapEx)은 어떻습니까? 꾸준히 양(+)의 값을 유지하고 있습니까? FCF는 기업의 실제 현금 창출 능력을 보여주는 중요한 지표입니다.

5.  **종합 평가 및 잠재적 위험:**
    *   위 분석들을 종합할 때, 이 회사의 **가장 큰 강점**은 무엇입니까?
    *   이 회사가 직면한 **가장 심각한 잠재적 위험** 또는 **"Red Flags"** 는 무엇이라고 판단합니까? (예: 과도한 단기 부채, 악화되는 수익성, 공격적인 회계 처리 가능성, 자금 조달의 어려움 등) 구체적인 근거를 제시하십시오.
    *   마치 '멈추면 쓰러지는 자전거'처럼 보이는 위험한 신호는 없습니까?

6.  **투자 의견 (제한적 정보 기반):**
    *   **오직 제공된 3년 재무 데이터만을 바탕으로** 당신의 전문적인 투자 의견(예: 강력 매수 고려, 관망/보유, 비중 축소/매도 고려, 분석 불가/정보 부족)을 제시하십시오.
    *   투자 의견의 핵심 근거와 가장 우려되는 위험 요인을 명확히 밝히십시오. 3년 데이터만으로는 완전한 판단이 어렵다는 점을 반드시 명시하십시오.

**보고서 형식:** 각 분석 항목(1~6)에 명확한 제목을 붙이고, 간결하면서도 논리적인 문장으로 작성하십시오. 숫자를 근거로 제시하되, 단순 나열이 아닌 **해석과 통찰력**을 보여주십시오.
"""

    # 3. Gemini API 호출 (이전과 동일)
    try:
        # .env 파일 또는 환경 변수에서 API 키 가져오기
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            return "오류: GEMINI_API_KEY가 환경 변수에 설정되지 않았습니다. .env 파일을 확인하세요."

        # Google API 키 설정
        genai.configure(api_key=gemini_api_key)

        # 사용할 Gemini 모델 설정
        model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17') # 모델명 확인 및 필요시 수정

        # 안전 설정 추가 (Gemini에서 특정 콘텐츠 차단 방지)
        # 필요에 따라 조정. HARM_CATEGORY_HARASSMENT 등 다른 카테고리도 추가 가능
        safety_settings = [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
             {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
             {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            }
        ]

        # Gemini API 호출하여 콘텐츠 생성 (안전 설정 포함)
        response = model.generate_content(prompt, safety_settings=safety_settings)

        # 응답 텍스트 추출
        # Gemini API 응답 구조 확인 필요 (response.text가 맞는지)
        # 때로는 response.parts[0].text 또는 다른 구조일 수 있음
        try:
            analysis_report = response.text
        except ValueError: # 가끔 response.text 접근 시 에러 발생 가능 (차단 등)
             print("Gemini 응답 파싱 오류. 전체 응답 객체 확인:")
             print(response)
             # 차단된 경우 원인 확인
             if response.prompt_feedback.block_reason:
                 print(f"Gemini 요청 차단됨: {response.prompt_feedback.block_reason}")
                 return f"Gemini 요청이 차단되었습니다. 이유: {response.prompt_feedback.block_reason}. 프롬프트나 안전 설정을 확인하세요."
             else:
                 return "Gemini 응답에서 텍스트를 추출하는 데 실패했습니다."

        print("Gemini 심층 분석 완료.")
        # Gemini가 빈 응답을 반환하는 경우 대비
        return analysis_report if analysis_report else "Gemini가 빈 응답을 반환했습니다."

    except Exception as e:
         # API 호출 중 오류 발생 시
         print(f"Gemini API 호출 오류: {e}")
         # 오류 메시지와 함께 포맷된 데이터 반환 (디버깅 목적)
         return f"Gemini 분석 중 오류 발생: {e}\n\n포맷된 데이터:\n{formatted_data}"


if __name__ == '__main__':
     # 예제 사용법 (get_financial.py 등으로 재무 데이터가 필요함)
     # 여기서는 시연을 위해 더미 데이터 생성
     print("llm_analysis.py 단독 실행 예제...")
     dummy_ticker = "XYZ"
     dummy_data = {
         "income": [
             {'date': '2023-12-31', 'symbol': 'XYZ', 'reportedCurrency': 'USD', 'revenue': 120000, 'grossProfit': 50000, 'operatingIncome': 25000, 'netIncome': 15000, 'eps': 1.5, 'operatingIncomeRatio': 0.2083, 'netIncomeRatio': 0.125, 'weightedAverageShsOut': 10000},
             {'date': '2022-12-31', 'symbol': 'XYZ', 'reportedCurrency': 'USD', 'revenue': 100000, 'grossProfit': 40000, 'operatingIncome': 20000, 'netIncome': 10000, 'eps': 1.0, 'operatingIncomeRatio': 0.2000, 'netIncomeRatio': 0.1000, 'weightedAverageShsOut': 10000},
             {'date': '2021-12-31', 'symbol': 'XYZ', 'reportedCurrency': 'USD', 'revenue': 90000, 'grossProfit': 35000, 'operatingIncome': 18000, 'netIncome': 9000, 'eps': 0.9, 'operatingIncomeRatio': 0.2000, 'netIncomeRatio': 0.1000, 'weightedAverageShsOut': 10000},
         ],
         "balance": [
             {'date': '2023-12-31', 'symbol': 'XYZ', 'reportedCurrency': 'USD', 'cashAndCashEquivalents': 20000, 'totalCurrentAssets': 60000, 'totalAssets': 150000, 'shortTermDebt': 10000, 'totalCurrentLiabilities': 30000, 'longTermDebt': 40000, 'totalLiabilities': 70000, 'commonStock': 5000, 'retainedEarnings': 75000, 'totalStockholdersEquity': 80000, 'totalLiabilitiesAndStockholdersEquity': 150000, 'totalDebt': 50000, 'netDebt': 30000},
             {'date': '2022-12-31', 'symbol': 'XYZ', 'reportedCurrency': 'USD', 'cashAndCashEquivalents': 15000, 'totalCurrentAssets': 50000, 'totalAssets': 130000, 'shortTermDebt': 8000, 'totalCurrentLiabilities': 25000, 'longTermDebt': 35000, 'totalLiabilities': 60000, 'commonStock': 5000, 'retainedEarnings': 65000, 'totalStockholdersEquity': 70000, 'totalLiabilitiesAndStockholdersEquity': 130000, 'totalDebt': 43000, 'netDebt': 28000},
             {'date': '2021-12-31', 'symbol': 'XYZ', 'reportedCurrency': 'USD', 'cashAndCashEquivalents': 12000, 'totalCurrentAssets': 45000, 'totalAssets': 120000, 'shortTermDebt': 5000, 'totalCurrentLiabilities': 22000, 'longTermDebt': 30000, 'totalLiabilities': 55000, 'commonStock': 5000, 'retainedEarnings': 60000, 'totalStockholdersEquity': 65000, 'totalLiabilitiesAndStockholdersEquity': 120000, 'totalDebt': 35000, 'netDebt': 23000},
         ],
         "cashflow": [
             {'date': '2023-12-31', 'symbol': 'XYZ', 'reportedCurrency': 'USD', 'netIncome': 15000, 'depreciationAndAmortization': 5000, 'operatingCashFlow': 22000, 'capitalExpenditure': -8000, 'freeCashFlow': 14000, 'debtRepayment': -3000, 'commonStockIssued': 0, 'commonStockRepurchased': -1000, 'dividendsPaid': -2000, 'netCashUsedProvidedByFinancingActivities': -6000, 'netChangeInCash': 5000, 'cashAtBeginningOfPeriod': 15000, 'cashAtEndOfPeriod': 20000},
             {'date': '2022-12-31', 'symbol': 'XYZ', 'reportedCurrency': 'USD', 'netIncome': 10000, 'depreciationAndAmortization': 4500, 'operatingCashFlow': 18000, 'capitalExpenditure': -7000, 'freeCashFlow': 11000, 'debtRepayment': -2000, 'commonStockIssued': 0, 'commonStockRepurchased': -500, 'dividendsPaid': -1500, 'netCashUsedProvidedByFinancingActivities': -4000, 'netChangeInCash': 3000, 'cashAtBeginningOfPeriod': 12000, 'cashAtEndOfPeriod': 15000},
             {'date': '2021-12-31', 'symbol': 'XYZ', 'reportedCurrency': 'USD', 'netIncome': 9000, 'depreciationAndAmortization': 4000, 'operatingCashFlow': 16000, 'capitalExpenditure': -6000, 'freeCashFlow': 10000, 'debtRepayment': -1000, 'commonStockIssued': 0, 'commonStockRepurchased': 0, 'dividendsPaid': -1000, 'netCashUsedProvidedByFinancingActivities': -2000, 'netChangeInCash': 2000, 'cashAtBeginningOfPeriod': 10000, 'cashAtEndOfPeriod': 12000},
         ]
     }

     # 더미 데이터로 분석 함수 호출
     analysis = analyze_financials_with_llm(dummy_data, dummy_ticker)
     print("\n--- 생성된 심층 분석 (단독 실행 예제) ---")
     print(analysis)