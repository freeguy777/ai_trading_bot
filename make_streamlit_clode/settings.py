# 앱 기본 설정
APP_TITLE = "투자 분석 대시보드"
APP_ICON = "📊"

# 색상 테마
COLORS = {
    "primary": "#1E3A8A",    # 딥 블루
    "secondary": "#D4AF37",  # 골드
    "positive": "#10B981",   # 그린
    "negative": "#EF4444",   # 레드
    "neutral": "#6B7280",    # 그레이
    "background": "#F9FAFB", # 라이트 그레이
    "text": "#111827",       # 다크 그레이
}

# 투자결정 별 색상 및 아이콘
DECISION_STYLES = {
    "매수": {"color": "#10B981", "icon": "▲"},
    "보유": {"color": "#6B7280", "icon": "◆"},
    "매도": {"color": "#EF4444", "icon": "▼"},
}

# 신뢰도 레벨별 스타일
CREDIBILITY_LEVELS = {
    "High": {"stars": 5, "color": "#10B981"},
    "Medium": {"stars": 3, "color": "#F59E0B"},
    "Low": {"stars": 1, "color": "#EF4444"},
}

# CSS 클래스 정의를 위한 기본 스타일
CSS_STYLES = """
.header-container {
    padding: 1rem 0;
    background: linear-gradient(to right, #1E3A8A, #3B82F6);
    border-radius: 10px;
    margin-bottom: 1rem;
    text-align: center;
}
.header-container h1 {
    color: white !important;
    margin: 0;
}
.metric-card {
    background-color: white;
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}
.metric-title {
    font-size: 0.9rem;
    color: #6B7280;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: bold;
}
/* 나머지 CSS 스타일 정의 */
"""