import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 시가총액 Top 10 글로벌 기업 (티커: 기업명)
TOP10_COMPANIES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet (Google)",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "BRK-B": "Berkshire Hathaway",
    "TSLA": "Tesla",
    "META": "Meta Platforms",
    "TSM": "Taiwan Semiconductor",
    "LLY": "Eli Lilly"
}

st.set_page_config(page_title="글로벌 Top10 기업 주가 시각화", layout="wide")
st.title("🌍 글로벌 시가총액 Top 10 기업 주가 분석")
st.markdown("최근 1년간 주가와 누적 수익률을 확인할 수 있습니다.")

# 사용자 종목 선택
selected_tickers = st.multiselect(
    "기업을 선택하세요 (복수 선택 가능):",
    options=list(TOP10_COMPANIES.keys()),
    format_func=lambda x: f"{TOP10_COMPANIES[x]} ({x})",
    default=["AAPL", "MSFT"]
)

if not selected_tickers:
    st.warning("하나 이상의 기업을 선택하세요.")
    st.stop()

# 날짜 범위 설정
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

@st.cache_data(show_spinner=False)
def load_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)

    # 복수 종목: MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        for col in ["Adj Close", "Close"]:
            if col in df.columns.get_level_values(0):
                df = df[col]
                break
        else:
            raise ValueError("'Adj Close' 또는 'Close' 컬럼이 없습니다.")
    else:
        # 단일 종목: 일반 DataFrame
        if "Adj Close" in df.columns:
            df = df[["Adj Close"]]
        elif "Close" in df.columns:
            df = df[["Close"]]
        else:
            raise ValueError("'Adj Close' 또는 'Close' 컬럼이 없습니다.")
        name = tickers if isinstance(tickers, str) else tickers[0]
        df.columns = [name]  # 단일 종목 이름 맞춤

    df.dropna(inplace=True)
    return df

try:
    # 데이터 로드 및 누적 수익률 계산
    df_prices = load_data(selected_tickers, start_date, end_date)
    df_returns = (df_prices / df_prices.iloc[0]) - 1

    # 주가 그래프
    st.subheader("📊 주가 추이 (USD)")
    fig_price = go.Figure()
    for ticker in df_prices.columns:
        fig_price.add_trace(go.Scatter(
            x=df_prices.index, y=df_prices[ticker],
            mode='lines', name=TOP10_COMPANIES.get(ticker, ticker)))
    fig_price.update_layout(xaxis_title="날짜", yaxis_title="가격 (USD)", hovermode="x unified")
    st.plotly_chart(fig_price, use_container_width=True)

    # 누적 수익률 그래프
    st.subheader("📈 누적 수익률")
    fig_return = go.Figure()
    for ticker in df_returns.columns:
        fig_return.add_trace(go.Scatter(
            x=df_returns.index, y=df_returns[ticker],
            mode='lines', name=TOP10_COMPANIES.get(ticker, ticker)))
    fig_return.update_layout(xaxis_title="날짜", yaxis_title="수익률", hovermode="x unified")
    st.plotly_chart(fig_return, use_container_width=True)

except Exception as e:
    st.error(f"❌ 데이터 로딩 오류: {e}")
