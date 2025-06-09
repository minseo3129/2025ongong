import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 글로벌 시가총액 Top 10 기업 (티커: 이름)
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

st.title("📈 글로벌 시가총액 Top 10 기업 주가 분석")
st.markdown("**최근 1년간 주가 및 누적 수익률을 시각화합니다.**")

# 기업 선택
selected_tickers = st.multiselect(
    "기업을 선택하세요 (복수 선택 가능):",
    options=list(TOP10_COMPANIES.keys()),
    format_func=lambda x: f"{TOP10_COMPANIES[x]} ({x})",
    default=["AAPL", "MSFT"]
)

if not selected_tickers:
    st.warning("적어도 하나의 기업을 선택하세요.")
    st.stop()

# 날짜 범위 설정
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

@st.cache_data(show_spinner=False)
def load_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end)["Adj Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df.dropna(inplace=True)
    return df

try:
    df_prices = load_data(selected_tickers, start_date, end_date)
    df_returns = (df_prices / df_prices.iloc[0]) - 1  # 누적 수익률

    # 주가 그래프
    st.subheader("📊 주가 추이")
    fig_price = go.Figure()
    for ticker in df_prices.columns:
        fig_price.add_trace(go.Scatter(x=df_prices.index, y=df_prices[ticker],
                                       mode='lines', name=TOP10_COMPANIES.get(ticker, ticker)))
    fig_price.update_layout(title="주가 변화", xaxis_title="날짜", yaxis_title="가격 (USD)")
    st.plotly_chart(fig_price, use_container_width=True)

    # 누적 수익률 그래프
    st.subheader("📈 누적 수익률")
    fig_return = go.Figure()
    for ticker in df_returns.columns:
        fig_return.add_trace(go.Scatter(x=df_returns.index, y=df_returns[ticker],
                                        mode='lines', name=TOP10_COMPANIES.get(ticker, ticker)))
    fig_return.update_layout(title="누적 수익률 변화", xaxis_title="날짜", yaxis_title="누적 수익률")
    st.plotly_chart(fig_return, use_container_width=True)

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류 발생: {e}")
