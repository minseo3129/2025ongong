@st.cache_data(show_spinner=False)
def load_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)

    # 다중 종목: MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        valid_field = None
        for field in ["Adj Close", "Close"]:
            if field in df.columns.get_level_values(0):
                valid_field = field
                break
        if not valid_field:
            raise ValueError("다운로드한 데이터에 'Adj Close' 또는 'Close' 컬럼이 없습니다.")

        df_out = df[valid_field]

    else:
        # 단일 종목: 단일 Index
        if "Adj Close" in df.columns:
            df_out = df[["Adj Close"]]
        elif "Close" in df.columns:
            df_out = df[["Close"]]
        else:
            raise ValueError("단일 종목 데이터에 'Adj Close' 또는 'Close' 컬럼이 없습니다.")

        # 컬럼 이름을 티커로 설정
        df_out.columns = [tickers if isinstance(tickers, str) else tickers[0]]

    df_out.dropna(inplace=True)
    return df_out
