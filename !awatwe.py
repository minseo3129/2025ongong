import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -------------------------------
# 1. 데이터 로딩 및 정제 함수
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Drinking_Water_Quality_Distribution_Monitoring_Data.csv")
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")

    df = df.rename(columns={
        "Residual_Free_Chlorine_mg/L": "Chlorine",
        "Turbidity_NTU": "Turbidity",
        "Coliform_Quanti-Tray_MPN_/100mL": "Coliform",
        "E.coliQuanti-Tray_MPN/100mL": "Ecoli",
        "Fluoride_mg/L": "Fluoride",
        "Sample_Date": "Date"
    })

    # Sample_class 통일
    for col in ["Sample_class", "Sample_Class", "SampleClass"]:
        if col in df.columns:
            df = df.rename(columns={col: "Sample_Class"})
            break

    def to_num(x):
        if isinstance(x, str):
            x = x.replace("<", "").replace(">", "").replace("+", "").strip()
            if x.lower() == "nd":
                return np.nan
        try:
            return float(x)
        except:
            return np.nan

    for col in ["Turbidity", "Chlorine", "Coliform", "Ecoli", "Fluoride"]:
        df[col] = df[col].apply(to_num)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Sample_Class", "Turbidity", "Chlorine", "Coliform", "Ecoli"])
    df["Month"] = df["Date"].dt.month
    return df

# -------------------------------
# 2. Streamlit 구성
# -------------------------------
df = load_data()

st.set_page_config(layout="wide")
st.title("💧 수질 기준 초과율 분석: Sample Class × Month")
st.caption("🔍 운영 목적 및 시기에 따라 NYC 수돗물 수질 악화 경향을 분석합니다.")

# -------------------------------
# 3. 사용자 설정
# -------------------------------
st.sidebar.header("⚙️ 분석 항목 선택")
indicator = st.sidebar.selectbox("수질 항목", ["Turbidity", "Chlorine", "Coliform", "Ecoli"])

# 기준값 자동 설정
if indicator == "Turbidity":
    threshold = 5
    operator = ">"
    exceed_cond = df[indicator] > threshold
elif indicator == "Chlorine":
    threshold = 0.2
    operator = "<"
    exceed_cond = df[indicator] < threshold
else:
    threshold = 0
    operator = ">"
    exceed_cond = df[indicator] > threshold

df["Exceed"] = exceed_cond

# -------------------------------
# 4. 그룹별 기준 초과율 계산
# -------------------------------
summary = df.groupby(["Sample_Class", "Month"])["Exceed"].mean().reset_index()
summary["Exceed"] = summary["Exceed"] * 100  # %
summary["Month"] = summary["Month"].astype(int)

# -------------------------------
# 5. 시각화
# -------------------------------
st.subheader(f"📊 기준 초과율: {indicator} {operator} {threshold}")
fig = px.line(
    summary,
    x="Month",
    y="Exceed",
    color="Sample_Class",
    markers=True,
    labels={"Exceed": "초과율 (%)", "Month": "월"},
    title=f"{indicator} {operator} {threshold} 기준 초과율 - Sample Class별 월별 비교"
)
fig.update_layout(xaxis=dict(tickmode="linear"))
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 6. 해석 및 정책적 의의
# -------------------------------
st.markdown("### 🧠 분석 해석")
st.markdown(f"- 선택한 항목 **{indicator}**에 대해 **{operator} {threshold}** 기준 초과율을 시각화했습니다.")
st.markdown("- **Sample Class**가 'Operational'인 경우 반복적인 초과가 발생하면, 운영 방식의 점검 필요성이 있습니다.")
st.markdown("- **월별 패턴**이 뚜렷하면, 시기별 수질 관리(예: 여름철 염소 강화)가 필요합니다.")
st.markdown("- 이 분석은 섬진강댐 연구처럼 시공간적 수질 위험 인자를 정량화하여 정책적 대응에 활용 가능합니다.")

st.markdown("---")
st.caption("📘 지속가능발전목표(SDG 6) · WHO 수질 기준 기반 정책 분석 도구")
