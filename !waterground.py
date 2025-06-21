import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import percentileofscore

# ------------------------
# 1. 데이터 로딩 및 전처리
# ------------------------
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("Drinking_Water_Quality_Distribution_Monitoring_Data.csv")
    df.columns = df.columns.str.replace(" ", "_").str.replace("(", "").str.replace(")", "")
    
    df = df.rename(columns={
        "Residual_Free_Chlorine_mg/L": "Chlorine",
        "Turbidity_NTU": "Turbidity",
        "Coliform_Quanti-Tray_MPN_/100mL": "Coliform",
        "E.coliQuanti-Tray_MPN/100mL": "Ecoli",
        "Fluoride_mg/L": "Fluoride",
        "Sample_Date": "Date",
        "Sample_Class": "Sample_Class"
    })

    def convert_text(v):
        if isinstance(v, str):
            v = v.strip().replace("<", "").replace(">", "").replace("+", "")
            if v.lower() == "nd":
                return np.nan
        try:
            return float(v)
        except:
            return np.nan

    for col in ["Turbidity", "Chlorine", "Coliform", "Ecoli", "Fluoride"]:
        df[col] = df[col].apply(convert_text)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Sample_Class", "Turbidity", "Chlorine", "Coliform", "Ecoli"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    return df

df = load_and_clean_data()

# ------------------------
# 2. Streamlit 설정
# ------------------------
st.set_page_config(layout="wide")
st.title("💧 NYC 수돗물 수질 진단 및 분석 대시보드")
st.caption("🎯 SDG 6 기반 수질 분석 · 위험 진단 · 사용자 맞춤 피드백 제공")

# ------------------------
# 3. 샘플 유형별 수질 분석
# ------------------------
st.header("① 샘플 유형별 수질 안정성 비교")

with st.expander("🔍 Compliance vs Operational 평균 비교"):
    summary = df.groupby("Sample_Class")[["Turbidity", "Chlorine", "Coliform", "Ecoli"]].agg(["mean", "std"])
    st.dataframe(summary)

# ------------------------
# 4. 기준 초과율 계산 및 시각화
# ------------------------
st.header("② 기준 초과 항목 비율 분석")

df["Turbidity_Exceed"] = df["Turbidity"] > 5
df["Chlorine_Low"] = df["Chlorine"] < 0.2
df["Coliform_Positive"] = df["Coliform"] > 0
df["Ecoli_Positive"] = df["Ecoli"] > 0

exceed_df = df[["Turbidity_Exceed", "Chlorine_Low", "Coliform_Positive", "Ecoli_Positive"]]
st.bar_chart(exceed_df.mean() * 100)

# ------------------------
# 5. 월별 수질 시계열
# ------------------------
st.header("③ 월별 수질 변화 추이")

monthly_avg = df.groupby("Month")[["Turbidity", "Chlorine"]].mean().reset_index()
fig1 = px.line(monthly_avg, x="Month", y="Turbidity", title="📈 월별 평균 탁도")
fig2 = px.line(monthly_avg, x="Month", y="Chlorine", title="📈 월별 평균 잔류 염소 농도")

st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)

# ------------------------
# 6. 사용자 진단 + 백분위 시각화
# ------------------------
st.header("④ 사용자 수질 입력 기반 진단 시스템")

turb = st.slider("탁도 (NTU)", 0.0, 10.0, 4.0)
chl = st.slider("잔류 염소 (mg/L)", 0.0, 1.0, 0.3)
coli = st.slider("대장균 (MPN/100mL)", 0, 10, 1)
fluor = st.slider("불소 농도 (mg/L)", 0.0, 2.0, 1.0)

def get_percentile(colname, value):
    col = df[colname].dropna()
    return percentileofscore(col, value, kind="mean")

with st.expander("📊 NYC 전체 분포 내 위치"):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("탁도 백분위", f"{get_percentile('Turbidity', turb):.1f} %")
        st.metric("염소 백분위", f"{get_percentile('Chlorine', chl):.1f} %")
    with col2:
        st.metric("대장균 백분위", f"{get_percentile('Coliform', coli):.1f} %")
        st.metric("불소 백분위", f"{get_percentile('Fluoride', fluor):.1f} %")

# ------------------------
# 7. 위험 피드백
# ------------------------
st.subheader("📋 진단 결과 및 관리 방안")

issues = []
if turb > 5:
    issues.append("🟠 **탁도 초과**: 병원균 보호 가능성 ↑ → 여과 시스템 점검 권장")
if chl < 0.2:
    issues.append("🔴 **염소 부족**: 소독력 저하 우려 → 말단 소독 강화 필요")
if coli > 0:
    issues.append("🔴 **대장균 검출**: 유입 경로 점검 및 재검사 필요")
if fluor > 1.5:
    issues.append("🟡 **불소 과다**: 투입량 재조정 필요")

if issues:
    for msg in issues:
        st.warning(msg)
else:
    st.success("✅ 모든 항목이 안전 기준 내에 있습니다.")

# ------------------------
# 8. 마무리 문구
# ------------------------
st.markdown("---")
st.caption("📘 WHO 기준 및 SDG 6 목표 기반: 수질 안전 + 데이터 기반 의사결정 시스템")

