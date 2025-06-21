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
        "Sample_Class": "Class"
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
    df = df.dropna(subset=["Date", "Class", "Turbidity", "Chlorine", "Coliform", "Ecoli"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    return df

df = load_and_clean_data()

st.set_page_config(layout="wide")
st.title("💧 NYC 수돗물 수질 진단 및 분석 대시보드")
st.caption("SDG 6 기반 수질 위험요소 감지 및 시민 참여형 진단 시스템")

# ------------------------
# 2. 샘플 유형별 수질 분석
# ------------------------
st.header("① 샘플 유형별 수질 안정성 비교")

with st.expander("🔍 Compliance vs Operational 수질 평균 비교"):
    summary = df.groupby("Class")[["Turbidity", "Chlorine", "Coliform", "Ecoli"]].agg(["mean", "std"])
    st.dataframe(summary)

# ------------------------
# 3. 기준 초과 항목 비율 분석
# ------------------------
st.header("② 기준 초과 항목 비율 분석")
df["Turbidity_Exceed"] = df["Turbidity"] > 5
df["Chlorine_Low"] = df["Chlorine"] < 0.2
df["Coliform_Positive"] = df["Coliform"] > 0
df["Ecoli_Positive"] = df["Ecoli"] > 0

exceed_df = df[["Turbidity_Exceed", "Chlorine_Low", "Coliform_Positive", "Ecoli_Positive"]]
st.bar_chart(exceed_df.mean() * 100)

# ------------------------
# 4. 월별 수질 추이 시각화
# ------------------------
st.header("③ 월별 수질 변화 추이")
monthly_avg = df.groupby("Month")[["Turbidity", "Chlorine"]].mean().reset_index()
fig1 = px.line(monthly_avg, x="Month", y="Turbidity", title="월별 평균 탁도")
fig2 = px.line(monthly_avg, x="Month", y="Chlorine", title="월별 평균 염소 농도")
st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)

# ------------------------
# 5. 사용자 진단 시스템
# ------------------------
st.header("④ 내 수질은 어디쯤? (백분위 진단 + 피드백)")
st.markdown("#### 💡 아래 항목을 입력해 수질 상태를 분석해보세요.")

turb = st.slider("탁도 (NTU)", 0.0, 10.0, 4.0)
chl = st.slider("잔류 염소 (mg/L)", 0.0, 1.0, 0.3)
coli = st.slider("대장균 (MPN/100mL)", 0, 10, 1)
fluor = st.slider("불소 농도 (mg/L)", 0.0, 2.0, 1.0)

def get_percentile(colname, value):
    col = df[colname].dropna()
    return percentileofscore(col, value, kind="mean")

with st.expander("📊 NYC 전체 분포 내 백분위 위치"):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("탁도 백분위", f"{get_percentile('Turbidity', turb):.1f} %")
        st.metric("염소 백분위", f"{get_percentile('Chlorine', chl):.1f} %")
    with col2:
        st.metric("대장균 백분위", f"{get_percentile('Coliform', coli):.1f} %")
        st.metric("불소 백분위", f"{get_percentile('Fluoride', fluor):.1f} %")

# ------------------------
# 6. 정책 피드백 제시
# ------------------------
st.subheader("📋 진단 결과 및 오염원 대응 방안")

issues = []
if turb > 5:
    issues.append("🟠 **탁도 초과**: 미생물 보호 가능성 증가 → 필터링 시스템 점검 필요")
if chl < 0.2:
    issues.append("🔴 **염소 부족**: 소독력 저하 우려 → 말단 염소 유지 보강 필요")
if coli > 0:
    issues.append("🔴 **대장균 검출**: 병원균 감염 가능성 ↑ → 재검사 및 유입경로 차단 필요")
if fluor > 1.5:
    issues.append("🟡 **불소 과다**: 만성노출 위험 → 주입량 조정 권고")

if issues:
    for i in issues:
        st.warning(i)
else:
    st.success("✅ 모든 수질 항목이 안전 기준 내에 있습니다.")

# ------------------------
# 7. 출처 및 SDG 연계
# ------------------------
st.markdown("---")
st.caption("""
🔎 본 시스템은 WHO 수질 기준 및 SDG 6 (깨끗한 물과 위생) 목표에 따라 NYC 수돗물 데이터를 분석하여,
도시 수질 관리의 데이터 기반 진단 체계를 구현합니다.
""")
