import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import percentileofscore

# -------------------------------
# 1. 데이터 로딩 및 정제 함수
# -------------------------------
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("Drinking_Water_Quality_Distribution_Monitoring_Data.csv")

    # 컬럼명 정제
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")

    # 수질 관련 주요 컬럼 리네이밍
    df = df.rename(columns={
        "Residual_Free_Chlorine_mg/L": "Chlorine",
        "Turbidity_NTU": "Turbidity",
        "Coliform_Quanti-Tray_MPN_/100mL": "Coliform",
        "E.coliQuanti-Tray_MPN/100mL": "Ecoli",
        "Fluoride_mg/L": "Fluoride",
        "Sample_Date": "Date"
    })

    # 문자열 수치 변환
    def convert_text(v):
        if isinstance(v, str):
            v = v.replace("<", "").replace(">", "").replace("+", "").strip()
            if v.lower() == "nd":
                return np.nan
        try:
            return float(v)
        except:
            return np.nan

    for col in ["Turbidity", "Chlorine", "Coliform", "Ecoli", "Fluoride"]:
        if col in df.columns:
            df[col] = df[col].apply(convert_text)

    # 날짜 처리
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Sample_Class 확인
    sample_class_col = None
    for candidate in ["Sample_Class", "SampleClass", "Sample_Class_"]:
        if candidate in df.columns:
            sample_class_col = candidate
            break

    if sample_class_col is None:
        st.error("❌ 'Sample_Class' 또는 유사 컬럼이 존재하지 않습니다. 실제 컬럼명을 확인해 주세요.")
        st.write("사용 가능한 컬럼:", df.columns.tolist())
        st.stop()

    df = df.rename(columns={sample_class_col: "Sample_Class"})
    df = df.dropna(subset=["Date", "Sample_Class", "Turbidity", "Chlorine", "Coliform", "Ecoli"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    return df

df = load_and_clean_data()

# -------------------------------
# 2. Streamlit 설정
# -------------------------------
st.set_page_config(layout="wide")
st.title("💧 NYC 수돗물 수질 진단 및 분석 대시보드")
st.caption("📌 SDG 6 연계: 시민 참여형 수질 분석 · 위험 진단 · 맞춤형 피드백 제공")

# -------------------------------
# 3. 샘플 유형별 수질 안정성 비교
# -------------------------------
st.header("① 샘플 유형별 수질 평균 비교")
with st.expander("🔍 Compliance vs Operational 평균 비교"):
    summary = df.groupby("Sample_Class")[["Turbidity", "Chlorine", "Coliform", "Ecoli"]].agg(["mean", "std"])
    st.dataframe(summary)

# -------------------------------
# 4. 기준 초과 항목 비율 시각화
# -------------------------------
st.header("② 기준 초과 항목 비율 분석")
df["Turbidity_Exceed"] = df["Turbidity"] > 5
df["Chlorine_Low"] = df["Chlorine"] < 0.2
df["Coliform_Positive"] = df["Coliform"] > 0
df["Ecoli_Positive"] = df["Ecoli"] > 0

st.bar_chart(df[["Turbidity_Exceed", "Chlorine_Low", "Coliform_Positive", "Ecoli_Positive"]].mean() * 100)

# -------------------------------
# 5. 월별 수질 변화 추이 시각화
# -------------------------------
st.header("③ 월별 수질 변화 추이")
monthly = df.groupby("Month")[["Turbidity", "Chlorine"]].mean().reset_index()
fig1 = px.line(monthly, x="Month", y="Turbidity", title="📈 월별 평균 탁도")
fig2 = px.line(monthly, x="Month", y="Chlorine", title="📈 월별 평균 잔류 염소")

st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# 6. 사용자 입력 기반 진단 시스템
# -------------------------------
st.header("④ 사용자 수질 입력 기반 진단 시스템")
turb = st.slider("탁도 (NTU)", 0.0, 10.0, 4.0)
chl = st.slider("잔류 염소 (mg/L)", 0.0, 1.0, 0.3)
coli = st.slider("대장균 (MPN/100mL)", 0, 10, 1)
fluor = st.slider("불소 농도 (mg/L)", 0.0, 2.0, 1.0)

def get_percentile(colname, value):
    return percentileofscore(df[colname].dropna(), value, kind="mean")

with st.expander("📊 NYC 전체 분포 내 위치"):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("탁도 백분위", f"{get_percentile('Turbidity', turb):.1f} %")
        st.metric("염소 백분위", f"{get_percentile('Chlorine', chl):.1f} %")
    with col2:
        st.metric("대장균 백분위", f"{get_percentile('Coliform', coli):.1f} %")
        st.metric("불소 백분위", f"{get_percentile('Fluoride', fluor):.1f} %")

# -------------------------------
# 7. 위험 요인 피드백
# -------------------------------
st.subheader("📋 진단 결과 및 맞춤형 대응 제안")

feedback = []
if turb > 5:
    feedback.append("🟠 **탁도 초과**: 여과 장치 이상 가능 → 정밀 점검 필요")
if chl < 0.2:
    feedback.append("🔴 **염소 농도 부족**: 소독력 저하 우려 → 급수 말단 염소 유지 필요")
if coli > 0:
    feedback.append("🔴 **대장균 검출**: 유입경로 차단 필요 → 재검사 권장")
if fluor > 1.5:
    feedback.append("🟡 **불소 과다**: 장기 노출 위험 가능성 → 투입량 조정 필요")

if feedback:
    for item in feedback:
        st.warning(item)
else:
    st.success("✅ 입력하신 수치는 모두 안전 기준 내에 있습니다.")

# -------------------------------
# 8. 마무리 문구
# -------------------------------
st.markdown("---")
st.caption("📘 본 프로젝트는 WHO 수질 기준 및 SDG 6(깨끗한 물과 위생) 목표 달성에 기여하기 위한 데이터 기반 수질 분석 시스템입니다.")


