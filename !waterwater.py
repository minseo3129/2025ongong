import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import percentileofscore

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

df = load_data()

# -------------------------------
# 2. Streamlit 설정
# -------------------------------
st.set_page_config(layout="wide")
st.title("💧 NYC 수돗물 수질 진단 및 운영 위험 분석 통합 대시보드")
st.caption("🔍 SDG 6 기반: Sample Class × Month 분석 + 사용자 진단 기능")

# -------------------------------
# 3. 분석 항목 설정 및 기준 초과율 분석
# -------------------------------
st.header("① Sample Class × 월별 수질 기준 초과율 분석")
st.sidebar.header("⚙️ 초과율 분석 항목")
indicator = st.sidebar.selectbox("수질 항목", ["Turbidity", "Chlorine", "Coliform", "Ecoli"])

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
summary = df.groupby(["Sample_Class", "Month"])["Exceed"].mean().reset_index()
summary["Exceed"] *= 100
summary["Month"] = summary["Month"].astype(int)

fig = px.line(
    summary,
    x="Month", y="Exceed", color="Sample_Class", markers=True,
    labels={"Exceed": "초과율 (%)", "Month": "월"},
    title=f"{indicator} {operator} {threshold} 기준 초과율 - 운영 유형별 월별 변화"
)
fig.update_layout(xaxis=dict(tickmode="linear"))
st.plotly_chart(fig, use_container_width=True)

st.markdown("#### 🧠 정책 해석")
st.markdown(f"- **{indicator} {operator} {threshold}** 기준을 초과한 비율을 시각화했습니다.")
st.markdown("- 특정 Sample Class에서 특정 월에 초과율이 집중되면, 해당 운영 상태 또는 계절에 따른 수질 악화 가능성을 의미합니다.")
st.markdown("- 이는 염소 보강, 여과 유지보수, 계절별 점검 전략 수립에 활용될 수 있습니다.")

# -------------------------------
# 4. 사용자 수질 진단 시스템
# -------------------------------
st.header("② 사용자 수질 입력 기반 진단 시스템")

turb = st.slider("탁도 (NTU)", 0.0, 10.0, 4.0)
chl = st.slider("잔류 염소 (mg/L)", 0.0, 1.0, 0.3)
coli = st.slider("대장균 (MPN/100mL)", 0, 10, 1)
fluor = st.slider("불소 농도 (mg/L)", 0.0, 2.0, 1.0)

def get_percentile(col, val):
    return percentileofscore(df[col].dropna(), val, kind="mean")

with st.expander("📊 NYC 전체 수질 분포 중 입력값 위치"):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("탁도 백분위", f"{get_percentile('Turbidity', turb):.1f}%")
        st.metric("염소 백분위", f"{get_percentile('Chlorine', chl):.1f}%")
    with col2:
        st.metric("대장균 백분위", f"{get_percentile('Coliform', coli):.1f}%")
        st.metric("불소 백분위", f"{get_percentile('Fluoride', fluor):.1f}%")

st.subheader("📋 진단 결과 및 정책적 대응 제안")
feedback = []
if turb > 5:
    feedback.append("🟠 **탁도 초과**: 여과 불량 또는 배관 침전물 가능 → 필터/관망 점검 권장")
if chl < 0.2:
    feedback.append("🔴 **염소 농도 부족**: 소독력 저하 → 말단 염소 잔류량 관리 필요")
if coli > 0:
    feedback.append("🔴 **대장균 검출**: 병원성 가능성 ↑ → 정밀 검사 및 오염원 추적 필요")
if fluor > 1.5:
    feedback.append("🟡 **불소 과다**: 장기 노출 위험 가능 → 불소 주입량 조정 권장")

if feedback:
    for line in feedback:
        st.warning(line)
else:
    st.success("✅ 입력 수치는 WHO 기준에 따라 모두 안정 범위 내에 있습니다.")

# -------------------------------
# 5. 마무리
# -------------------------------
st.markdown("---")
st.caption("📘 본 시스템은 실제 NYC 수질 데이터를 기반으로 수질 위험 모니터링과 맞춤형 대응 방안을 제공합니다.")
