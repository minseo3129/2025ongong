
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
    df["Month"] = df["Date"].dt.month.astype(str)
    df["Month_Num"] = df["Date"].dt.month
    return df

# -------------------------------
# 2. Streamlit 설정
# -------------------------------
df = load_data()
st.set_page_config(layout="wide")
st.title("💧 NYC 수돗물 수질 진단 및 운영 위험 분석 통합 대시보드")
st.caption("📌 SDG 6 연계: 수질 초과율 분석 + 사용자 진단 기능 통합 제공")

# -------------------------------
# 3. Sample Class × 월별 기준 초과율 분석
# -------------------------------
st.header("① Sample Class × 월별 수질 기준 초과율 분석")
st.sidebar.header("⚙️ 초과율 분석 항목 선택")
indicator = st.sidebar.selectbox("수질 항목", ["Turbidity", "Chlorine", "Coliform", "Ecoli"])

# 초과 기준 설정
if indicator == "Turbidity":
    threshold = 5
    operator = ">"
    condition = df[indicator] > threshold
elif indicator == "Chlorine":
    threshold = 0.2
    operator = "<"
    condition = df[indicator] < threshold
else:
    threshold = 0
    operator = ">"
    condition = df[indicator] > threshold

df["Exceed"] = condition
summary = df.groupby(["Sample_Class", "Month_Num"])["Exceed"].mean().reset_index()
summary["Exceed"] = summary["Exceed"] * 100  # %
summary["Month_Num"] = summary["Month_Num"].astype(int)

# 시각화
fig = px.line(
    summary,
    x="Month_Num", y="Exceed", color="Sample_Class", markers=True,
    labels={"Exceed": "초과율 (%)", "Month_Num": "월"},
    title=f"{indicator} {operator} {threshold} 기준 초과율 (Sample Class × 월)"
)
fig.update_layout(xaxis=dict(tickmode="linear"))
st.plotly_chart(fig, use_container_width=True)

st.markdown("#### 🧠 정책 해석")
st.markdown(f"- 선택 항목 **{indicator} {operator} {threshold}** 기준 초과율을 시각화했습니다.")
st.markdown("- 특정 Sample Class에서 특정 월에 반복적으로 초과되는 경향이 보인다면, 해당 운영 구간의 보완이 필요함을 시사합니다.")

# -------------------------------
# 4. Sample Class별 수질 통계 비교
# -------------------------------
st.header("② Sample Class별 수질 평균 및 분산 비교")
with st.expander("🔍 평균 및 표준편차 비교"):
    summary_stats = df.groupby("Sample_Class")[["Turbidity", "Chlorine", "Coliform", "Ecoli"]].agg(["mean", "std"])
    st.dataframe(summary_stats)

# -------------------------------
# 5. 월별 평균 수질 변화 추이
# -------------------------------
st.header("③ 월별 평균 수질 변화 추이")
monthly = df.groupby("Month_Num")[["Turbidity", "Chlorine"]].mean().reset_index()

fig1 = px.line(monthly, x="Month_Num", y="Turbidity", title="📈 월별 평균 탁도")
fig2 = px.line(monthly, x="Month_Num", y="Chlorine", title="📈 월별 평균 잔류 염소")

st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)






# -------------------------------
# 8. MCMC 없이 예측 스타일 초과율 추정
# -------------------------------
import scipy.stats as stats

st.header("⑤ MCMC 없이 월별 초과율 예측 (Beta 분포 기반)")

# 월별 초과 횟수 및 총 샘플 수
monthly_stat = df.groupby("Month_Num")["Exceed"].agg(
    Exceed_Count="sum",
    Total_Sample="count"
).reset_index()

# 베타 사전: alpha=2, beta=2 (균형잡힌 비편향)
alpha_prior, beta_prior = 2, 2

# 사후 분포 계산 (Beta + Binomial → Beta 사후)
monthly_stat["alpha_post"] = alpha_prior + monthly_stat["Exceed_Count"]
monthly_stat["beta_post"] = beta_prior + (monthly_stat["Total_Sample"] - monthly_stat["Exceed_Count"])

# 평균 및 95% 신뢰 구간 계산
monthly_stat["Posterior_Mean"] = monthly_stat["alpha_post"] / (monthly_stat["alpha_post"] + monthly_stat["beta_post"])
monthly_stat["Lower_CI"] = stats.beta.ppf(0.025, monthly_stat["alpha_post"], monthly_stat["beta_post"])
monthly_stat["Upper_CI"] = stats.beta.ppf(0.975, monthly_stat["alpha_post"], monthly_stat["beta_post"])

# 시각화
fig = px.line(monthly_stat, x="Month_Num", y="Posterior_Mean", markers=True,
              title="📈 월별 수질 초과율 추정 (Beta 분포 기반)",
              labels={"Month_Num": "월", "Posterior_Mean": "예측 초과율"})

fig.add_traces([
    px.scatter(monthly_stat, x="Month_Num", y="Lower_CI").data[0],
    px.scatter(monthly_stat, x="Month_Num", y="Upper_CI").data[0]
])
fig.update_traces(name="95% 신뢰 하한", selector=dict(name="Lower_CI"))
fig.update_traces(name="95% 신뢰 상한", selector=dict(name="Upper_CI"))
fig.update_layout(legend=dict(title="범례"))

st.plotly_chart(fig, use_container_width=True)

with st.expander("📘 해석"):
    st.markdown("""
    - 각 월의 초과율은 **Beta(2+성공, 2+실패)** 분포 기반으로 추정됩니다.
    - 이는 MCMC 없이도 **불확실성을 반영한 예측**이 가능합니다.
    - 특정 월에 **신뢰구간이 넓거나 평균이 높은 경우**, 추가 관찰 또는 선제적 대응이 필요함을 의미합니다.
    """)









# -------------------------------
# 6. 사용자 입력 기반 수질 진단
# -------------------------------
st.header("④ 사용자 수질 입력 기반 진단 시스템")
turb = st.slider("탁도 (NTU)", 0.0, 10.0, 4.0)
chl = st.slider("잔류 염소 (mg/L)", 0.0, 1.0, 0.3)
coli = st.slider("대장균 (MPN/100mL)", 0, 10, 1)
fluor = st.slider("불소 농도 (mg/L)", 0.0, 2.0, 1.0)

def get_percentile(colname, value):
    return percentileofscore(df[colname].dropna(), value, kind="mean")

with st.expander("📊 NYC 전체 수질 분포 내 사용자 위치"):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("탁도 백분위", f"{get_percentile('Turbidity', turb):.1f} %")
        st.metric("염소 백분위", f"{get_percentile('Chlorine', chl):.1f} %")
    with col2:
        st.metric("대장균 백분위", f"{get_percentile('Coliform', coli):.1f} %")
        st.metric("불소 백분위", f"{get_percentile('Fluoride', fluor):.1f} %")

st.subheader("📋 진단 결과 및 대응 제안")

feedback = []
if turb > 5:
    feedback.append("🟠 **탁도 초과**: 여과 장치 또는 관망 침전물 점검 필요")
if chl < 0.2:
    feedback.append("🔴 **염소 부족**: 소독력 저하 위험, 말단 보강 필요")
if coli > 0:
    feedback.append("🔴 **대장균 검출**: 오염 가능성, 추가 검사 및 원인 조사 필요")
if fluor > 1.5:
    feedback.append("🟡 **불소 과다**: 장기 노출 위험 가능성, 공급량 조정 고려")

if feedback:
    for f in feedback:
        st.warning(f)
else:
    st.success("✅ 입력하신 수질은 모든 항목에서 안정 기준을 충족합니다.")

# -------------------------------
# 7. 마무리
# -------------------------------
st.markdown("---")
st.caption("📘 본 시스템은 NYC 수돗물 데이터를 기반으로 수질 초과율 및 사용자 상태를 진단하며, SDG 6(깨끗한 물과 위생) 달성을 지원합니다.")


























