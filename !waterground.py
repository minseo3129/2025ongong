import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 0. 데이터 불러오기 및 정리
@st.cache_data
def load_data():
    df = pd.read_csv("Drinking_Water_Quality_Distribution_Monitoring_Data.csv")
    df.columns = df.columns.str.replace(" ", "_").str.replace("(", "").str.replace(")", "")
    df = df.rename(columns={
        "Residual_Free_Chlorine_mg/L": "Chlorine",
        "Turbidity_NTU": "Turbidity",
        "Coliform_Quanti-Tray_MPN_/100mL": "Coliform",
        "E.coliQuanti-Tray_MPN/100mL": "Ecoli",
        "Fluoride_mg/L": "Fluoride"
    })
    df = df[["Sample_Site", "Sample_Date", "Chlorine", "Turbidity", "Coliform", "Ecoli", "Fluoride"]]
    for col in ["Chlorine", "Turbidity", "Coliform", "Ecoli", "Fluoride"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Sample_Date"] = pd.to_datetime(df["Sample_Date"], errors="coerce")
    df = df.dropna()
    return df

df = load_data()

st.set_page_config(page_title="NYC 수질 분석", layout="wide")
st.title("🌆 NYC 수돗물 수질 영향 요인 분석 대시보드")

# 1단계: Sample Site별 요약
st.header("1단계. 📊 Sample Site별 수질 요약")
numeric_cols = ["Turbidity", "Chlorine", "Coliform", "Ecoli", "Fluoride"]
summary = df.groupby("Sample_Site")[numeric_cols].agg(["mean", "max"])
st.dataframe(summary)

# 2단계: 시계열 변화 보기
st.header("2단계. 🕒 탁도 시계열 변화")
selected_site = st.selectbox("Sample Site 선택", df["Sample_Site"].unique())
site_df = df[df["Sample_Site"] == selected_site]
fig_ts = px.line(site_df, x="Sample_Date", y="Turbidity", title=f"{selected_site} 탁도 변화")
st.plotly_chart(fig_ts, use_container_width=True)

# 3단계: 상관관계 분석
st.header("3단계. 🔗 수질 항목 간 상관 분석")
st.dataframe(df[numeric_cols].corr())

# 4단계: 회귀분석 (탁도 예측)
st.header("4단계. 📈 회귀분석 (탁도 예측)")
X = df[["Chlorine", "Coliform", "Fluoride"]]
y = df["Turbidity"]
model = LinearRegression().fit(X, y)
st.write("회귀계수:", dict(zip(X.columns, model.coef_)))
st.write("절편:", model.intercept_)
st.write("R² score:", model.score(X, y))

# 5단계: PCA 요인분석
st.header("5단계. 🧠 요인분석(PCA)")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numeric_cols])
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)
df["PC1"], df["PC2"] = pca_components[:, 0], pca_components[:, 1]
st.write("설명된 분산 비율:", pca.explained_variance_ratio_)

# 6단계: LDC 유사 위험지수 생성
st.header("6단계. 📉 기준 초과 기반 위험지수 (LDC 유사)")
def calc_risk(row):
    score = 0
    if row["Turbidity"] > 5: score += 1
    if row["Chlorine"] < 0.2: score += 2
    if row["Coliform"] > 0: score += 2
    if row["Ecoli"] > 0: score += 3
    return score

df["Risk_Index"] = df.apply(calc_risk, axis=1)
st.dataframe(df[["Sample_Site", "Sample_Date", "Risk_Index"]].sort_values("Risk_Index", ascending=False).head(10))

# 7단계: PCA 시각화
st.header("7단계. 🗺️ 위험군 군집 시각화 (PCA)")
fig_pca = px.scatter(df, x="PC1", y="PC2", color="Risk_Index", hover_data=["Sample_Site"],
                     title="PCA 기반 수질 위험 군집")
st.plotly_chart(fig_pca, use_container_width=True)

# 8단계: 사용자 입력 기반 위험도 평가
st.header("8단계. 🧪 사용자 입력 기반 위험도 평가")
turb = st.slider("탁도 (NTU)", 0.0, 10.0, 4.0)
chl = st.slider("잔류 염소 (mg/L)", 0.0, 1.0, 0.3)
col = st.slider("Coliform (MPN/100mL)", 0, 10, 0)
eco = st.slider("E.coli (MPN/100mL)", 0, 10, 0)
flu = st.slider("불소 농도 (mg/L)", 0.0, 2.0, 1.0)

if st.button("📋 위험도 평가"):
    score = 0
    if turb > 5: score += 1
    if chl < 0.2: score += 2
    if col > 0: score += 2
    if eco > 0: score += 3
    if flu > 1.5: score += 1
    st.subheader(f"위험 점수: {score}점")
    if score >= 6:
        st.error("🚨 고위험 상태")
    elif score >= 3:
        st.warning("⚠️ 주의 필요")
    else:
        st.success("✅ 안전 상태")

# 출처
st.caption("📖 참고: 정용훈 외 (2025), 『유역모델을 이용한 섬진강댐 수질 영향 인자 분석』, 대한환경공학회지")

