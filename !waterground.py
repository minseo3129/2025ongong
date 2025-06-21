import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 데이터 로딩
@st.cache_data
def load_data():
    df = pd.read_csv("Drinking_Water_Quality_Distribution_Monitoring_Data.csv")
    df = df.rename(columns=lambda x: x.strip().replace(' ', '_'))
    df = df.rename(columns={
        'Residual_Free_Chlorine_(mg/L)': 'Chlorine',
        'Turbidity_(NTU)': 'Turbidity',
        'Coliform_(Quanti-Tray)_(MPN_/100mL)': 'Coliform',
        'E.coli(Quanti-Tray)_(MPN/100mL)': 'Ecoli',
        'Fluoride_(mg/L)': 'Fluoride'
    })
    df = df[['Sample_Site', 'Sample_Date', 'Chlorine', 'Turbidity', 'Coliform', 'Ecoli', 'Fluoride']].dropna()
    df["Date"] = pd.to_datetime(df["Sample_Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    return df

df = load_data()

st.set_page_config(page_title="NYC 수질 분석", layout="wide")
st.title("🌆 NYC 수돗물 수질 영향 요인 분석 대시보드")

# 1단계: 지점별 요약
st.header("1단계. 📊 Sample Site별 수질 지표 요약")
st.dataframe(df.groupby("Sample_Site")[["Turbidity", "Chlorine", "Coliform", "Ecoli", "Fluoride"]].agg(['mean', 'max']))

# 2단계: 시간대별 변화
st.header("2단계. 🕒 시계열 수질 변화")
site_selected = st.selectbox("지점 선택", df["Sample_Site"].unique())
df_site = df[df["Sample_Site"] == site_selected]
fig = px.line(df_site, x="Date", y="Turbidity", title=f"{site_selected} - 시간에 따른 탁도 변화")
st.plotly_chart(fig, use_container_width=True)

# 3단계: 상관분석
st.header("3단계. 🔗 상관관계 분석")
corr = df[["Turbidity", "Chlorine", "Coliform", "Ecoli", "Fluoride"]].corr()
st.dataframe(corr)

# 4단계: 회귀분석
st.header("4단계. 📈 탁도 예측 회귀분석")
X = df[["Chlorine", "Coliform", "Fluoride"]]
y = df["Turbidity"]
model = LinearRegression().fit(X, y)
st.write("회귀계수:", dict(zip(X.columns, model.coef_)))
st.write("R² score:", model.score(X, y))

# 5단계: PCA 요인 축소
st.header("5단계. 🧠 요인분석 (PCA)")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[["Turbidity", "Chlorine", "Coliform", "Ecoli", "Fluoride"]])
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
df["PC1"], df["PC2"] = components[:, 0], components[:, 1]
st.write("설명 분산 비율:", pca.explained_variance_ratio_)

# 6단계: LDC 유사 지표
st.header("6단계. 📉 기준 초과율 기반 위험지수 (LDC 유사)")
def calculate_risk(row):
    score = 0
    if row["Turbidity"] > 5: score += 1
    if row["Chlorine"] < 0.2: score += 2
    if row["Coliform"] > 0: score += 2
    if row["Ecoli"] > 0: score += 3
    return score

df["Risk_Index"] = df.apply(calculate_risk, axis=1)
st.dataframe(df[["Sample_Site", "Date", "Turbidity", "Chlorine", "Coliform", "Ecoli", "Risk_Index"]].sort_values("Risk_Index", ascending=False).head(10))

# 7단계: 위험 지도 시각화
st.header("7단계. 🔥 고위험 지역 시각화 (PC1 기반)")
fig2 = px.scatter(df, x="PC1", y="PC2", color="Risk_Index", hover_name="Sample_Site",
                  title="PCA 기반 위험도 군집화")
st.plotly_chart(fig2, use_container_width=True)

# 8단계: 사용자 입력 기반 수질 위험도 진단
st.header("8단계. 🧪 사용자 입력 기반 수질 진단")
turb = st.slider("탁도 (NTU)", 0.0, 10.0, 4.0)
chl = st.slider("잔류 염소 (mg/L)", 0.0, 1.0, 0.3)
col = st.slider("Coliform (MPN/100mL)", 0, 10, 0)
eco = st.slider("E.coli (MPN/100mL)", 0, 10, 0)
flu = st.slider("불소 농도 (mg/L)", 0.0, 2.0, 1.0)

if st.button("📋 위험도 평가 실행"):
    score = 0
    if turb > 5: score += 1
    if chl < 0.2: score += 2
    if col > 0: score += 2
    if eco > 0: score += 3
    if flu > 1.5: score += 1

    st.subheader(f"총 위험점수: {score}점")
    if score >= 6:
        st.error("🚨 고위험 지역 수준입니다.")
    elif score >= 3:
        st.warning("⚠️ 주의가 필요한 수질 상태입니다.")
    else:
        st.success("✅ 비교적 안전한 수질 상태입니다.")

st.caption("📖 참고: 정용훈 외 (2025), 『유역모델을 이용한 섬진강댐 수질 영향 인자 분석』")

sns.barplot(data=importance_df, x="중요도", y="항목", ax=ax, palette="crest")
st.pyplot(fig)
