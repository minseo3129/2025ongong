import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("cities_air_quality_water_pollution.csv")
    df.columns = df.columns.str.replace('"', '').str.strip()
    df = df.rename(columns={"AirQuality": "Air_Quality", "WaterPollution": "Water_Pollution"})
    df["Air_Quality"] = pd.to_numeric(df["Air_Quality"], errors='coerce')
    df["Water_Pollution"] = pd.to_numeric(df["Water_Pollution"], errors='coerce')
    return df.dropna()

df = load_data()

# Streamlit UI
st.set_page_config(page_title="NYC 수질 대시보드", layout="wide")
st.title("🌆 NYC 수질 영향 요인 분석 및 고위험 지역 제안")

# 1단계: 요약 통계
st.header("1단계. 📊 도시별 수질 통계")
st.dataframe(df[['Air_Quality', 'Water_Pollution']].describe())

# 2단계: 시간 또는 지역 기반 비교는 시간 데이터 없으므로 대체 생략 가능

# 3단계: 상관분석
st.header("3단계. 🔗 상관분석")
corr = df[['Air_Quality', 'Water_Pollution']].corr()
st.dataframe(corr)

# 4단계: 회귀분석 (Air_Quality로 Water_Pollution 예측)
st.header("4단계. 📈 회귀분석 (수질 오염도 예측)")
X = df[['Air_Quality']]
y = df['Water_Pollution']
model = LinearRegression().fit(X, y)
st.write("회귀계수:", model.coef_[0])
st.write("절편:", model.intercept_)
st.write("R²:", model.score(X, y))

# 5단계: PCA
st.header("5단계. 🧠 요인분석(PCA)")
scaler = StandardScaler()
features = df[['Air_Quality', 'Water_Pollution']]
features_scaled = scaler.fit_transform(features)
pca = PCA(n_components=2)
components = pca.fit_transform(features_scaled)
df['PC1'] = components[:, 0]
df['PC2'] = components[:, 1]
st.write("PCA 설명 분산 비율:", pca.explained_variance_ratio_)

# 6단계: 초과빈도 기반 위험 지표 생성
st.header("6단계. 📉 위험지수 계산 (LDC 유사)")
df['Risk_Index'] = (100 - df['Air_Quality']) + df['Water_Pollution']
df['Risk_Level'] = pd.cut(df['Risk_Index'], bins=[0, 80, 120, 200], labels=["Low", "Moderate", "High"])
st.dataframe(df[['City', 'Country', 'Risk_Index', 'Risk_Level']].sort_values('Risk_Index', ascending=False).head(10))

# 7단계: 지도 시각화 (위치 정보가 없는 경우 대체 바차트)
st.header("7단계. 🗺️ 고위험 도시 시각화")
top_risk = df.sort_values('Risk_Index', ascending=False).head(15)
fig_map = px.bar(top_risk, x='City', y='Risk_Index', color='Risk_Level', title='상위 위험 도시')
st.plotly_chart(fig_map, use_container_width=True)

# 8단계: 사용자 입력 기반 위험도 진단
st.header("8단계. 🧪 사용자 도시 수질 입력 진단")
air_q = st.slider("공기질 (0: 나쁨 ~ 100: 좋음)", 0.0, 100.0, 60.0)
water_p = st.slider("수질 오염도 (0: 없음 ~ 100: 심각)", 0.0, 100.0, 50.0)
user_risk = (100 - air_q) + water_p

if st.button("📋 도시 위험도 평가"):
    st.markdown(f"**위험지수: {user_risk:.1f}**")
    if user_risk >= 120:
        st.error("🚨 고위험 지역")
    elif user_risk >= 80:
        st.warning("⚠️ 주의가 필요한 지역")
    else:
        st.success("✅ 안전 수준")

# 부록: 참고문헌 링크
st.markdown("---")
st.caption("참고: 정용훈 외 (2025), 『유역모델을 이용한 섬진강댐 수질 영향 인자 분석 및 오염부하 특성 평가』, 대한환경공학회지.")

fig, ax = plt.subplots()
sns.barplot(data=importance_df, x="중요도", y="항목", ax=ax, palette="crest")
st.pyplot(fig)
