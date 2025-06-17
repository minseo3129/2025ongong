import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🌿 스마트팜 생장 실패 분석 대시보드")

st.sidebar.header("📂 데이터 불러오기")
df = pd.read_csv("https://raw.githubusercontent.com/사용자아이디/저장소명/main/plant_growth_data.csv")  # 경로 수정 필요

# 전처리: 실패율 컬럼 생성
df["Failure"] = 1 - df["Growth_Milestone"]

# 📊 1. 성공 vs 실패군 박스플롯
st.subheader("📊 1. 생장 성공/실패군의 주요 변수 분포 (Boxplot)")
for feature in ["Sunlight_Hours", "Temperature", "Humidity"]:
    fig = px.box(df, x="Failure", y=feature, color="Failure",
                 title=f"{feature}에 따른 생장 성공/실패 분포",
                 labels={"Failure": "성공(0)/실패(1)"})
    st.plotly_chart(fig, use_container_width=True)

# 📊 2. 조건 조합별 생장 실패율 히트맵
st.subheader("📊 2. 조건 조합별 생장 실패율 히트맵")
combo_df = df.groupby(["Soil_Type", "Water_Frequency", "Fertilizer_Type"])["Failure"].mean().reset_index()
pivot_df = combo_df.pivot_table(index="Soil_Type", columns=["Water_Frequency", "Fertilizer_Type"], values="Failure")
st.dataframe((pivot_df * 100).round(1), use_container_width=True)

# 📊 3. 연속형 변수 임계값 분석
st.subheader("📊 3. 연속형 변수별 임계값 구간에 따른 생장 실패율")
for feature, bins in [("Sunlight_Hours", 6), ("Temperature", 6), ("Humidity", 6)]:
    df[f"{feature}_bin"] = pd.cut(df[feature], bins)
    bin_df = df.groupby(f"{feature}_bin")["Failure"].mean().reset_index()
    fig = px.bar(bin_df, x=f"{feature}_bin", y="Failure", title=f"{feature} 구간별 생장 실패율",
                 labels={"Failure": "실패율"})
    st.plotly_chart(fig, use_container_width=True)

# 📊 4. 온도 & 습도 상호작용 분석
st.subheader("📊 4. 변수 간 상호작용 분석: 온도 & 습도 조합별 실패율")
df["Temp_bin"] = pd.cut(df["Temperature"], bins=[0, 20, 25, 30, 35, 40])
df["Humidity_bin"] = pd.cut(df["Humidity"], bins=[0, 40, 50, 60, 70, 100])
cross_df = df.groupby(["Temp_bin", "Humidity_bin"])["Failure"].mean().reset_index()
fig = px.density_heatmap(cross_df, x="Temp_bin", y="Humidity_bin", z="Failure",
                         color_continuous_scale="Reds",
                         title="온도 & 습도 조합별 생장 실패율")
st.plotly_chart(fig, use_container_width=True)

st.success("✅ 분석 완료. 위 시각화를 기반으로 리스크 기반 작물 재배 전략 수립 가능")
