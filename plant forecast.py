# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 내장 데이터 로드
df = pd.read_csv("plant_growth_data.csv")
features = df.columns[:-1]

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(layout="wide")
st.title("🌱 스마트팜 생장 데이터 분석 및 조건 기반 작물 재배 매뉴얼")

# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 내장 데이터 로드
df = pd.read_csv("plant_growth_data.csv")
features = df.columns[:-1]


# 📊 1. 박스플롯
st.subheader("📊 1. 생장 성공/실패군의 주요 변수 분포 (Boxplot)")
for feature in ["Sunlight_Hours", "Temperature", "Humidity"]:
    fig = px.box(df, x="Failure", y=feature, color="Failure",
                 title=f"{feature}에 따른 생장 성공/실패 분포",
                 labels={"Failure": "성공(0)/실패(1)"})
    st.plotly_chart(fig, use_container_width=True)

# 📊 2. 조건 조합별 실패율 히트맵
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

# 📊 5. 연관규칙 기반 위험 조건 탐색
st.subheader("📊 5. 연관규칙 기반 위험 조합 탐색")
rule_df = df.copy()
rule_df = pd.get_dummies(rule_df[["Soil_Type", "Water_Frequency", "Fertilizer_Type"]])
rule_df["Failure"] = df["Failure"]
frequent_items = apriori(rule_df, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1)
risk_rules = rules[rules['consequents'].astype(str).str.contains('Failure')]
st.write("위험 조합 규칙:")
st.dataframe(risk_rules[['antecedents', 'support', 'confidence', 'lift']])

# 📊 6. 사용자 입력 기반 실패율 예측
st.subheader("📊 6. 사용자 조건 기반 실패 리스크 예측")
soil = st.selectbox("토양 유형", df["Soil_Type"].unique())
water = st.selectbox("물 주기", df["Water_Frequency"].unique())
fert = st.selectbox("비료 유형", df["Fertilizer_Type"].unique())
sun = st.slider("햇빛 노출 시간", float(df["Sunlight_Hours"].min()), float(df["Sunlight_Hours"].max()), 6.0)
temp = st.slider("온도", float(df["Temperature"].min()), float(df["Temperature"].max()), 25.0)
hum = st.slider("습도", float(df["Humidity"].min()), float(df["Humidity"].max()), 60.0)

input_data = pd.DataFrame([[soil, water, fert, sun, temp, hum]],
                          columns=["Soil_Type", "Water_Frequency", "Fertilizer_Type", "Sunlight_Hours", "Temperature", "Humidity"])
all_data = pd.concat([df, input_data], ignore_index=True)
all_encoded = pd.get_dummies(all_data.drop("Failure", axis=1, errors='ignore'))
input_vector = all_encoded.iloc[[-1]]
data_vector = all_encoded.iloc[:-1]
labels = df["Failure"]

model = KNeighborsClassifier(n_neighbors=5)
model.fit(data_vector, labels)
pred_prob = model.predict_proba(input_vector)[0][1]

st.markdown(f"### 🔍 예측된 실패 확률: **{round(pred_prob * 100, 1)}%**")
if pred_prob >= 0.6:
    st.error("⚠️ 높은 실패 위험. 차광, 냉방, 환기 필요")
elif pred_prob >= 0.3:
    st.warning("⚠️ 중간 위험. 조건 조정 고려")
else:
    st.success("✅ 양호한 조건")

st.success("✅ 전체 분석 및 사용자 예측 완료")
