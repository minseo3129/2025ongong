import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.neighbors import KNeighborsClassifier

# ✅ 가장 먼저 설정해야 오류 없음
st.set_page_config(layout="wide")
st.title("🌱 식물 생장 분포 분석을 통한 스마트팜의 리스크 기반 작물관리 전략 : 30129 김민서")

# 📁 데이터 불러오기
df = pd.read_csv("plant_growth_data.csv")
df["Failure"] = 1 - df["Growth_Milestone"]

# 🗺️ 변수 이름 매핑
name_map = {
    "Sunlight_Hours": "햇빛 노출 시간",
    "Temperature": "온도",
    "Humidity": "습도",
    "Failure": "실패율",
    "Temp_bin": "온도 구간",
    "Humidity_bin": "습도 구간"
}

# ✅ 시각화에 필요한 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 1. 박스플롯
st.subheader("1. 생장 성공/실패군의 주요 변수 분포")
for feature in ["Sunlight_Hours", "Temperature", "Humidity"]:
    fig = px.box(df, x="Failure", y=feature, color="Failure",
                 title=f"{name_map[feature]}에 따른 생장 성공/실패 분포",
                 labels={"Failure": "성공(0)/실패(1)", feature: name_map[feature]})
    st.plotly_chart(fig, use_container_width=True)
    











# 📊 2. 조건 조합별 생장 실패율 히트맵
st.subheader("2. 조건 조합별 생장 실패율 히트맵")
combo_df = df.groupby(["Soil_Type", "Water_Frequency", "Fertilizer_Type"])["Failure"].mean().reset_index()
pivot_df = combo_df.pivot_table(index="Soil_Type", columns=["Water_Frequency", "Fertilizer_Type"], values="Failure")
st.dataframe((pivot_df * 100).round(1), use_container_width=True)

# 3. 연속형 변수별 임계 구간 분석
st.subheader("3. 연속형 변수별 임계 구간에 따른 생장 실패율")
bin_settings = {
    "Sunlight_Hours": [4, 5, 6, 7, 8, 9, 10, 11, 12],
    "Temperature": [15, 20, 22, 25, 28, 30, 32, 35],
    "Humidity": [30, 40, 50, 60, 70, 80, 90]
}
for var in bin_settings:
    df[f"{var}_bin"] = pd.cut(df[var], bins=bin_settings[var])
    grouped = df.groupby(f"{var}_bin")["Failure"].mean().reset_index()
    grouped[f"{var}_bin"] = grouped[f"{var}_bin"].astype(str)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=grouped, x=f"{var}_bin", y="Failure", color="skyblue", ax=ax)
    ax.set_title(f"{name_map[var]} 구간별 생장 실패율", fontsize=14)
    ax.set_ylabel("실패율")
    ax.set_xlabel(f"{name_map[var]} 구간")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# 3. 임계값별 실패율 분석
st.subheader("3. 연속형 변수별 임계값 구간에 따른 생장 실패율")
for feature, bins in [("Sunlight_Hours", 6), ("Temperature", 6), ("Humidity", 6)]:
    df[f"{feature}_bin"] = pd.cut(df[feature], bins)
    bin_df = df.groupby(f"{feature}_bin")["Failure"].mean().reset_index()
    bin_df[f"{feature}_bin"] = bin_df[f"{feature}_bin"].astype(str)
    fig = px.bar(bin_df, x=f"{feature}_bin", y="Failure",
                 title=f"{name_map[feature]} 구간별 생장 실패율",
                 labels={"Failure": "실패율", f"{feature}_bin": f"{name_map[feature]} 구간"})
    st.plotly_chart(fig, use_container_width=True)

from sklearn.metrics import mutual_info_score

st.subheader("3. 연속형 변수별 임계값 분석")

for feature in ["Sunlight_Hours", "Temperature", "Humidity"]:
    st.markdown(f"#### 📈 {name_map[feature]} 기준 임계값 분석")

    best_threshold = None
    max_diff = 0
    best_group_info = None

    # 가능한 임계값 후보를 순차적으로 시도
    for threshold in np.linspace(df[feature].min(), df[feature].max(), 30):
        group_low = df[df[feature] <= threshold]["Failure"]
        group_high = df[df[feature] > threshold]["Failure"]

        # 그룹이 너무 작으면 건너뜀
        if len(group_low) < 10 or len(group_high) < 10:
            continue

        diff = abs(group_low.mean() - group_high.mean())

        if diff > max_diff:
            max_diff = diff
            best_threshold = threshold
            best_group_info = (group_low.mean(), group_high.mean())

    if best_threshold is not None:
        st.markdown(f"- 🔍 최적 임계값: **{best_threshold:.2f}**")
        st.markdown(f"- 하위 그룹 실패율: `{best_group_info[0]:.2f}`")
        st.markdown(f"- 상위 그룹 실패율: `{best_group_info[1]:.2f}`")
        st.markdown(f"- 실패율 차이: `{max_diff:.2f}`")

        # 시각화
        df["임계기준"] = np.where(df[feature] <= best_threshold, f"{name_map[feature]} ↓", f"{name_map[feature]} ↑")
        fig = px.box(df, x="임계기준", y="Failure", color="임계기준",
                     title=f"{name_map[feature]} 임계값({best_threshold:.2f})에 따른 실패율 분포",
                     labels={"Failure": "실패율"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"'{name_map[feature]}'에 대해 유의미한 임계값을 찾지 못했습니다~")

# 📊 4. 변수 간 상호작용 분석: 온도 & 습도 조합별 실패율
st.subheader("4. 변수 간 상호작용 분석: 온도 & 습도 조합별 실패율")
df["Temp_bin"] = pd.cut(df["Temperature"], bins=[0, 20, 25, 30, 35, 40])
df["Humidity_bin"] = pd.cut(df["Humidity"], bins=[0, 40, 50, 60, 70, 100])
cross_df = df.groupby(["Temp_bin", "Humidity_bin"])["Failure"].mean().reset_index()
cross_df["Temp_bin"] = cross_df["Temp_bin"].astype(str)
cross_df["Humidity_bin"] = cross_df["Humidity_bin"].astype(str)
fig = px.density_heatmap(cross_df,
                         x="Temp_bin", y="Humidity_bin", z="Failure",
                         color_continuous_scale="Reds",
                         title="온도 & 습도 조합별 생장 실패율",
                         labels={"Temp_bin": "온도 구간", "Humidity_bin": "습도 구간", "Failure": "실패율"})
st.plotly_chart(fig, use_container_width=True)

# 📊 5. 연관규칙 기반 위험 조건 탐색
st.subheader("5. 연관규칙 기반 위험 조합 탐색")
rule_df = pd.get_dummies(df[["Soil_Type", "Water_Frequency", "Fertilizer_Type"]])
rule_df["Failure"] = df["Failure"]
frequent_items = apriori(rule_df, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1)
risk_rules = rules[rules['consequents'].astype(str).str.contains('Failure')]
st.dataframe(risk_rules[['antecedents', 'support', 'confidence', 'lift']].rename(columns={
    'antecedents': '조건 조합', 'support': '지지도', 'confidence': '신뢰도', 'lift': '향상도'
}))

# 📊 6. 사용자 조건 기반 실패율 예측
st.subheader("6. 사용자 조건 기반 실패 리스크 예측")
soil = st.selectbox("토양 유형", df["Soil_Type"].unique())
water = st.selectbox("물 주기", df["Water_Frequency"].unique())
fert = st.selectbox("비료 유형", df["Fertilizer_Type"].unique())
sun = st.slider("햇빛 노출 시간", float(df["Sunlight_Hours"].min()), float(df["Sunlight_Hours"].max()), 6.0)
temp = st.slider("온도", float(df["Temperature"].min()), float(df["Temperature"].max()), 25.0)
hum = st.slider("습도", float(df["Humidity"].min()), float(df["Humidity"].max()), 60.0)

input_data = pd.DataFrame([[soil, water, fert, sun, temp, hum]],
                          columns=["Soil_Type", "Water_Frequency", "Fertilizer_Type",
                                   "Sunlight_Hours", "Temperature", "Humidity"])
all_data = pd.concat([df, input_data], ignore_index=True)
all_encoded = pd.get_dummies(all_data.drop("Failure", axis=1, errors='ignore'))
all_encoded = all_encoded.fillna(0)
input_vector = all_encoded.iloc[[-1]]
data_vector = all_encoded.iloc[:-1]
input_vector = input_vector.reindex(columns=data_vector.columns, fill_value=0)

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
