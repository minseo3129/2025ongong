import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px

# ✅ 페이지 설정
st.set_page_config(layout="wide")
st.title("🌱 식물 생장 분포 분석을 통한 스마트팜의 리스크 기반 작물관리 전략 : 30129 김민서")

# ✅ 시각화 스타일
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

# ✅ 데이터 로딩
df = pd.read_csv("plant_growth_data.csv")
df["Failure"] = 1 - df["Growth_Milestone"]

# ✅ 변수 이름 매핑
name_map = {
    "Sunlight_Hours": "☀ 햇빛 노출 시간",
    "Temperature": "🌡 온도",
    "Humidity": "💧 습도"
}

# 📊 1. 생장 성공/실패군의 주요 변수 분포
st.subheader("1. 생장 성공/실패군의 주요 변수 분포")
for feature in name_map:
    fig = px.box(df, x="Failure", y=feature, color="Failure",
                 title=f"{name_map[feature]}에 따른 생장 성공/실패 분포",
                 labels={"Failure": "성공(0)/실패(1)", feature: name_map[feature]})
    st.plotly_chart(fig, use_container_width=True)
    




# 📊 4. 조건 조합별 생장 실패율 히트맵
st.subheader("4. 조건 조합별 생장 실패율 히트맵")
df_rename = df.rename(columns={"Soil_Type": "토양", "Water_Frequency": "물주기", "Fertilizer_Type": "비료", "Failure": "실패율"})
combo_df = df_rename.groupby(["토양", "물주기", "비료"])["실패율"].mean().reset_index()
combo_df["물비료"] = combo_df["물주기"] + " × " + combo_df["비료"]
pivot_df = combo_df.pivot(index="토양", columns="물비료", values="실패율")
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="Blues", cbar_kws={"label": "실패율"}, square=True, ax=ax)
plt.title("토양 유형, 물 주기, 비료 조합별 생장 실패율")
plt.ylabel("토양 유형")
plt.xlabel("물주기 × 비료 조합")
st.pyplot(fig)


# 📊 2. 조건별 생장 결과의 분산 분석
st.subheader("2. 조건별 생장 결과의 분산 분석")

# 조건조합 컬럼 생성
df["조건조합"] = df["Soil_Type"] + " | " + df["Water_Frequency"] + " | " + df["Fertilizer_Type"]

# 조건별 통계량 계산
group_stats = df.groupby("조건조합")["Growth_Milestone"].agg(['mean', 'var', 'std', 'count']).reset_index()
group_stats.columns = ['조건 조합', '평균 생장값', '분산', '표준편차', '샘플 수']

# 샘플 수 3개 이상 조건만 필터링
filtered = group_stats[group_stats['샘플 수'] >= 3].sort_values(by='분산', ascending=False)

# 📋 상위 7개 조건 데이터프레임 표시
st.markdown("### 🔍 분산값 기준 상위 7개 불안정 조건 그룹")
st.dataframe(filtered.head(7), use_container_width=True)

# 📈 상위 조건들에 대한 생장률 분포 박스플롯
st.markdown("### 📊 상위 분산 조건 그룹별 생장값 분포")
top_conditions = filtered.head(7)['조건 조합'].tolist()
subset = df[df["조건조합"].isin(top_conditions)]

fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(data=subset, x="조건조합", y="Growth_Milestone", palette="Blues", ax=ax)
ax.set_title("상위 분산 조건 그룹의 생장값 분포 (Top 7)", fontsize=15)
ax.set_xlabel("조건 조합", fontsize=12)
ax.set_ylabel("Growth_Milestone (생장률)", fontsize=12)
plt.xticks(rotation=45)
st.pyplot(fig)




# 📊 5. 연관규칙 기반 위험 조합 탐색
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

soil = st.selectbox("토양 유형", df["Soil_Type"].unique(), key="soil_selectbox")
water = st.selectbox("물 주기", df["Water_Frequency"].unique(), key="water_selectbox")
fert = st.selectbox("비료 유형", df["Fertilizer_Type"].unique(), key="fert_selectbox")
sun = st.slider("햇빛 노출 시간", float(df["Sunlight_Hours"].min()), float(df["Sunlight_Hours"].max()), 6.0, key="sun_slider")
temp = st.slider("온도", float(df["Temperature"].min()), float(df["Temperature"].max()), 25.0, key="temp_slider")
hum = st.slider("습도", float(df["Humidity"].min()), float(df["Humidity"].max()), 60.0, key="hum_slider")

input_data = pd.DataFrame([[soil, water, fert, sun, temp, hum]],
                          columns=["Soil_Type", "Water_Frequency", "Fertilizer_Type",
                                   "Sunlight_Hours", "Temperature", "Humidity"])
all_data = pd.concat([df, input_data], ignore_index=True)
all_encoded = pd.get_dummies(all_data.drop("Failure", axis=1, errors='ignore'))

input_vector = all_encoded.iloc[[-1]]
data_vector = all_encoded.iloc[:-1]
input_vector = input_vector.reindex(columns=data_vector.columns, fill_value=0)
input_vector = input_vector.fillna(0)  # 🔧 NaN 오류 방지

labels = df["Failure"]
model = KNeighborsClassifier(n_neighbors=5)
model.fit(data_vector, labels)
pred_prob = model.predict_proba(input_vector)[0][1]

# ✅ 조건 조합 기반 신뢰도 정보 추가
st.markdown(f"### 🔍 예측된 실패 확률: **{round(pred_prob * 100, 1)}%**")

user_group = f"{soil} | {water} | {fert}"

group_stats = df.copy()
group_stats["조건조합"] = group_stats["Soil_Type"] + " | " + group_stats["Water_Frequency"] + " | " + group_stats["Fertilizer_Type"]
group_stats = group_stats.groupby("조건조합")["Growth_Milestone"].agg(['mean', 'var', 'std', 'count']).reset_index()
group_stats = group_stats.rename(columns={
    '조건조합': '조건 조합',
    'mean': '평균 생장값',
    'var': '분산',
    'std': '표준편차',
    'count': '샘플 수'
})

uncertainty_info = group_stats[group_stats["조건 조합"] == user_group]
if not uncertainty_info.empty:
    std_val = uncertainty_info["표준편차"].values[0]
    st.markdown(f"📉 해당 조건의 생장 결과 **표준편차**: `{std_val:.3f}`")

    high_thresh = group_stats["표준편차"].quantile(0.8)
    low_thresh = group_stats["표준편차"].quantile(0.2)

    if std_val > high_thresh:
        st.warning("⚠ **예측 신뢰도 낮음**: 동일 조건 내 결과의 편차가 큽니다.")
    elif std_val < low_thresh:
        st.success("✅ **안정된 조건**: 결과 일관성이 높습니다.")
    else:
        st.info("ℹ **평균 수준의 변동성**을 가진 조건입니다.")
else:
    st.info("🔎 해당 조건 조합에 대한 충분한 통계 데이터가 없습니다.")

# ✅ 예측 결과에 따른 피드백
if pred_prob >= 0.6:
    st.error("🔥 실패 확률이 높습니다. 차광, 냉방, 환기 등 관리 강화 필요")
elif pred_prob >= 0.3:
    st.warning("🌤 중간 수준의 실패 위험. 조건 조정 고려")
else:
    st.success("🌱 조건이 양호합니다. 안정적인 생장이 기대됩니다.")
