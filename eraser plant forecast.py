from scipy.stats import f_oneway  # ✅ 추가
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.neighbors import KNeighborsClassifier

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(layout="wide")
st.title("식물 생장 분산 분석을 통한 스마트팜의 리스크 기반 작물관리 전략 : 🌱30129 김민서")

df = pd.read_csv("plant_growth_data.csv")
df["Failure"] = 1 - df["Growth_Milestone"]

name_map = {
    "Sunlight_Hours": "햇빛 노출 시간",
    "Temperature": "온도",
    "Humidity": "습도",
    "Failure": "실패율",
    "Temp_bin": "온도 구간",
    "Humidity_bin": "습도 구간"
}

# 1. 박스플롯
st.subheader("1. 생장 성공/실패군의 주요 변수 분포")
for feature in ["Sunlight_Hours", "Temperature", "Humidity"]:
    fig = px.box(df, x="Failure", y=feature, color="Failure",
                 title=f"{name_map[feature]}에 따른 생장 성공/실패 분포",
                 labels={"Failure": "성공(0)/실패(1)", feature: name_map[feature]})
    st.plotly_chart(fig, use_container_width=True)

import seaborn as sns
import matplotlib.pyplot as plt

st.subheader("2. 조건 조합별 생장 실패율 히트맵")

# 한글 컬럼명으로 변경
df_rename = df.rename(columns={
    "Soil_Type": "토양",
    "Water_Frequency": "물주기",
    "Fertilizer_Type": "비료",
    "Failure": "실패율"
})

# 조합별 평균 실패율 계산
combo_df = df_rename.groupby(["토양", "물주기", "비료"])["실패율"].mean().reset_index()
combo_df["물비료"] = combo_df["물주기"] + " × " + combo_df["비료"]

# 피벗 테이블 생성
pivot_df = combo_df.pivot(index="토양", columns="물비료", values="실패율")

# 히트맵 시각화
fig, ax = plt.subplots(figsize=(10, 10))  # 정사각형에 맞게 figsize 수정
sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="Blues",
            cbar_kws={"label": "실패율"}, square=True, ax=ax)

plt.title("토양 유형, 물 주기, 비료 조합별 생장 실패율")
plt.ylabel("토양 유형")
plt.xlabel("물주기 × 비료 조합")
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


# 4. 상호작용 분석
st.subheader("4. 변수 간 상호작용 분석: 온도 & 습도 조합별 실패율")
df["Temp_bin"] = pd.cut(df["Temperature"], bins=[0, 20, 25, 30, 35, 40])
df["Humidity_bin"] = pd.cut(df["Humidity"], bins=[0, 40, 50, 60, 70, 100])
cross_df = df.groupby(["Temp_bin", "Humidity_bin"])["Failure"].mean().reset_index()
cross_df["Temp_bin"] = cross_df["Temp_bin"].astype(str)
cross_df["Humidity_bin"] = cross_df["Humidity_bin"].astype(str)
fig = px.density_heatmap(cross_df, x="Temp_bin", y="Humidity_bin", z="Failure",
                         color_continuous_scale="Reds",
                         labels={"Temp_bin": "온도 구간", "Humidity_bin": "습도 구간", "Failure": "실패율"})
st.plotly_chart(fig, use_container_width=True)

# ✅ 4-2. 분산분석(ANOVA)
st.subheader("📌 [ANOVA] 조건 그룹 간 생장 평균값 유의미 차이 검정")
df["Condition_Group"] = df["Soil_Type"] + "_" + df["Water_Frequency"] + "_" + df["Fertilizer_Type"]
grouped = df.groupby("Condition_Group")["Growth_Milestone"].apply(list)
valid_groups = [g for g in grouped if len(g) >= 3]

if len(valid_groups) >= 2:
    anova_result = f_oneway(*valid_groups)
    st.write(f"F값: {anova_result.statistic:.3f}, p값: {anova_result.pvalue:.4f}")
    if anova_result.pvalue < 0.05:
        st.success("✅ 조건 간 생장 결과에 유의미한 차이가 존재합니다. (p < 0.05)")
    else:
        st.warning("⚠️ 유의미한 차이가 발견되지 않았습니다.")
else:
    st.warning("⚠️ 분산분석에 필요한 조건 그룹 수 또는 표본 수가 부족합니다.")

# 5. 연관 규칙
st.subheader("5. 연관규칙 기반 위험 조합 탐색")
rule_df = df.copy()
rule_df = pd.get_dummies(rule_df[["Soil_Type", "Water_Frequency", "Fertilizer_Type"]])
rule_df["Failure"] = df["Failure"]
frequent_items = apriori(rule_df, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1)
risk_rules = rules[rules['consequents'].astype(str).str.contains('Failure')]
st.dataframe(risk_rules[['antecedents', 'support', 'confidence', 'lift']].rename(columns={
    'antecedents': '조건 조합', 'support': '지지도', 'confidence': '신뢰도', 'lift': '향상도'
}))

from scipy.stats import f_oneway

for i, (feature, bins) in enumerate([("Sunlight_Hours", 6), ("Temperature", 6), ("Humidity", 6)]):
    df[f"{feature}_bin"] = pd.cut(df[feature], bins)
    bin_df = df.groupby(f"{feature}_bin")["Failure"].mean().reset_index()
    bin_df[f"{feature}_bin"] = bin_df[f"{feature}_bin"].astype(str)

    fig = px.bar(bin_df, x=f"{feature}_bin", y="Failure",
                 title=f"{name_map[feature]} 구간별 생장 실패율",
                 labels={"Failure": "실패율", f"{feature}_bin": f"{name_map[feature]} 구간"})

    # ✅ key 추가로 중복 방지
    st.plotly_chart(fig, use_container_width=True, key=f"bar_{feature}_{i}")

    # 분산분석
    from scipy.stats import f_oneway
    groups = [df[df[f"{feature}_bin"] == bin_group]["Failure"] for bin_group in df[f"{feature}_bin"].unique()]
    anova_result = f_oneway(*groups)

    st.markdown(f"**🔬 {name_map[feature]}에 따른 실패율 분산분석 결과:**")
    st.markdown(f"- F값: `{anova_result.statistic:.3f}`")
    st.markdown(f"- p값: `{anova_result.pvalue:.4f}`")

    if anova_result.pvalue < 0.05:
        st.success("👉 구간별 실패율 차이가 통계적으로 유의함 (p < 0.05)~")
    else:
        st.info("➖ 통계적으로 유의한 차이는 없음 (p ≥ 0.05)~")


# 6. 사용자 예측
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
all_encoded = pd.get_dummies(all_data.drop("Failure", axis=1, errors='ignore')).fillna(0)
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
