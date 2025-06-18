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
    











st.subheader("3. 조건 조합별 생장 실패율 히트맵")

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








import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "Malgun Gothic"

st.set_page_config(layout="wide")
st.title("📊 3. 연속형 변수별 임계 구간에 따른 생장 실패율 분석")

df = pd.read_csv("plant_growth_data.csv")
df["Failure"] = 1 - df["Growth_Milestone"]

bin_settings = {
    "Sunlight_Hours": [4, 5, 6, 7, 8, 9, 10, 11, 12],
    "Temperature": [15, 20, 22, 25, 28, 30, 32, 35],
    "Humidity": [30, 40, 50, 60, 70, 80, 90]
}
name_map = {
    "Sunlight_Hours": "☀ 햇빛 노출 시간",
    "Temperature": "🌡 온도",
    "Humidity": "💧 습도"
}

for var in bin_settings:
    df[f"{var}_bin"] = pd.cut(df[var], bins=bin_settings[var])
    grouped = df.groupby(f"{var}_bin")["Failure"].mean().reset_index()
    grouped[f"{var}_bin"] = grouped[f"{var}_bin"].astype(str)

    x_labels = grouped[f"{var}_bin"].tolist()
    x_pos = list(range(len(x_labels)))
    y_values = grouped["Failure"].tolist()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_pos, y_values, marker='o', color='steelblue', linewidth=2.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_title(f"{name_map[var]}에 따른 생장 실패율 변화", fontsize=15)
    ax.set_ylabel("실패율", fontsize=12)
    ax.set_xlabel(f"{name_map[var]} 구간", fontsize=12)

    # 강조 점
    if var == "Sunlight_Hours":
        ax.scatter(x_pos[1], y_values[1], s=150, color="green", zorder=5)
        ax.text(x_pos[1], y_values[1]+0.015, "✅ 실패율 낮음", color="green", ha="center")

        ax.scatter(x_pos[6], y_values[6], s=150, color="red", zorder=5)
        ax.text(x_pos[6], y_values[6]+0.015, "⚠ 실패율 증가", color="red", ha="center")

    elif var == "Temperature":
        ax.scatter(x_pos[1], y_values[1], s=150, color="green", zorder=5)
        ax.text(x_pos[1], y_values[1]+0.015, "✅ 최적 온도", color="green", ha="center")

        ax.scatter(x_pos[5], y_values[5], s=150, color="red", zorder=5)
        ax.text(x_pos[5], y_values[5]+0.015, "⚠ 고온 위험", color="red", ha="center")

    elif var == "Humidity":
        ax.scatter(x_pos[2], y_values[2], s=150, color="green", zorder=5)
        ax.text(x_pos[2], y_values[2]+0.015, "✅ 적절 습도", color="green", ha="center")

        ax.scatter(x_pos[5], y_values[5], s=150, color="red", zorder=5)
        ax.text(x_pos[5], y_values[5]+0.015, "⚠ 고습 실패율 급등", color="red", ha="center")

    st.pyplot(fig)





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

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 📌 한글 폰트 설정 (윈도우 기준)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ✅ 페이지 설정
st.set_page_config(layout="wide")
st.title("📊 4. 변수 간 상호작용 분석: 온도 & 습도 조합별 생장 실패율")

# ✅ 데이터 불러오기
df = pd.read_csv("plant_growth_data.csv")
df["Failure"] = 1 - df["Growth_Milestone"]

# ✅ 온도 및 습도 구간화
df["Temp_bin"] = pd.cut(df["Temperature"], bins=[15, 20, 25, 30, 35])
df["Humid_bin"] = pd.cut(df["Humidity"], bins=[30, 40, 50, 60, 70, 80])

# ✅ 그룹별 실패율 계산
pivot_df = df.groupby(["Temp_bin", "Humid_bin"])["Failure"].mean().reset_index()
pivot_table = pivot_df.pivot(index="Humid_bin", columns="Temp_bin", values="Failure")

# ✅ 시각화 (Heatmap)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, cmap="YlOrRd", fmt=".2f", linewidths=0.5, ax=ax)

ax.set_title("🌡️ 온도 & 💧습도 조합에 따른 생장 실패율", fontsize=16)
ax.set_xlabel("온도 구간 (°C)")
ax.set_ylabel("습도 구간 (%)")

st.pyplot(fig)

# ✅ 인사이트 출력
st.markdown("### 🔍 주요 인사이트")
st.markdown("""
- ✅ **30~35°C & 70~80% 습도** 조합에서 **실패율 최고** → **고온 다습 환경은 생장을 저해**  
  → 스마트팜 내 통풍, 냉방, 수분 조절 필요

- ⚠️ **25~30°C & 60~70%**도 실패율 중상위 → **중온 다습 환경도 주의 대상**

- 🌿 **20~25°C & 50~60%** 조합은 가장 안정적 → **성장에 적합한 최적 환경**
""")

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
