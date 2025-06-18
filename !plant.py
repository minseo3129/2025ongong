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

# 📌 폰트 및 스타일 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

# ✅ 페이지 설정
st.set_page_config(layout="wide")
st.title("📊 온도 & 습도 조합별 생장 실패율 분석")

# ✅ 데이터 로딩
df = pd.read_csv("plant_growth_data.csv")
df["Failure"] = 1 - df["Growth_Milestone"]

# ✅ 구간화
temp_bins = [15, 20, 25, 30, 35]
humid_bins = [40, 50, 60, 70, 80]
df["Temp_bin"] = pd.cut(df["Temperature"], bins=temp_bins, include_lowest=True)
df["Humid_bin"] = pd.cut(df["Humidity"], bins=humid_bins, include_lowest=True)

# ✅ 조합별 실패율 계산
combo_df = df.groupby(["Temp_bin", "Humid_bin"])["Failure"].mean().reset_index()
combo_df.dropna(inplace=True)
combo_df["조합"] = combo_df["Temp_bin"].astype(str) + " & " + combo_df["Humid_bin"].astype(str)
combo_df = combo_df.sort_values(by="Failure", ascending=False)

# ✅ 시각화
fig, ax = plt.subplots(figsize=(14, 6))
sns.barplot(data=combo_df, x="조합", y="Failure", color="cornflowerblue", ax=ax)
ax.set_title("Failure Rate by Temperature & Humidity Combination", fontsize=15)
ax.set_xlabel("온도 & 습도 조합", fontsize=12)
ax.set_ylabel("실패율", fontsize=12)
plt.xticks(rotation=90)
st.pyplot(fig)

# ✅ 분석 요약
st.markdown("""
### 🔍 주요 분석 결과
- ✅ **30~35°C & 70~80%** 조합 → **가장 높은 실패율**  
  → 고온다습 환경은 생장을 강하게 저해하며 통풍/냉방 필요

- ⚠️ **25~30°C & 60~70%** 조합 → **중상위 실패율**  
  → 중온다습 환경도 위험할 수 있어 습도 제어 필요

- 🌱 **20~25°C & 50~60%** 조합 → **가장 낮은 실패율**  
  → 최적의 생장 조건으로 판단됨
""")


# 조건 그룹 생성
df["Condition"] = df["Soil_Type"] + "_" + df["Water_Frequency"] + "_" + df["Fertilizer_Type"]

# 조건별 생장 결과 분산 계산
group_var = df.groupby("Condition")["Growth_Milestone"].var().reset_index()
group_var.columns = ["조건", "생장결과 분산"]

# 상위 분산 조건 확인
top_var = group_var.sort_values(by="생장결과 분산", ascending=False).head(5)
st.subheader("📌 동일 조건 내 생장결과 분산이 큰 조건 Top 5")
st.dataframe(top_var)




import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.subheader("2. 조건별 생장 결과의 분산 분석")

# ✅ 조건 그룹 생성
df["조건조합"] = df["Soil_Type"] + " | " + df["Water_Frequency"] + " | " + df["Fertilizer_Type"]

# ✅ 그룹별 통계 계산
group_stats = df.groupby("조건조합")["Growth_Milestone"].agg(['mean', 'var', 'std', 'count']).reset_index()
group_stats = group_stats.rename(columns={
    '조건조합': '조건 조합',
    'mean': '평균 생장값',
    'var': '분산',
    'std': '표준편차',
    'count': '샘플 수'
})

# ✅ 샘플 수가 충분한 조건만 필터링 (예: 3개 이상)
filtered = group_stats[group_stats['샘플 수'] >= 3].sort_values(by='분산', ascending=False)

# ✅ 표 형태로 출력
st.markdown("### 🔍 분산값 기준 상위 불안정 조건 그룹")
st.dataframe(filtered.head(7), use_container_width=True)

# ✅ 상위 그룹 시각화
st.markdown("### 📊 상위 분산 조건 그룹별 생장값 분포")
top_conditions = filtered.head(5)['조건 조합'].tolist()
subset = df[df["조건조합"].isin(top_conditions)]

fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=subset, x="조건조합", y="Growth_Milestone", palette="Set2", ax=ax)
ax.set_title("상위 분산 조건 그룹의 생장값 분포", fontsize=14)
ax.set_xlabel("조건 조합", fontsize=12)
ax.set_ylabel("Growth_Milestone (생장 도달률)", fontsize=12)
plt.xticks(rotation=45)
st.pyplot(fig)









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








# ✅ 사용자 입력 조건 조합 생성
user_group = f"{soil} | {water} | {fert}"

# ✅ 분산 분석 데이터프레임(group_stats) 재사용
group_stats = df.groupby("조건조합")["Growth_Milestone"].agg(['mean', 'var', 'std', 'count']).reset_index()
group_stats = group_stats.rename(columns={
    '조건조합': '조건 조합',
    'mean': '평균 생장값',
    'var': '분산',
    'std': '표준편차',
    'count': '샘플 수'
})

# ✅ 해당 조건의 분산값 가져오기
uncertainty_info = group_stats[group_stats["조건 조합"] == user_group]

if not uncertainty_info.empty:
    std_val = uncertainty_info["표준편차"].values[0]

    st.markdown(f"📉 해당 조건의 생장 결과 **표준편차**: `{std_val:.3f}`")

    # ✅ 신뢰도에 따른 메시지 제공
    if std_val > group_stats["표준편차"].quantile(0.8):
        st.warning("⚠ 예측 신뢰도 낮음: 동일 조건 내 결과의 편차가 큽니다. 관측 결과가 불안정할 수 있습니다.")
    elif std_val < group_stats["표준편차"].quantile(0.2):
        st.success("✅ 안정된 조건: 예측 신뢰도 높음 (동일 조건 내 결과 일관성 높음)")
    else:
        st.info("ℹ 평균 수준의 변동성 조건입니다.")
else:
    st.info("해당 조건 조합에 대한 충분한 분산 데이터가 없습니다.")

