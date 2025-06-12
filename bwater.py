import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os

from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"  # 또는 "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

# ✅ 한글 폰트 및 그래프 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = 'tight'

# ✅ 페이지 설정
st.set_page_config(page_title="물의 음용 가능성 판단 시스템", layout="wide")
st.title("💧 수질 기반 음용 가능성 예측 시스템")

# ✅ 데이터 로드
df = pd.read_csv("water_potability.csv")
features = df.columns[:-1]

# ✅ 변수 설명 및 WHO 기준
feature_meta = {
    "ph": {
        "label": "수소 이온 농도 (pH)", "unit": "", "min": 6.5, "max": 8.5,
        "desc": "🧪 WHO 권장: 6.5~8.5\n- 물의 산성/알칼리성 지표이며 극단값은 자극 유발",
        "cause": "산성/알칼리성 자극 가능", "solution": "중화제 사용 또는 자연 여과"
    },
    "Hardness": {
        "label": "경도", "unit": "mg/L", "max": 500,
        "desc": "🧪 WHO 권장: 최대 500\n- 칼슘/마그네슘 농도. 세제 거품 감소 등 영향",
        "cause": "물맛, 세제 작용 저하", "solution": "연수기 또는 이온교환 장치"
    },
    "Solids": {
        "label": "총 용존 고형물 (TDS)", "unit": "mg/L", "max": 1000,
        "desc": "🧪 WHO 권장: 최대 1000\n- 무기/유기물 혼합. 물맛, 색에 영향",
        "cause": "미네랄 과잉, 색/맛 변화", "solution": "역삼투압 또는 활성탄 여과"
    },
    "Chloramines": {
        "label": "클로라민", "unit": "ppm", "max": 4,
        "desc": "🧪 WHO 권장: 최대 4\n- 염소 기반 소독제의 잔류 성분",
        "cause": "잔류염소 독성 가능성", "solution": "활성탄 필터 또는 UV 소독"
    },
    "Sulfate": {
        "label": "황산염", "unit": "mg/L", "max": 250,
        "desc": "🧪 권장 기준: 최대 250\n- 고농도 시 설사 등 소화장애 유발",
        "cause": "설사, 위장 장애", "solution": "역삼투압, 석회화 처리"
    },
    "Conductivity": {
        "label": "전기전도도", "unit": "μS/cm", "max": 400,
        "desc": "🧪 WHO 권장: 최대 400\n- 이온 농도 지표. 고농도 시 심혈관계에 영향 가능",
        "cause": "이온 과잉으로 생리적 부담", "solution": "탈염 시스템, 증류"
    },
    "Organic_carbon": {
        "label": "유기 탄소", "unit": "mg/L", "max": 2,
        "desc": "🧪 WHO 권장: 최대 2\n- 소독 부산물(THMs) 생성 원인. 발암 우려",
        "cause": "THMs 유발 → 발암 가능성", "solution": "오존 처리, 유기물 여과"
    },
    "Trihalomethanes": {
        "label": "트리할로메탄", "unit": "ppb", "max": 80,
        "desc": "🧪 WHO 권장: 최대 80\n- 염소 소독 부산물. 간/신장에 해로움",
        "cause": "장기 노출 시 암 유발", "solution": "UV 소독 또는 유기물 사전 제거"
    },
    "Turbidity": {
        "label": "탁도", "unit": "NTU", "max": 5,
        "desc": "🧪 WHO 권장: 최대 5\n- 부유물. 병원성 미생물 번식 가능",
        "cause": "병원균 존재 가능성", "solution": "응집, 침전, 모래 여과"
    }
}

# ✅ 모델 학습
X = df.drop("Potability", axis=1)
y = df["Potability"]
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# ✅ 사용자 입력
st.header("🔎 수질 항목 입력")
user_input = {}
for f in features:
    meta = feature_meta[f]
    val = st.number_input(f"{meta['label']} ({meta['unit']})", min_value=0.0, step=0.1, key=f)
    st.caption(meta["desc"])
    user_input[f] = val

# ✅ 예측 실행
if st.button("📈 예측 실행"):
    input_df = pd.DataFrame([user_input])
    input_df_imputed = pd.DataFrame(imputer.transform(input_df), columns=features)

    violations = []
    for f, val in user_input.items():
        meta = feature_meta[f]
        if "min" in meta and val < meta["min"]:
            violations.append((meta["label"], f"{val} → 기준 미달", meta["cause"], meta["solution"]))
        elif "max" in meta and val > meta["max"]:
            violations.append((meta["label"], f"{val} → 기준 초과", meta["cause"], meta["solution"]))

    if violations:
        st.error("🚫 음용 불가 - WHO 기준 초과 항목 존재")
        st.subheader("📌 문제 항목 및 해결 방안")
        for label, 상태, 원인, 해결 in violations:
            st.markdown(f"- 🔍 **{label}**\n  - 상태: {상태}\n  - 원인: {원인}\n  - 해결 방안: {해결}")
    else:
        pred = model.predict(input_df_imputed)[0]
        prob = model.predict_proba(input_df_imputed)[0][pred]
        if pred == 1:
            st.success(f"✅ 이 물은 **음용 가능합니다**. (신뢰도: {prob*100:.2f}%)")
        else:
            st.warning(f"⚠ 이 물은 **음용 불가능합니다**. (신뢰도: {prob*100:.2f}%)")

# 중요도 시각화
st.subheader("📊 변수 중요도 (예측 모델 기반)")
importance_df = pd.DataFrame({
    "항목": [feature_meta[f]["label"] for f in features],
    "중요도": model.feature_importances_
}).sort_values(by="중요도", ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=importance_df, x="중요도", y="항목", ax=ax, palette="crest")
ax.set_title("수질 항목별 중요도", fontsize=15)
ax.tick_params(axis='y', labelsize=12)
ax.set_xlabel("중요도", fontsize=12)
ax.set_ylabel("")  # y축 이름 제거
plt.tight_layout()
st.pyplot(fig)


# ✅ 상관관계 히트맵 시각화
st.subheader("🔗 수질 항목 간 상관관계 분석")

# 상관행렬 계산
corr_matrix = pd.DataFrame(X_imputed, columns=features).corr()

# 히트맵 그리기
fig_corr, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    corr_matrix,
    annot=True, fmt=".2f",
    cmap="coolwarm",
    cbar=True,
    square=True,
    linewidths=0.5,
    annot_kws={"size": 9}
)
ax.set_title("💡 수질 항목 간 상관관계 히트맵", fontsize=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

st.pyplot(fig_corr)

# 간단한 해설 추가
st.markdown("""
🔍 **해설**  
- **상관계수 1.00**: 완전 양의 상관 (예: 자기 자신과의 관계)  
- **0.0 부근**: 거의 관계 없음  
- **음수(-)**: 한 값이 커질수록 다른 값이 작아지는 경향  
- 대부분 항목 간 상관성이 낮음(0.1 이하) → 예측 변수로서 서로 독립적인 정보 제공 가능성 ↑  
""")


