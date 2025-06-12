import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ✅ 글자 잘림 방지 설정
plt.rcParams["font.family"] = "Malgun Gothic"   # 한글 폰트 설정
plt.rcParams["axes.unicode_minus"] = False      # 마이너스 부호 깨짐 방지
plt.rcParams["figure.dpi"] = 120                # 해상도 높이기
plt.rcParams["savefig.bbox"] = 'tight'          # 여백 없이 저장

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=importance_df, x="중요도", y="항목", ax=ax, palette="crest")

# ✅ 축 라벨 크기 및 정렬 설정
ax.set_title("수질 항목별 중요도", fontsize=14)
ax.tick_params(axis='y', labelsize=12)  # y축 글자 크기 키우기
ax.set_xlabel("중요도", fontsize=12)
ax.set_ylabel("")  # y축 제목 제거


# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False



# 페이지 설정
st.set_page_config(page_title="물의 음용 가능성 판단 시스템", layout="wide")
st.title("💧 수질 기반 음용 가능성 예측 시스템")

# 내장 데이터 로드
df = pd.read_csv("water_potability.csv")
features = df.columns[:-1]

# 변수 정보 및 WHO 기준
feature_meta = {
    "ph": {
        "label": "수소 이온 농도 (pH)",
        "unit": "",
        "desc": "🧪 WHO 권장: 6.5~8.5 \n- pH는 물의 산염기 균형을 평가하는 중요한 지표이며, 물이 산성인지 알칼리성인지를 나타냅니다.",
        "min": 6.5, "max": 8.5,
        "cause": "극단적인 산도는 소화기 및 피부 자극 가능",
        "solution": "중화제 또는 자연 여과로 조절"
    },
    "Hardness": {
        "label": "경도",
        "unit": "mg/L",
        "desc": "🧪 WHO 권장: 최대 500 \n- 경도는 주로 칼슘(Ca)과 마그네슘(Mg) 염에 의해 발생합니다. 물이 암석 등을 통과하며 이 물질들과 접촉하는 시간에 따라 경도 수준이 결정됩니다. 원래는 물이 비누와 반응해 침전물을 생성하는 정도로 경도를 정의했습니다.",
        "max": 500,
        "cause": "높은 경도는 미각 변화 및 세제 작용 저하",
        "solution": "연수기 또는 이온교환 필터 사용"
    },
    "Solids": {
        "label": "총 용존 고형물 (TDS)",
        "unit": "mg/L",
        "desc": "🧪 WHO 권장: 최대 1000 \n- 물은 칼륨, 칼슘, 나트륨, 중탄산염, 염화물, 마그네슘, 황산염 등의 무기물 및 일부 유기물을 용해시킬 수 있습니다. TDS가 높을수록 물의 맛이 나빠지고 색깔이 탁해질 수 있습니다. \n- WHO는 500mg/L을 권장하며, 1000mg/L을 최대 한도로 설정하고 있습니다.",
        "max": 1000,
        "cause": "무기물·유기물 과잉으로 물맛 저하 및 위장장애 가능성",
        "solution": "활성탄 필터, 역삼투압 여과 적용"
    },
    "Chloramines": {
        "label": "클로라민",
        "unit": "ppm",
        "desc": "🧪 WHO 권장: 최대 4 \n- 염소와 암모니아를 결합하여 생성되는 소독제로, 공공 수돗물 정수에 사용됩니다.\n-WHO는 4mg/L(또는 4ppm)까지는 안전한 수준으로 보고 있습니다.",
        "max": 4,
        "cause": "잔류 염소가 인체에 해로울 수 있음",
        "solution": "탄소 필터 또는 UV 소독"
    },
    "Sulfate": {
        "label": "황산염",
        "unit": "mg/L",
        "desc": "🧪 권장 기준: 최대 250 \n- 황산염은 자연 상태에서 토양, 암석, 공기, 식물, 지하수 등에 존재합니다. 바닷물에는 약 2700mg/L 수준으로 존재하며, 민물에는 일반적으로 3~30mg/L 범위로 나타나지만 특정 지역에선 1000mg/L까지도 측정됩니다.",
        "max": 250,
        "cause": "설사, 위장 자극 가능성",
        "solution": "석회화, 역삼투압 처리"
    },
    "Conductivity": {
        "label": "전기전도도",
        "unit": "μS/cm",
        "desc": "🧪 WHO 권장: 최대 400 \n- 순수한 물은 전기를 거의 통하지 않지만, 이온 농도가 높아질수록 전도성이 증가합니다. 전기전도도는 이온 농도를 간접적으로 반영하는 지표",
        "max": 400,
        "cause": "과다 이온 농도는 심혈관계 문제 유발 가능",
        "solution": "탈염 시스템 적용"
    },
    "Organic_carbon": {
        "label": "유기 탄소",
        "unit": "mg/L",
        "desc": "🧪 WHO 권장: 최대 2\n- 유기 탄소는 물 속 유기물의 총량을 나타냅니다.\n- 유기 탄소가 높을수록 소독 시 발암성 부산물(THMs) 생성 가능성이 큽니다.",
        "max": 2,
        "cause": "염소와 반응 시 발암물질(THMs) 생성",
        "solution": "오존 처리, 유기물 여과"
    },
    "Trihalomethanes": {
        "label": "트리할로메탄",
        "unit": "ppb",
        "desc": "🧪 WHO 권장: 최대 80\n- 염소 소독 부산물로 장기 노출 시 간·신장 손상 가능",
        "max": 80,
        "cause": "장기 노출 시 암 유발 위험",
        "solution": "UV 소독 또는 유기물 제거"
    },
    "Turbidity": {
        "label": "탁도",
        "unit": "NTU",
        "desc": "🧪 WHO 권장: 최대 5\n- 탁도가 높으면 미생물 번식 위험이 커집니다.",
        "max": 5,
        "cause": "부유물은 병원성 미생물 서식 위험",
        "solution": "응집, 침전, 모래 여과"
    }
}


# 모델 학습
X = df.drop("Potability", axis=1)
y = df["Potability"]
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 사용자 입력
st.header("🔎 수질 항목 입력")
user_input = {}
for f in features:
    meta = feature_meta[f]
    val = st.number_input(f"{meta['label']} ({meta['unit']})", min_value=0.0, step=0.1, key=f)
    st.caption(meta["desc"])
    user_input[f] = val

# 예측 실행
if st.button("📈 예측 실행"):
    input_df = pd.DataFrame([user_input])
    input_df_imputed = pd.DataFrame(imputer.transform(input_df), columns=features)

    # WHO 기준 초과 여부 판단
    violations = []
    for f, val in user_input.items():
        meta = feature_meta[f]
        if "min" in meta and val < meta["min"]:
            violations.append((meta["label"], f"{val} → 기준 미달", meta["cause"], meta["solution"]))
        elif "max" in meta and val > meta["max"]:
            violations.append((meta["label"], f"{val} → 기준 초과", meta["cause"], meta["solution"]))

    # 결과 출력
    if violations:
        st.error("🚫 음용 불가 - WHO 기준 초과 항목 존재")
        st.subheader("📌 문제 항목 및 해결 방안")
        for label, 상태, 원인, 해결 in violations:
            st.markdown(f"""
            - 🔍 **{label}**  
              상태: {상태}  
              원인: {원인}  
              해결 방안: {해결}
            """)
    else:
        pred = model.predict(input_df_imputed)[0]
        prob = model.predict_proba(input_df_imputed)[0][pred]
        if pred == 1:
            st.success(f"✅ 이 물은 **음용 가능합니다**. (신뢰도: {prob*100:.2f}%)")
        else:
            st.warning(f"⚠ 음용 **불가능**합니다. (신뢰도: {prob*100:.2f}%)")

fig, ax = plt.subplots(figsize=(10, 8))  # ✅ 크기를 넉넉하게 키우기
sns.barplot(data=importance_df, x="중요도", y="항목", ax=ax, palette="crest")

# ✅ 폰트 크기 및 y축 간격 넓히기
ax.set_title("수질 항목별 중요도", fontsize=16)
ax.set_xlabel("중요도", fontsize=13)
ax.set_ylabel("")  # y축 제목 제거
ax.tick_params(axis='y', labelsize=13)  # y축 글자 키우기

plt.tight_layout()  # ✅ 모든 글자 잘리지 않도록 자동 여백 조정
st.pyplot(fig)

# 변수 중요도 시각화
st.subheader("📊 변수 중요도 (예측 모델 기반)")
importance_df = pd.DataFrame({
    "항목": [feature_meta[f]["label"] for f in features],
    "중요도": model.feature_importances_
}).sort_values(by="중요도", ascending=False)

fig, ax = plt.subplots()
sns.barplot(data=importance_df, x="중요도", y="항목", ax=ax, palette="crest")
st.pyplot(fig)
