import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

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
        "desc": "🧪 WHO 권장: 6.5~8.5",
        "min": 6.5, "max": 8.5,
        "cause": "극단적인 산도는 소화기 및 피부 자극 가능",
        "solution": "중화제 또는 자연 여과로 조절"
    },
    "Hardness": {
        "label": "경도",
        "unit": "mg/L",
        "desc": "🧪 WHO 권장: 최대 500",
        "max": 500,
        "cause": "높은 경도는 미각 변화 및 세제 작용 저하",
        "solution": "연수기 또는 이온교환 필터 사용"
    },
    "Solids": {
        "label": "총 용존 고형물 (TDS)",
        "unit": "mg/L",
        "desc": "🧪 WHO 권장: 최대 1000",
        "max": 1000,
        "cause": "무기물·유기물 과잉으로 물맛 저하 및 위장장애 가능성",
        "solution": "활성탄 필터, 역삼투압 여과 적용"
    },
    "Chloramines": {
        "label": "클로라민",
        "unit": "ppm",
        "desc": "🧪 WHO 권장: 최대 4",
        "max": 4,
        "cause": "잔류 염소가 인체에 해로울 수 있음",
        "solution": "탄소 필터 또는 UV 소독"
    },
    "Sulfate": {
        "label": "황산염",
        "unit": "mg/L",
        "desc": "🧪 권장 기준: 최대 250",
        "max": 250,
        "cause": "설사, 위장 자극 가능성",
        "solution": "석회화, 역삼투압 처리"
    },
    "Conductivity": {
        "label": "전기전도도",
        "unit": "μS/cm",
        "desc": "🧪 WHO 권장: 최대 400",
        "max": 400,
        "cause": "과다 이온 농도는 심혈관계 문제 유발 가능",
        "solution": "탈염 시스템 적용"
    },
    "Organic_carbon": {
        "label": "유기 탄소",
        "unit": "mg/L",
        "desc": "🧪 WHO 권장: 최대 2",
        "max": 2,
        "cause": "염소와 반응 시 발암물질(THMs) 생성",
        "solution": "오존 처리, 유기물 여과"
    },
    "Trihalomethanes": {
        "label": "트리할로메탄",
        "unit": "ppb",
        "desc": "🧪 WHO 권장: 최대 80",
        "max": 80,
        "cause": "장기 노출 시 암 유발 위험",
        "solution": "UV 소독 또는 유기물 제거"
    },
    "Turbidity": {
        "label": "탁도",
        "unit": "NTU",
        "desc": "🧪 WHO 권장: 최대 5",
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

# 변수 중요도 시각화
st.subheader("📊 변수 중요도 (예측 모델 기반)")
importance_df = pd.DataFrame({
    "항목": [feature_meta[f]["label"] for f in features],
    "중요도": model.feature_importances_
}).sort_values(by="중요도", ascending=False)

fig, ax = plt.subplots()
sns.barplot(data=importance_df, x="중요도", y="항목", ax=ax, palette="crest")
st.pyplot(fig)
 
