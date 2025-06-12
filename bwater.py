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

# 데이터 불러오기
df = pd.read_csv("water_potability.csv")
features = df.columns[:-1]

# WHO 설명 요약 출력
st.markdown("""
### 💡 수질 항목별 설명 요약 (WHO 기준 중심)
- **pH**: 산염기 상태 평가. 적정 범위: `6.5~8.5`
- **Hardness**: 칼슘, 마그네슘 염. 권장: `≤ 500 mg/L`
- **Solids (TDS)**: 무기물+유기물. 권장: `≤ 1000 mg/L`
- **Chloramines**: 염소+암모니아 소독제. `≤ 4 ppm`
- **Sulfate**: 천연 존재. `≤ 250 mg/L`
- **Conductivity**: 이온 농도 반영. `≤ 400 μS/cm`
- **Organic Carbon**: 유기물 양. `≤ 2 mg/L`
- **Trihalomethanes**: 염소 소독 부산물. `≤ 80 ppb`
- **Turbidity**: 부유물. `≤ 5 NTU`
""")

# ▶ feature_meta 선언 (여기 먼저 와야 함!)
feature_meta = {
    "ph": {"label": "수소 이온 농도 (pH)", "unit": "", "desc": "🧪 WHO 권장: 6.5~8.5", "min": 6.5, "max": 8.5, "cause": "산성/염기성으로 인한 자극", "solution": "중화제 사용"},
    "Hardness": {"label": "경도", "unit": "mg/L", "desc": "🧪 WHO 권장: 최대 500", "max": 500, "cause": "미각 변화", "solution": "연수기 사용"},
    "Solids": {"label": "총 용존 고형물", "unit": "mg/L", "desc": "🧪 WHO 권장: 최대 1000", "max": 1000, "cause": "맛 저하, 색 변화", "solution": "역삼투압 필터"},
    "Chloramines": {"label": "클로라민", "unit": "ppm", "desc": "🧪 WHO 권장: 최대 4", "max": 4, "cause": "잔류염소", "solution": "탄소필터"},
    "Sulfate": {"label": "황산염", "unit": "mg/L", "desc": "🧪 WHO 권장: 최대 250", "max": 250, "cause": "위장 장애", "solution": "석회화, 여과"},
    "Conductivity": {"label": "전기전도도", "unit": "μS/cm", "desc": "🧪 WHO 권장: 최대 400", "max": 400, "cause": "과다 이온 농도", "solution": "탈염"},
    "Organic_carbon": {"label": "유기 탄소", "unit": "mg/L", "desc": "🧪 WHO 권장: 최대 2", "max": 2, "cause": "THMs 생성", "solution": "오존 처리"},
    "Trihalomethanes": {"label": "트리할로메탄", "unit": "ppb", "desc": "🧪 WHO 권장: 최대 80", "max": 80, "cause": "장기 노출 위험", "solution": "UV 제거"},
    "Turbidity": {"label": "탁도", "unit": "NTU", "desc": "🧪 WHO 권장: 최대 5", "max": 5, "cause": "미생물 서식", "solution": "모래 여과"}
}

# ▶ 모델 학습
X = df.drop("Potability", axis=1)
y = df["Potability"]
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# ▶ 사용자 입력
st.header("🔎 수질 항목 입력")
user_input = {}
for f in features:
    meta = feature_meta[f]
    val = st.number_input(f"{meta['label']} ({meta['unit']})", min_value=0.0, step=0.1, key=f)
    st.caption(meta["desc"])
    user_input[f] = val

# ▶ 예측 실행
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
            st.markdown(f"""- **{label}**  
  상태: {상태}  
  원인: {원인}  
  해결 방안: {해결}""")
    else:
        pred = model.predict(input_df_imputed)[0]
        prob = model.predict_proba(input_df_imputed)[0][pred]
        if pred == 1:
            st.success(f"✅ 이 물은 **음용 가능합니다**. (신뢰도: {prob*100:.2f}%)")
        else:
            st.warning(f"⚠ 음용 **불가능**합니다. (신뢰도: {prob*100:.2f}%)")

# ▶ 변수 중요도 시각화
st.subheader("📊 변수 중요도 (예측 모델 기반)")
importance_df = pd.DataFrame({
    "항목": [feature_meta[f]["label"] for f in features],
    "중요도": model.feature_importances_
}).sort_values(by="중요도", ascending=False)

fig, ax = plt.subplots()
sns.barplot(data=importance_df, x="중요도", y="항목", ax=ax, palette="crest")
st.pyplot(fig)
