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

# Streamlit 기본 설정
st.set_page_config(page_title="물의 음용 가능성 판단 시스템", layout="wide")
st.title("💧 수질 기반 음용 가능성 예측 시스템")

# ✅ 사전 준비된 수질 데이터 불러오기
df = pd.read_csv("water_potability.csv")  # 같은 폴더에 있어야 함
features = df.columns[:-1]

# 수질 항목 메타데이터
feature_meta = {
    "ph": {"label": "수소 이온 농도 (pH)", "unit": "", "desc": "산도. WHO 기준 6.5~8.5"},
    "Hardness": {"label": "경도", "unit": "mg/L", "desc": "칼슘/마그네슘 농도. 비누 반응도"},
    "Solids": {"label": "총 용존 고형물 (TDS)", "unit": "mg/L", "desc": "미네랄 농도. 500~1000 mg/L 적정"},
    "Chloramines": {"label": "클로라민", "unit": "ppm", "desc": "소독 잔류물. 4ppm 이하 권장"},
    "Sulfate": {"label": "황산염", "unit": "mg/L", "desc": "과다시 설사 유발. 250mg/L 이하"},
    "Conductivity": {"label": "전기전도도", "unit": "μS/cm", "desc": "이온 농도. 400 μS/cm 이하"},
    "Organic_carbon": {"label": "유기 탄소", "unit": "mg/L", "desc": "발암성 부산물 유발. 2mg/L 이하"},
    "Trihalomethanes": {"label": "트리할로메탄", "unit": "ppb", "desc": "소독 부산물. 80ppb 이하"},
    "Turbidity": {"label": "탁도", "unit": "NTU", "desc": "부유물 농도. 5 NTU 이하"}
}

# 전처리
X = df[features]
y = df["Potability"]
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# 모델 학습
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 사용자 입력
st.header("🔎 수질 데이터 직접 입력")
user_input = {}
for feature in features:
    label = feature_meta[feature]["label"]
    unit = feature_meta[feature]["unit"]
    desc = feature_meta[feature]["desc"]
    val = st.number_input(f"{label} ({unit})", min_value=0.0, step=0.1, key=feature)
    st.caption(f"🧪 {desc}")
    user_input[feature] = val

# 예측 버튼
if st.button("📊 예측 실행"):
    input_df = pd.DataFrame([user_input])
    input_df_imputed = pd.DataFrame(imputer.transform(input_df), columns=features)
    pred = model.predict(input_df_imputed)[0]
    prob = model.predict_proba(input_df_imputed)[0][pred]

    if pred == 1:
        st.success(f"✅ 이 물은 음용 **가능**합니다. (신뢰도: {prob * 100:.2f}%)")
    else:
        st.error(f"🚫 이 물은 음용 **불가능**합니다. (신뢰도: {prob * 100:.2f}%)")
        st.subheader("🧯 주요 원인과 해결 방안")

        # 영향 큰 상위 3개 변수
        importance = model.feature_importances_
        top_idx = importance.argsort()[::-1][:3]
        top_features = [features[i] for i in top_idx]

        solutions = {
            "ph": "중화제 또는 자연 여과로 pH 조절 필요",
            "Hardness": "연수기 또는 역삼투압으로 경도 감소",
            "Solids": "활성탄 여과나 이온교환으로 TDS 제거",
            "Chloramines": "탄소 필터 또는 UV 소독 활용",
            "Sulfate": "역삼투압이나 석회화로 황산염 제거",
            "Conductivity": "증류 또는 탈염 장치 활용",
            "Organic_carbon": "오존 처리나 유기물 전처리",
            "Trihalomethanes": "염소 대신 UV 소독 또는 활성탄 필터",
            "Turbidity": "응집, 침전, 모래 여과 적용"
        }

        for f in top_features:
            st.markdown(f"🔍 **{feature_meta[f]['label']}**: {solutions[f]}")

# 변수 중요도 시각화
st.subheader("📈 수질 항목별 중요도")
importance_df = pd.DataFrame({
    "항목": [feature_meta[col]["label"] for col in features],
    "중요도": model.feature_importances_
}).sort_values(by="중요도", ascending=False)

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=importance_df, x="중요도", y="항목", ax=ax, palette="Blues_d")
ax.set_title("수질 항목별 결정트리 중요도", fontsize=14)
st.pyplot(fig)