import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

# 📌 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 🌊 사전 학습용 데이터 로딩
df = pd.read_csv("water_potability.csv")
features = df.columns[:-1]

# 변수 설명
feature_info = {
    "ph": ("수소 이온 농도 (pH)", "", "산도·알칼리도를 나타내며, WHO 권장 범위는 6.5~8.5입니다."),
    "Hardness": ("경도", "mg/L", "칼슘과 마그네슘 이온의 농도. 물의 비누 반응도와 관련됩니다."),
    "Solids": ("총 용존 고형물", "mg/L", "무기/유기물 총량. 500~1000mg/L 권장."),
    "Chloramines": ("클로라민", "ppm", "소독 잔류물. WHO 기준 4ppm 이하."),
    "Sulfate": ("황산염", "mg/L", "고농도 섭취 시 설사 유발. 250~1000mg/L 이하 권장."),
    "Conductivity": ("전기전도도", "μS/cm", "이온 농도 지표. WHO 기준 400μS/cm 이하."),
    "Organic_carbon": ("유기 탄소", "mg/L", "유기물 총량. TOC 2mg/L 이하 권장."),
    "Trihalomethanes": ("트리할로메탄", "ppb", "염소 소독 부산물. 80ppb 이하 권장."),
    "Turbidity": ("탁도", "NTU", "부유물 농도 지표. 5 NTU 이하 권장.")
}

# 해결 방안
solutions = {
    "ph": "중화제 또는 자연 여과를 통해 pH 조절이 필요합니다.",
    "Hardness": "연수기 사용 또는 역삼투압으로 칼슘/마그네슘 제거하세요.",
    "Solids": "TDS가 높다면 활성탄 여과 또는 이온교환 기술이 필요합니다.",
    "Chloramines": "탄소 필터나 UV 소독법으로 제거 가능합니다.",
    "Sulfate": "석회화 또는 역삼투압으로 제거 가능합니다.",
    "Conductivity": "증류 또는 탈염 장치를 이용하세요.",
    "Organic_carbon": "오존 처리, 활성탄 여과 등으로 TOC를 낮추세요.",
    "Trihalomethanes": "UV 소독 또는 염소 전 단계 유기물 제거가 필요합니다.",
    "Turbidity": "침전, 응집제, 모래 여과를 활용하세요."
}

# 🎯 데이터 전처리 및 모델 학습
X = df.drop("Potability", axis=1)
y = df["Potability"]
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 🖥 Streamlit 인터페이스
st.set_page_config(page_title="물의 음용 가능성 판단기", layout="centered")
st.title("💧 수질 기반 음용 가능성 예측 시스템")

st.header("🔢 수질 항목 입력")
user_input = {}
critical_flag = False
critical_reasons = []

for feature in features:
    name, unit, desc = feature_info[feature]
    val = st.number_input(f"{name} ({unit})", min_value=0.0, step=0.1, key=feature)
    st.caption(f"🧪 {desc}")
    user_input[feature] = val

    # 조건 위반 감지
    if feature == "ph" and not (6.5 <= val <= 8.5):
        critical_flag = True
        critical_reasons.append("pH가 권장 범위(6.5~8.5)를 벗어났습니다.")
    if feature == "Chloramines" and val > 4:
        critical_flag = True
        critical_reasons.append("클로라민 수치가 WHO 기준 4ppm을 초과합니다.")
    if feature == "Trihalomethanes" and val > 80:
        critical_flag = True
        critical_reasons.append("트리할로메탄 수치가 WHO 기준 80ppb를 초과합니다.")
    if feature == "Turbidity" and val > 5:
        critical_flag = True
        critical_reasons.append("탁도가 WHO 권장 5 NTU를 초과합니다.")

if st.button("📊 예측 실행"):
    input_df = pd.DataFrame([user_input])
    input_df_imputed = pd.DataFrame(imputer.transform(input_df), columns=features)

    if critical_flag:
        st.error("🚫 수질 항목이 WHO 기준을 벗어나 음용 불가능합니다.")
        for reason in critical_reasons:
            st.markdown(f"- ❗ {reason}")
    else:
        pred = model.predict(input_df_imputed)[0]
        prob = model.predict_proba(input_df_imputed)[0][pred]

        if pred == 1:
            st.success(f"✅ 이 물은 음용 **가능**합니다. (신뢰도: {prob*100:.2f}%)")
        else:
            st.error(f"🚫 이 물은 음용 **불가능**합니다. (신뢰도: {prob*100:.2f}%)")

            # 🎯 상위 원인 분석
            st.subheader("🧯 원인별 개선 방안")
            importance = model.feature_importances_
            sorted_idx = importance.argsort()[::-1]
            top_features = [features[i] for i in sorted_idx[:3]]

            for f in top_features:
                st.markdown(f"🔍 **{feature_info[f][0]}**: {solutions[f]}")

    # 📈 변수 중요도 시각화
    st.subheader("📊 변수 중요도 (트리 기반)")
    importance_df = pd.DataFrame({
        "항목": [feature_info[col][0] for col in features],
        "중요도": model.feature_importances_
    }).sort_values(by="중요도", ascending=True)

    fig, ax = plt.subplots()
    sns.barplot(data=importance_df, x="중요도", y="항목", ax=ax, palette="viridis")
    ax.set_title("수질 항목별 중요도")
    st.pyplot(fig)