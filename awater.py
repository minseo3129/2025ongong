import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 📌 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 📁 CSV 업로드
st.title("💧 물의 음용 가능성 판단기")
uploaded_file = st.file_uploader("수질 데이터를 업로드하세요 (예: water_potability.csv)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # 전처리
    X = df.drop("Potability", axis=1)
    y = df["Potability"]
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # 모델 학습
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    # 변수 설명
    feature_info = {
        "ph": ("pH (산도)", "", "pH는 물의 산도/알칼리성을 나타내며 WHO 권장 범위는 6.5~8.5입니다."),
        "Hardness": ("경도", "mg/L", "칼슘, 마그네슘 농도. 경도가 높으면 물맛과 비누 거품에 영향."),
        "Solids": ("총 용존 고형물 (TDS)", "mg/L", "무기/유기물 농도. WHO는 500~1000mg/L를 권장."),
        "Chloramines": ("클로라민", "ppm", "소독 잔류물. 4ppm 이하가 안전 기준입니다."),
        "Sulfate": ("황산염", "mg/L", "높은 농도는 설사를 유발. 기준은 약 250mg/L 이하."),
        "Conductivity": ("전기전도도", "μS/cm", "이온 농도 지표. WHO 기준 400μS/cm 이하."),
        "Organic_carbon": ("유기 탄소", "mg/L", "2mg/L 이하 권장. 유해 소독 부산물과 관련 있음."),
        "Trihalomethanes": ("트리할로메탄", "ppm", "염소 소독 부산물. WHO 기준 80ppm 이하."),
        "Turbidity": ("탁도", "NTU", "부유물로 인한 탁도. 기준은 5 NTU 이하.")
    }

    # 사용자 입력
    st.header("🔎 수질 데이터 입력")
    user_input = {}
    for feature in df.columns[:-1]:
        name, unit, desc = feature_info[feature]
        val = st.number_input(f"{name} ({unit})", min_value=0.0, step=0.1, key=feature)
        st.caption(f"🧪 {desc}")
        user_input[feature] = val

    # 예측 및 결과
    if st.button("📊 예측 실행"):
        input_df = pd.DataFrame([user_input])
        input_df_imputed = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)
        pred = model.predict(input_df_imputed)[0]
        prob = model.predict_proba(input_df_imputed)[0][pred]

        if pred == 1:
            st.success(f"✅ 이 물은 음용 **가능**합니다. (신뢰도: {prob*100:.2f}%)")
        else:
            st.error(f"🚫 이 물은 음용 **불가능**합니다. (신뢰도: {prob*100:.2f}%)")

            # 원인 분석 및 해결방안 제시
            st.subheader("🧯 원인별 개선 방안")
            importance = model.feature_importances_
            sorted_idx = importance.argsort()[::-1]
            top_features = [df.columns[:-1][i] for i in sorted_idx[:3]]

            solutions = {
                "ph": "중화제 또는 자연 여과를 통해 pH 조절이 필요합니다.",
                "Hardness": "연수기를 사용하거나 역삼투압으로 칼슘/마그네슘 제거하세요.",
                "Solids": "TDS가 높다면 활성탄 여과 또는 이온교환 기술을 고려하세요.",
                "Chloramines": "탄소 필터나 UV 소독법을 도입해 클로라민을 제거할 수 있습니다.",
                "Sulfate": "석회화 및 역삼투압으로 황산염을 줄일 수 있습니다.",
                "Conductivity": "증류법 또는 탈염 장치로 이온 농도를 낮출 수 있습니다.",
                "Organic_carbon": "유기물 여과, 오존 처리 등으로 TOC를 줄이세요.",
                "Trihalomethanes": "염소 대신 UV 소독을 사용하거나 유기물 사전 제거가 필요합니다.",
                "Turbidity": "침전, 응집제 사용, 모래 여과로 탁도 제거가 가능합니다."
            }

            for f in top_features:
                st.markdown(f"🔍 **{feature_info[f][0]}**: {solutions[f]}")

    # 변수 중요도 시각화
    st.subheader("📈 수질 항목별 중요도")
    importance_df = pd.DataFrame({
        "항목": [feature_info[col][0] for col in df.columns[:-1]],
        "중요도": model.feature_importances_
    }).sort_values(by="중요도", ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(data=importance_df, x="중요도", y="항목", ax=ax, palette="viridis")
    st.pyplot(fig)