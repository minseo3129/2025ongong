import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

# ✅ 한글 폰트 설정 (운영체제별 적용)
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:  # Linux
    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False

# ✅ 페이지 설정
st.set_page_config(page_title="💧 수질 기반 음용 가능성 예측 시스템", layout="wide")
st.title("💧 수질 기반 음용 가능성 예측 시스템")

# ✅ 내장 데이터 로드
df = pd.read_csv("water_potability.csv")
features = df.columns[:-1]

# ✅ 항목별 메타 정보
feature_info = {
    "ph": ("수소 이온 농도 (pH)", "", "산도·알칼리도를 나타냅니다. WHO 권장 범위는 6.5~8.5입니다."),
    "Hardness": ("경도", "mg/L", "칼슘, 마그네슘 이온 농도. 비누 반응성과 연관됩니다."),
    "Solids": ("총 용존 고형물", "mg/L", "무기/유기물 농도. WHO 권장: 500~1000 mg/L"),
    "Chloramines": ("클로라민", "ppm", "소독 잔류물. 기준: 4ppm 이하"),
    "Sulfate": ("황산염", "mg/L", "고농도 섭취 시 설사 유발. 권장: 250 mg/L 이하"),
    "Conductivity": ("전기전도도", "μS/cm", "이온 농도. 권장: 400 μS/cm 이하"),
    "Organic_carbon": ("유기 탄소", "mg/L", "유기물 총량. WHO 기준: 2 mg/L 이하"),
    "Trihalomethanes": ("트리할로메탄", "ppb", "염소 소독 부산물. WHO 기준: 80 ppb 이하"),
    "Turbidity": ("탁도", "NTU", "부유물 지표. WHO 기준: 5 NTU 이하")
}

# ✅ 모델 준비
X = df.drop("Potability", axis=1)
y = df["Potability"]
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_imputed, y)

# ✅ 중요도 시각화 (예측 전에)
st.subheader("📊 수질 항목별 중요도")
importance_df = pd.DataFrame({
    "항목": [feature_info[col][0] for col in features],
    "중요도": model.feature_importances_
}).sort_values(by="중요도", ascending=True)

fig, ax = plt.subplots()
sns.barplot(data=importance_df, x="중요도", y="항목", ax=ax, palette="viridis")
st.pyplot(fig)

# ✅ 사용자 입력
st.header("🔎 수질 정보 수동 입력")
user_input = {}
for col in features:
    label, unit, desc = feature_info[col]
    val = st.number_input(f"{label} ({unit})", min_value=0.0, step=0.1, key=col)
    st.caption(f"🧪 {desc}")
    user_input[col] = val

# ✅ 예측 버튼
if st.button("📌 음용 가능성 예측"):
    input_df = pd.DataFrame([user_input])
    input_df_imputed = pd.DataFrame(imputer.transform(input_df), columns=features)

    # ✅ 규정 위반 항목 검사
    thresholds = {
        "ph": (6.5, 8.5),
        "Hardness": (None, None),  # 참고용
        "Solids": (0, 1000),
        "Chloramines": (0, 4),
        "Sulfate": (0, 250),
        "Conductivity": (0, 400),
        "Organic_carbon": (0, 2),
        "Trihalomethanes": (0, 80),
        "Turbidity": (0, 5)
    }

    violated = []
    for key, (low, high) in thresholds.items():
        val = user_input[key]
        if (low is not None and val < low) or (high is not None and val > high):
            violated.append((feature_info[key][0], val))

    if violated:
        st.error("🚫 이 물은 음용 **불가능**합니다. 기준 초과 항목이 있습니다.")
        for name, val in violated:
            st.markdown(f"- ❗ **{name}** 수치 이상 또는 미달 → 입력값: {val}")
    else:
        pred = model.predict(input_df_imputed)[0]
        prob = model.predict_proba(input_df_imputed)[0][pred]
        if pred == 1:
            st.success(f"✅ 이 물은 **음용 가능**합니다! (신뢰도: {prob*100:.2f}%)")
        else:
            st.warning(f"⚠️ 데이터상으론 음용 **불가능**으로 예측됩니다. (신뢰도: {prob*100:.2f}%)")