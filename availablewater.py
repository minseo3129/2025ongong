import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

# ✅ 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# 페이지 설정
st.set_page_config(page_title="물의 음용 가능성 예측기", layout="wide")
st.title("💧 수질 항목 기반 음용 가능성 예측기")

# -------------------------------
# 1. 수질 항목 정보 설정
# -------------------------------
parameter_info = {
    "ph": {
        "label": "pH (산성/알칼리성 지표)",
        "unit": "",
        "description": "물의 산도 또는 염기성을 나타냄. WHO 권장: 6.5–8.5"
    },
    "Hardness": {
        "label": "경도 (mg/L)",
        "unit": "mg/L",
        "description": "칼슘/마그네슘 염 농도. 물에 미네랄이 많을수록 높아짐"
    },
    "Solids": {
        "label": "총 용존 고형물 (TDS, mg/L)",
        "unit": "mg/L",
        "description": "물에 녹아 있는 무기물+유기물. WHO 권장 상한: 1000 mg/L"
    },
    "Chloramines": {
        "label": "클로라민 (mg/L)",
        "unit": "mg/L",
        "description": "소독 화합물. 4 mg/L 이내가 안전"
    },
    "Sulfate": {
        "label": "황산염 (mg/L)",
        "unit": "mg/L",
        "description": "지질 기원 염류. 고농도는 장내 불편 유발 가능"
    },
    "Conductivity": {
        "label": "전기전도도 (μS/cm)",
        "unit": "μS/cm",
        "description": "이온 농도. WHO 권장: 400 μS/cm 이하"
    },
    "Organic_carbon": {
        "label": "유기 탄소 (mg/L)",
        "unit": "mg/L",
        "description": "유기물 총량. EPA 기준: 2 mg/L 이하 권장"
    },
    "Trihalomethanes": {
        "label": "트리할로메탄 (μg/L)",
        "unit": "μg/L",
        "description": "염소 소독 부산물. WHO 안전 기준: 80 μg/L 이하"
    },
    "Turbidity": {
        "label": "탁도 (NTU)",
        "unit": "NTU",
        "description": "부유 물질의 혼탁도. WHO 권장: 5 NTU 이하"
    }
}

# -------------------------------
# 2. 데이터 불러오기 및 모델 학습
# -------------------------------
df = pd.read_csv("water_potability.csv")
imputer = SimpleImputer(strategy="mean")
X = df.drop("Potability", axis=1)
y = df["Potability"]
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 3. 수치 입력 UI + 설명 표시
# -------------------------------
st.subheader("🧪 수질 항목 수치 입력")

user_input = {}
for col in df.columns[:-1]:
    info = parameter_info[col]
    with st.expander(f"➤ {info['label']}"):
        value = st.number_input(f"{info['label']} 입력값", min_value=0.0, step=0.1, key=col)
        st.caption(f"단위: **{info['unit']}**")
        st.markdown(f"💡 {info['description']}")
        user_input[col] = value

# -------------------------------
# 4. 예측 결과 출력
# -------------------------------
if st.button("예측하기"):
    input_df = pd.DataFrame([user_input])
    input_df_imputed = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)
    prediction = model.predict(input_df_imputed)[0]
    prob = model.predict_proba(input_df_imputed)[0][prediction]

    st.markdown("### ❓ 이 물은 음용해도 될까요?")
    if prediction == 1:
        st.success(f"🟢 **예**, 이 물은 음용 가능합니다! (신뢰도: {prob*100:.2f}%)")
    else:
        st.error(f"🔴 **아니요**, 이 물은 음용에 적합하지 않습니다. (신뢰도: {prob*100:.2f}%)")

# -------------------------------
# 5. 변수 중요도 시각화
# -------------------------------
st.subheader("📊 수질 항목 중요도 시각화")

importances = model.feature_importances_
features_kr = [parameter_info[col]["label"] for col in df.columns[:-1]]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(features_kr, importances, color='teal')
ax.set_title("음용 가능성에 영향을 주는 주요 수질 항목")
ax.invert_yaxis()
st.pyplot(fig)

# -------------------------------
# 6. 결론 설명
# -------------------------------
st.markdown("""
---
📘 **설명 요약**
- 위 그래프는 예측 모델이 각 항목을 얼마나 참고했는지 보여줍니다.
- 예를 들어, `유기 탄소`, `탁도`, `총 용존 고형물` 등이 높게 나오면 음용 불가 확률이 올라갑니다.
- 향후 `게이지 시각화`, `오염 점수 시뮬레이션` 등으로 확장 가능합니다.
""")
