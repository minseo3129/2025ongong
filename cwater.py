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

# CSV 데이터 로드 (업로드 없이 사전 내장)
df = pd.read_csv("water_potability.csv")
features = df.columns[:-1]

# 변수 정보 (설명 포함)
feature_meta = {
    "ph": {
        "label": "수소 이온 농도 (pH)",
        "unit": "",
        "desc": "🧪 WHO 권장: 6.5~8.5. 산도·알칼리도 지표. 극단적인 pH는 건강에 해롭습니다."
    },
    "Hardness": {
        "label": "경도",
        "unit": "mg/L",
        "desc": "🧪 칼슘·마그네슘 농도. WHO 권장: 100~500mg/L. 미각 및 세제 작용 영향."
    },
    "Solids": {
        "label": "총 용존 고형물 (TDS)",
        "unit": "mg/L",
        "desc": "🧪 WHO 권장: 500~1000 mg/L. TDS가 높으면 미네랄 과다로 물맛 변화 및 건강 문제."
    },
    "Chloramines": {
        "label": "클로라민",
        "unit": "ppm",
        "desc": "🧪 WHO 기준: 4ppm 이하. 염소계 소독 잔류물."
    },
    "Sulfate": {
        "label": "황산염",
        "unit": "mg/L",
        "desc": "🧪 고농도(>250mg/L)는 설사 유발. 자연에서 흔히 존재함."
    },
    "Conductivity": {
        "label": "전기전도도",
        "unit": "μS/cm",
        "desc": "🧪 WHO 기준: 400 μS/cm 이하. 이온 농도 지표."
    },
    "Organic_carbon": {
        "label": "유기 탄소",
        "unit": "mg/L",
        "desc": "🧪 WHO 기준: 2mg/L 이하. 유기물 총량으로 수질 오염도와 관련.\n• 유기 탄소는 염소와 반응해 **발암성 부산물(THMs)** 생성 위험."
    },
    "Trihalomethanes": {
        "label": "트리할로메탄",
        "unit": "ppb",
        "desc": "🧪 WHO 기준: 80ppb 이하. 소독 부산물로 장기 섭취 시 암 유발 가능성."
    },
    "Turbidity": {
        "label": "탁도",
        "unit": "NTU",
        "desc": "🧪 WHO 기준: 5 NTU 이하. 부유물 농도 지표로 정수 상태 평가."
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

# 변수 중요도 시각화
st.subheader("📊 수질 변수 중요도 분석 (모델 기반)")
importance_df = pd.DataFrame({
    "항목": [feature_meta[f]["label"] for f in features],
    "중요도": model.feature_importances_
}).sort_values(by="중요도", ascending=False)

fig, ax = plt.subplots()
sns.barplot(data=importance_df, x="중요도", y="항목", ax=ax, palette="Blues_d")
st.pyplot(fig)

# 사용자 입력
st.header("🔎 수질 데이터 직접 입력")
user_input = {}
for f in features:
    meta = feature_meta[f]
    val = st.number_input(f"{meta['label']} ({meta['unit']})", min_value=0.0, step=0.1, key=f)
    st.caption(meta["desc"])
    user_input[f] = val

# 예측 버튼
if st.button("📈 예측 실행"):
    input_df = pd.DataFrame([user_input])
    input_df_imputed = pd.DataFrame(imputer.transform(input_df), columns=features)

    # WHO 기준 절대 조건 위반 여부
    unsafe = (
        user_input["ph"] < 6.5 or user_input["ph"] > 8.5 or
        user_input["Solids"] > 1000 or
        user_input["Trihalomethanes"] > 80 or
        user_input["Organic_carbon"] > 2 or
        user_input["Turbidity"] > 5 or
        user_input["Conductivity"] > 400 or
        user_input["Chloramines"] > 4
    )

    if unsafe:
        st.error("🚫 WHO 기준 초과 → **음용 불가능**")
    else:
        pred = model.predict(input_df_imputed)[0]
        prob = model.predict_proba(input_df_imputed)[0][pred]
        if pred == 1:
            st.success(f"✅ 이 물은 **음용 가능합니다**. (신뢰도: {prob*100:.2f}%)")
        else:
            st.warning(f"⚠ 이 물은 **음용 불가능**합니다. (신뢰도: {prob*100:.2f}%)")
            st.subheader("🧯 개선이 필요한 주요 항목")
            importance = model.feature_importances_
            sorted_idx = importance.argsort()[::-1]
            top_features = [features[i] for i in sorted_idx[:3]]

            solutions = {
                "ph": "→ pH 조절을 위해 중화제 또는 자연 여과 사용",
                "Hardness": "→ 연수기, 이온교환 필터 사용",
                "Solids": "→ 활성탄 여과 또는 이온 교환 방식 적용",
                "Chloramines": "→ 탄소 필터 또는 UV 소독 방식 사용",
                "Sulfate": "→ 역삼투압 또는 석회화 처리 고려",
                "Conductivity": "→ 증류 또는 탈염 처리 적용",
                "Organic_carbon": "→ 오존 처리, 활성탄 여과로 TOC 저감",
                "Trihalomethanes": "→ UV 소독 또는 유기물 사전 제거",
                "Turbidity": "→ 응집, 침전, 여과 방식 도입"
            }

            for f in top_features:
                st.markdown(f"🔍 **{feature_meta[f]['label']}**: {solutions[f]}")
