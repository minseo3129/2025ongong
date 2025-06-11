import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"

# Streamlit 페이지 설정
st.set_page_config(page_title="물의 음용 가능성 판단 시스템", layout="wide")

st.title("💧 수질 기반 음용 가능성 예측 시스템")

# 데이터 로드
df = pd.read_csv("water_potability.csv")
features = df.columns[:-1]

# 항목 메타데이터 (라벨, 단위, 설명)
feature_meta = {
    "ph": {
        "label": "수소 이온 농도 (pH)",
        "unit": "",
        "desc": "산도·알칼리도를 나타내며, WHO 권장 범위는 6.5~8.5입니다."
    },
    "Hardness": {
        "label": "경도 (mg/L)",
        "unit": "mg/L",
        "desc": "칼슘과 마그네슘 이온의 농도. 물이 비누와 반응하는 능력을 나타냅니다."
    },
    "Solids": {
        "label": "총 용존 고형물 (mg/L)",
        "unit": "mg/L",
        "desc": "TDS 수치가 높으면 미네랄화가 심하며, 500~1000 mg/L 이하 권장됩니다."
    },
    "Chloramines": {
        "label": "클로라민 (ppm)",
        "unit": "ppm",
        "desc": "소독제 역할. WHO 기준 4ppm 이하에서 안전하다고 여겨집니다."
    },
    "Sulfate": {
        "label": "황산염 (mg/L)",
        "unit": "mg/L",
        "desc": "고농도 섭취 시 설사 유발 가능. 지역에 따라 최대 1000mg/L 이상 검출됨."
    },
    "Conductivity": {
        "label": "전기전도도 (μS/cm)",
        "unit": "μS/cm",
        "desc": "이온 농도와 밀접. 400 μS/cm 이하가 권장됩니다."
    },
    "Organic_carbon": {
        "label": "유기 탄소 (mg/L)",
        "unit": "mg/L",
        "desc": "유기물 총량을 의미. 염소와 반응 시 발암물질인 THMs 유발 가능. WHO 권장: 2mg/L 이하"
    },
    "Trihalomethanes": {
        "label": "트리할로메탄 (ppb)",
        "unit": "ppb",
        "desc": "염소 소독 부산물. 장기 섭취 시 암 유발 가능성 존재. 80ppb 이하 권장"
    },
    "Turbidity": {
        "label": "탁도 (NTU)",
        "unit": "NTU",
        "desc": "부유물 농도 지표. 5 NTU 이하가 안전한 물로 간주됩니다."
    }
}

# 전처리
X = df.drop("Potability", axis=1)
y = df["Potability"]
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 변수 중요도 시각화
st.subheader("📊 수질 항목이 음용 가능성에 미치는 영향")
importance_df = pd.DataFrame({
    "항목": [feature_meta[f]["label"] for f in features],
    "중요도": model.feature_importances_
}).sort_values(by="중요도", ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=importance_df, x="중요도", y="항목", palette="YlGnBu", ax=ax)
st.pyplot(fig)

# 사용자 입력
st.subheader("🧪 수질 데이터를 직접 입력해보세요")
user_input = {}
for f in features:
    label = feature_meta[f]["label"]
    unit = feature_meta[f]["unit"]
    desc = feature_meta[f]["desc"]
    user_input[f] = st.number_input(f"{label} ({unit})", min_value=0.0, step=0.1, format="%.2f")
    st.caption(f"ℹ️ {desc}")

# 예측 실행
if st.button("예측하기"):
    user_df = pd.DataFrame([user_input])
    user_df_imputed = pd.DataFrame(imputer.transform(user_df), columns=user_df.columns)
    prediction = model.predict(user_df_imputed)[0]
    prob = model.predict_proba(user_df_imputed)[0][prediction]

    st.markdown("### 🔍 예측 결과")
    if prediction == 1:
        st.success(f"✅ 이 물은 **음용 가능합니다**. (신뢰도: {prob*100:.2f}%)")
    else:
        st.error(f"⚠️ 이 물은 **음용에 적합하지 않습니다**. (신뢰도: {prob*100:.2f}%)")

# 해설 섹션
st.markdown("---")
st.subheader("📘 주요 항목 설명 및 기준")
for f in features:
    st.markdown(f"**{feature_meta[f]['label']}**")
    st.markdown(f"- 단위: {feature_meta[f]['unit']}")
    st.markdown(f"- 설명: {feature_meta[f]['desc']}")

# 향후 시각화 아이디어
st.markdown("""
---
🎨 **향후 확장 시각화 제안**
- 입력값 기반 방사형(Radar) 그래프 비교
- 과거 데이터 대비 위치 추정 시각화
- 다차원 축소(UMAP, PCA) 기반 위험도 클러스터링
""")