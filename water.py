import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# ✅ 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

# 📄 페이지 설정
st.set_page_config(page_title="물의 음용 가능성 분석기", layout="wide")
st.title("🚰 수질 분석을 통한 물의 음용 가능성 판단")

# 📁 데이터 불러오기
df = pd.read_csv("water_potability.csv")
imputer = SimpleImputer(strategy="mean")
X = df.drop("Potability", axis=1)
y = df["Potability"]
X_imputed = imputer.fit_transform(X)

# 🧠 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 📊 변수 중요도 시각화
st.subheader("📌 수질 항목이 음용 여부에 미치는 영향 (중요도 순)")

feature_labels = {
    "ph": "pH",
    "Hardness": "경도",
    "Solids": "총 용존 고형물",
    "Chloramines": "클로라민",
    "Sulfate": "황산염",
    "Conductivity": "전기전도도",
    "Organic_carbon": "유기 탄소",
    "Trihalomethanes": "트리할로메탄",
    "Turbidity": "탁도"
}

importance_df = pd.DataFrame({
    "Feature": df.columns[:-1],
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)
importance_df["Feature_KR"] = importance_df["Feature"].map(feature_labels)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=importance_df, x="Importance", y="Feature_KR", palette="viridis", ax=ax)
ax.set_title("각 수질 항목의 중요도")
ax.set_xlabel("중요도")
ax.set_ylabel("수질 항목")
st.pyplot(fig)

# 🧪 사용자 입력
st.subheader("🔍 수질 정보를 입력해 예측해보세요")

input_data = {}
for col in df.columns[:-1]:
    label = feature_labels.get(col, col)
    input_data[col] = st.number_input(f"{label}", min_value=0.0, step=0.1)

if st.button("예측하기"):
    user_df = pd.DataFrame([input_data])
    user_df_imputed = pd.DataFrame(imputer.transform(user_df), columns=user_df.columns)
    prediction = model.predict(user_df_imputed)[0]
    prob = model.predict_proba(user_df_imputed)[0][prediction]

    st.markdown("### ❓ 예측 결과")
    if prediction == 1:
        st.success(f"🟢 이 물은 음용 가능합니다! (신뢰도: {prob*100:.2f}%)")
    else:
        st.error(f"🔴 이 물은 음용에 적합하지 않습니다. (신뢰도: {prob*100:.2f}%)")

# 📘 과학적 설명
st.markdown("""
---
### 💡 유기 탄소(Organic Carbon)의 의미와 중요성

- 유기 탄소는 물 속 유기물의 총량을 나타냅니다. 예: 식물 잔해, 미생물, 부패물질 등
- 유기 탄소가 높을수록 **소독 과정 중 염소와 반응해 유해 부산물(DBPs)**이 생성될 가능성이 큽니다.
- 대표적인 부산물인 **트리할로메탄(THMs)**는 장기 섭취 시 **발암 가능성**이 있어, WHO는 **TOC 2mg/L 이하**를 권장합니다.
- 따라서 유기 탄소는 물의 오염 정도를 간접적으로 나타내며, **음용 여부 판단에 핵심적인 변수**입니다.

📚 출처:
- WHO: Guidelines for Drinking-water Quality
- US EPA: National Primary Drinking Water Regulations
""")
