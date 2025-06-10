import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# 페이지 설정
st.set_page_config(page_title="물의 음용 가능성 판단 트리 분석기", layout="wide")
st.title("🌳 트리 자료구조 기반 음용 가능성 영향 요인 분석 및 예측")

# 1. 데이터 불러오기 및 전처리
df = pd.read_csv("water_potability.csv")
imputer = SimpleImputer(strategy="mean")
X = df.drop("Potability", axis=1)
y = df["Potability"]
X_imputed = imputer.fit_transform(X)

# 2. 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)

# 3. 시각화 1: 결정 트리 구조
st.subheader("🌳 결정 트리 구조 시각화")
fig, ax = plt.subplots(figsize=(18, 10))
plot_tree(tree_model,
          feature_names=df.columns[:-1],
          class_names=["음용 불가", "음용 가능"],
          filled=True,
          rounded=True,
          fontsize=10,
          ax=ax)
st.pyplot(fig)

# 4. 시각화 2: 변수 중요도 바그래프
st.subheader("📌 어떤 항목이 음용 여부에 가장 큰 영향을 주었을까?")
importance_df = pd.DataFrame({
    "Feature": df.columns[:-1],
    "Importance": tree_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

# 한글 이름으로 변환
feature_labels = {
    "ph": "pH",
    "Hardness": "경도",
    "Solids": "총 용존 고형물(물 속 무기물 + 유기물의 총량)",
    "Chloramines": "클로라민",
    "Sulfate": "황산염",
    "Conductivity": "전기전도도",
    "Organic_carbon": "유기 탄소",
    "Trihalomethanes": "트리할로메탄",
    "Turbidity": "탁도"
}
importance_df["Feature_KR"] = importance_df["Feature"].map(feature_labels)

fig2, ax2 = plt.subplots()
sns.barplot(data=importance_df, x="Importance", y="Feature_KR", palette="viridis", ax=ax2)
ax2.set_title("각 항목의 중요도 (의사결정트리 기준)")
st.pyplot(fig2)

# 5. 사용자 입력을 통한 예측
st.subheader("🧪 수질 항목을 입력해 예측해보세요")

# 입력값 받기
input_data = {}
for col in df.columns[:-1]:
    label = feature_labels.get(col, col)
    input_data[col] = st.number_input(f"{label}", min_value=0.0, step=0.1)

# 예측 실행
if st.button("예측하기"):
    user_df = pd.DataFrame([input_data])
    user_df_imputed = pd.DataFrame(imputer.transform(user_df), columns=user_df.columns)
    prediction = tree_model.predict(user_df_imputed)[0]
    prob = tree_model.predict_proba(user_df_imputed)[0][prediction]

    st.markdown("### ❓ 이 물은 음용해도 될까요?")
    if prediction == 1:
        st.success(f"🟢 **예**, 이 물은 음용 가능합니다. (신뢰도: {prob*100:.2f}%)")
    else:
        st.error(f"🔴 **아니요**, 이 물은 음용에 적합하지 않습니다. (신뢰도: {prob*100:.2f}%)")

# 6. 설명
st.markdown("""
---
📘 **설명**
- 트리 시각화는 모델이 어떤 수질 항목을 가장 먼저 분기 기준으로 사용하는지 보여줍니다.
- 예: `pH < 6.8` → 산성 → Potability=0 가능성 ↑
- `중요 변수 그래프`는 예측에 가장 영향을 많이 준 요소를 시각적으로 보여줍니다.
- `총 용존 고형물`은 물 속에 녹아 있는 무기물·유기물의 총량을 뜻합니다.
""")
