import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# 앱 제목
st.set_page_config(page_title="Water Potability Predictor", layout="centered")
st.title("💧 수질 음용 가능성 예측기")
st.write("📈 CSV 데이터를 기반으로 머신러닝으로 물이 음용 가능한지를 예측합니다.")

# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("water_potability.csv")
    return df

df = load_data()

# 결측값 처리
imputer = SimpleImputer(strategy="mean")
X = df.drop("Potability", axis=1)
X_imputed = imputer.fit_transform(X)
y = df["Potability"]

# 데이터 분할 및 학습
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 정확도 표시
acc = accuracy_score(y_test, model.predict(X_test))
st.markdown(f"✅ **모델 정확도: {acc*100:.2f}%**")

# 시각화
st.subheader("📊 주요 특성 히스토그램")
selected_col = st.selectbox("히스토그램으로 볼 항목을 선택하세요", df.columns[:-1])
fig, ax = plt.subplots()
df[selected_col].hist(bins=30, ax=ax)
ax.set_title(f"{selected_col} 분포")
st.pyplot(fig)

# 사용자 입력
st.subheader("🔍 직접 입력하여 예측하기")

user_input = {}
for col in df.columns[:-1]:
    user_input[col] = st.number_input(f"{col}", min_value=0.0, step=0.1)

# 예측 버튼
if st.button("예측하기"):
    input_df = pd.DataFrame([user_input])
    input_imputed = imputer.transform(input_df)
    prediction = model.predict(input_imputed)[0]
    proba = model.predict_proba(input_imputed)[0][prediction]

    if prediction == 1:
        st.success(f"✅ 이 물은 음용 가능합니다. (신뢰도: {proba*100:.2f}%)")
    else:
        st.error(f"❌ 이 물은 음용에 적합하지 않습니다. (신뢰도: {proba*100:.2f}%)")
