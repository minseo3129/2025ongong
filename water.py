import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load model and imputer
model = joblib.load("water_model.pkl")
imputer = joblib.load("water_imputer.pkl")

# 페이지 설정
st.set_page_config(page_title="음용수 예측 시스템", layout="centered")
st.title("💧 물의 음용 가능성 예측 시스템")
st.write("이 웹앱은 수질 데이터를 기반으로 해당 물이 음용 가능한지 예측합니다.")

# 수질 지표 시각화 (탐색적 분석)
st.subheader("📊 수질 지표 시각화")

uploaded_file = st.file_uploader("water_potability.csv 파일을 업로드하세요", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    # 그래프 시각화
    fig, ax = plt.subplots()
    df["ph"].hist(ax=ax, bins=30)
    ax.set_title("pH 분포")
    ax.set_xlabel("pH")
    ax.set_ylabel("빈도")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    df.boxplot(column="Conductivity", by="Potability", ax=ax2)
    ax2.set_title("전도도 vs 음용 가능 여부")
    ax2.set_ylabel("Conductivity")
    st.pyplot(fig2)

# 사용자 입력
st.subheader("🔍 수질 항목 직접 입력")

ph = st.number_input("pH (6.5~8.5 권장)", min_value=0.0, max_value=14.0, step=0.1)
hardness = st.number_input("Hardness", min_value=0.0)
solids = st.number_input("Total Dissolved Solids", min_value=0.0)
chloramines = st.number_input("Chloramines", min_value=0.0)
sulfate = st.number_input("Sulfate", min_value=0.0)
conductivity = st.number_input("Conductivity", min_value=0.0)
organic_carbon = st.number_input("Organic Carbon", min_value=0.0)
trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0)
turbidity = st.number_input("Turbidity", min_value=0.0)

if st.button("예측하기"):
    user_data = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate,
                               conductivity, organic_carbon, trihalomethanes, turbidity]],
                             columns=["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
                                      "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"])
    user_data_imputed = pd.DataFrame(imputer.transform(user_data), columns=user_data.columns)
    prediction = model.predict(user_data_imputed)[0]
    
    if prediction == 1:
        st.success("✅ 이 물은 음용 가능합니다.")
    else:
        st.error("❌ 이 물은 음용에 적합하지 않습니다.")
