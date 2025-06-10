import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

st.title("💧 GitHub 수질 데이터 분석기")

# GitHub URL 입력받기
url = st.text_input("🔗 GitHub에서 CSV Raw 링크 입력:", 
                    "https://raw.githubusercontent.com/yourname/repo-name/main/water_potability.csv")

if url:
    try:
        df = pd.read_csv(url)
        st.success("✅ 데이터 불러오기 성공")
        st.write(df.head())

        # 학습 준비
        X = df.drop("Potability", axis=1)
        y = df["Potability"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        model = DecisionTreeClassifier(max_depth=4)
        model.fit(X_train, y_train)

        st.subheader("🌲 결정 트리 시각화")
        fig, ax = plt.subplots(figsize=(14, 6))
        plot_tree(model, feature_names=X.columns, class_names=["Not Potable", "Potable"], filled=True)
        st.pyplot(fig)

        st.subheader("🧪 직접 수질 수치 입력 → 예측")
        user_input = {}
        for feature in X.columns:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())
            user_input[feature] = st.slider(feature, min_val, max_val, mean_val)

        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        st.markdown(f"🎯 예측 결과: **{'✅ 마실 수 있음' if prediction == 1 else '🚫 마실 수 없음'}**")

    except Exception as e:
        st.error(f"❌ 데이터 불러오기 실패: {e}")
