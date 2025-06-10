import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 1. 파일 업로드
st.title("🌍 오염물질 상관 그래프 탐색기")
uploaded_file = st.file_uploader("📂 Air_Quality.csv 파일을 업로드하세요", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # 2. 유효한 오염물질 열 추출
    pollutants = ['CO', 'CO2', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']
    pollutants = [col for col in pollutants if col in df.columns]

    st.success(f"데이터 불러오기 성공! 📊 분석할 오염물질: {', '.join(pollutants)}")

    # 3. 상관행렬 계산
    corr = df[pollutants].corr()

    # 4. 그래프 생성
    G = nx.Graph()
    for i in range(len(pollutants)):
        for j in range(i+1, len(pollutants)):
            p1, p2 = pollutants[i], pollutants[j]
            weight = corr.loc[p1, p2]
            G.add_edge(p1, p2, weight=round(weight, 2))

    # 5. 그래프 시각화
    st.subheader("🔗 오염물질 간 상관관계 그래프")
    pos = nx.spring_layout(G, seed=42)
    edge_weights = nx.get_edge_attributes(G, 'weight')

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, font_size=14)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights, font_color='red')
    st.pyplot(plt)

    # 6. 사용자 상호작용
    st.subheader("🔎 특정 오염물질의 상관관계 조회")
    selected = st.selectbox("기체 선택", pollutants)
    st.write("### 📈 상관관계 목록")
    st.dataframe(corr[selected].sort_values(ascending=False).round(3))

else:
    st.info("먼저 CSV 파일을 업로드하세요.")