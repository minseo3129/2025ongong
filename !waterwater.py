import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# 1. 데이터 불러오기 및 정제
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Drinking_Water_Quality_Distribution_Monitoring_Data.csv")
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")
    df = df.rename(columns={
        "Residual_Free_Chlorine_mg/L": "Chlorine",
        "Turbidity_NTU": "Turbidity",
        "Coliform_Quanti-Tray_MPN_/100mL": "Coliform",
        "E.coliQuanti-Tray_MPN/100mL": "Ecoli",
        "Fluoride_mg/L": "Fluoride",
        "Sample_Date": "Date"
    })
    for col in ["Sample_class", "Sample_Class"]:
        if col in df.columns:
            df = df.rename(columns={col: "Sample_Class"})
            break

    def to_num(x):
        if isinstance(x, str):
            x = x.replace("<", "").replace(">", "").replace("+", "").strip()
            if x.lower() == "nd":
                return np.nan
        try:
            return float(x)
        except:
            return np.nan

    for col in ["Turbidity", "Chlorine", "Coliform", "Ecoli"]:
        df[col] = df[col].apply(to_num)

    df = df.dropna(subset=["Sample_Site", "Turbidity", "Chlorine", "Coliform", "Ecoli"])
    return df

df = load_data()

# -------------------------------
# 2. Sample Site 간 유사도 기반 그래프 구성
# -------------------------------
st.title("🌐 NYC 수질 네트워크 그래프: Sample Site 유사도 기반 연결")
st.caption("📊 수질 유사도에 기반한 그래프 구조 분석 + DFS/BFS 탐색")

# 평균 수질 계산
features = ["Turbidity", "Chlorine", "Coliform", "Ecoli"]
site_mean = df.groupby("Sample_Site")[features].mean()

# 유사도 거리 계산 (정규화 후 유클리드 거리)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(site_mean)
dist_matrix = euclidean_distances(scaled)
sites = site_mean.index.tolist()

# 거리 기반 연결 그래프 만들기 (임계값 이하만 연결)
threshold = st.slider("🎯 유사도 연결 임계값 (작을수록 가까움)", 0.0, 1.5, 0.6)
G = nx.Graph()
for i, site_i in enumerate(sites):
    for j, site_j in enumerate(sites):
        if i < j and dist_matrix[i, j] < threshold:
            G.add_edge(site_i, site_j, weight=round(dist_matrix[i, j], 2))

st.markdown(f"🔗 총 연결 수: **{G.number_of_edges()}**, 노드 수: **{G.number_of_nodes()}**")

# -------------------------------
# 3. DFS/BFS 탐색 결과 출력
# -------------------------------
selected_node = st.selectbox("🧭 시작 지점을 선택하세요", sorted(G.nodes))
method = st.radio("탐색 방식", ["DFS (깊이 우선 탐색)", "BFS (너비 우선 탐색)"])

if method == "DFS (깊이 우선 탐색)":
    visited = list(nx.dfs_preorder_nodes(G, source=selected_node))
else:
    visited = list(nx.bfs_tree(G, source=selected_node))

st.write(f"🔍 탐색 경로: {visited}")

# -------------------------------
# 4. 그래프 시각화
# -------------------------------
pos = nx.spring_layout(G, seed=42)  # 위치 고정
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

node_x = [pos[node][0] for node in G.nodes]
node_y = [pos[node][1] for node in G.nodes]

fig = go.Figure()
fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1), hoverinfo='none'))
fig.add_trace(go.Scatter(
    x=node_x, y=node_y, mode='markers+text',
    marker=dict(size=10, color='skyblue'),
    text=list(G.nodes), textposition="top center"
))
fig.update_layout(title="📍 Sample Site 유사도 기반 그래프", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 5. 트리 기반 이상 구간 감지 예시
# -------------------------------
st.subheader("🌲 트리 구조 기반 이상지점 클러스터 확인")
components = list(nx.connected_components(G))
st.write(f"총 연결된 서브그래프 수: **{len(components)}**")
for i, comp in enumerate(components[:3]):
    st.write(f"서브그래프 {i+1}: {sorted(list(comp))}")

