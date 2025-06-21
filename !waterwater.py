import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
import plotly.graph_objects as go

# -------------------------------
# 1. 데이터 로딩 및 정제
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
        "Fluoride_mg/L": "Fluoride"
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
# 2. Sample Site 유사도 계산
# -------------------------------
st.title("🧭 Sample Site 유사도 기반 연결 시각화")
st.caption("📈 networkx 없이 Plotly로 연결 시각화 구현")

features = ["Turbidity", "Chlorine", "Coliform", "Ecoli"]
site_avg = df.groupby("Sample_Site")[features].mean()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(site_avg)
distance_matrix = euclidean_distances(scaled_data)

site_labels = site_avg.index.tolist()
threshold = st.slider("🔗 연결 임계 거리", 0.1, 2.0, 0.6)

# -------------------------------
# 3. 임의 배치 기반 시각화 레이아웃 생성
# -------------------------------
np.random.seed(42)
positions = {site: (np.cos(i * 2 * np.pi / len(site_labels)),
                    np.sin(i * 2 * np.pi / len(site_labels)))
             for i, site in enumerate(site_labels)}

# -------------------------------
# 4. Plotly 그래프 구성
# -------------------------------
edge_x = []
edge_y = []
for i in range(len(site_labels)):
    for j in range(i + 1, len(site_labels)):
        if distance_matrix[i, j] < threshold:
            x0, y0 = positions[site_labels[i]]
            x1, y1 = positions[site_labels[j]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

node_x = [positions[site][0] for site in site_labels]
node_y = [positions[site][1] for site in site_labels]

fig = go.Figure()
fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                         line=dict(width=1, color='gray'), hoverinfo='none'))

fig.add_trace(go.Scatter(
    x=node_x, y=node_y, mode='markers+text',
    marker=dict(size=12, color='skyblue'),
    text=site_labels, textposition="top center"
))

fig.update_layout(
    title="🧬 Sample Site 간 유사 연결 구조 (거리 기반)",
    showlegend=False,
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False),
    height=700
)

st.plotly_chart(fig, use_container_width=True)

