pip install streamlit scikit-fuzzy pandas matplotlib



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_csv("Advanced Soybean new.csv")
    return df

df = load_data()

st.title("🌿 FGS-PID 기반 스마트팜 생육 제어 시뮬레이션")

# 1. 사용자 입력
st.sidebar.header("🎯 목표 생육 조건")
target_vars = {
    'ChlorophyllA663': st.sidebar.slider("엽록소 A663 (≥)", 0.0, 10.0, 5.0),
    'NP': st.sidebar.slider("꼬투리 수 (≥)", 0, 100, 40),
    'PPE': st.sidebar.slider("단백질 비율 (%) (≥)", 0.0, 50.0, 20.0),
    'RWCL': st.sidebar.slider("잎의 수분 함량 (%) (≥)", 0.0, 100.0, 75.0),
    'NSP': st.sidebar.slider("생산된 종자 수 (≥)", 0.0, 4.0, 2.0),
}

st.sidebar.header("⚙️ 제어 방법 선택")
control_method = st.sidebar.selectbox("제어 방식", ["고정 PID", "FGS-PID"])

# 2. 간단한 오차 기반 제어 모델링
def simulate_pid(var_name, y0, ref, method='FGS-PID', steps=50):
    Kp, Ki, Kd = 0.6, 0.4, 0.2
    y = [y0]
    e_prev = ref - y0
    integral = 0

    for t in range(steps):
        e = ref - y[-1]
        integral += e
        derivative = e - e_prev

        if method == 'FGS-PID':
            # 퍼지 가중치: 예시 적용
            e_level = np.clip(e / ref, -1, 1)
            de_level = np.clip(derivative / ref, -1, 1)

            # 퍼지 규칙 기반 간단한 weight (실제 퍼지 시스템은 더 정교함)
            kp_weight = 1 + 0.5 * np.sign(e_level)
            ki_weight = 1 + 0.3 * np.sign(e_level)
            kd_weight = 1 - 0.2 * np.sign(e_level)

            Kp_adj = Kp * kp_weight
            Ki_adj = Ki * ki_weight
            Kd_adj = Kd * kd_weight
        else:
            Kp_adj, Ki_adj, Kd_adj = Kp, Ki, Kd

        u = Kp_adj * e + Ki_adj * integral + Kd_adj * derivative
        y_new = y[-1] + 0.05 * u  # 시스템 반응 모델 (단순화)
        y.append(y_new)
        e_prev = e

    return y

# 3. 시뮬레이션 실행 및 결과 시각화
st.subheader(f"🧪 시뮬레이션 결과: {control_method}")
fig, ax = plt.subplots(figsize=(10, 5))

for var in target_vars:
    ref = target_vars[var]
    y0 = df[var].mean()
    y_sim = simulate_pid(var, y0, ref, method=control_method)
    ax.plot(y_sim, label=f"{var} (목표: {ref})")

ax.axhline(0, color='gray', linestyle='--')
ax.set_xlabel("시간 단계")
ax.set_ylabel("생육 변수 값")
ax.set_title(f"{control_method} 제어 시뮬레이션 결과")
ax.legend()
st.pyplot(fig)

# 4. 원본 평균 비교
if st.checkbox("📊 생육 변수 평균 비교"):
    avg_vals = df[list(target_vars)].mean()
    st.write("📘 데이터셋 평균 생육값")
    st.dataframe(avg_vals)

