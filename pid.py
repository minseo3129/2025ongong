import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 🎯 목표값 정의
TARGET_VALUES = {
    'ChlorophyllA663': 5.0,
    'Number of Pods (NP)': 40,
    'Protein Percentage (PPE)': 20.0,
    'Relative Water Content in Leaves (RWCL)': 75.0,
    'Number of Seeds per Pod (NSP)': 2.0
}

# 🧪 퍼지 게인 함수 (간단한 오차 기반 조정)
def fuzzy_gain(e, de):
    def weight(x):
        if abs(x) < 0.5:
            return 1.0
        elif abs(x) < 1.5:
            return 1.2
        else:
            return 1.5
    return weight(e), weight(e), weight(de)

# 🚀 PID 시뮬레이션 (FGS or Fixed)
def simulate_pid(y0, ref, method='FGS-PID', steps=30):
    Kp, Ki, Kd = 0.6, 0.4, 0.2
    y, u, e_list = [y0], [], []
    integral, e_prev = 0, ref - y0

    for _ in range(steps):
        e = ref - y[-1]
        integral += e
        de = e - e_prev

        if method == 'FGS-PID':
            kp_w, ki_w, kd_w = fuzzy_gain(e, de)
        else:
            kp_w, ki_w, kd_w = 1, 1, 1

        u_t = (Kp * kp_w) * e + (Ki * ki_w) * integral + (Kd * kd_w) * de
        y_t = y[-1] + 0.05 * u_t

        u.append(u_t)
        y.append(y_t)
        e_list.append(e)
        e_prev = e

    return y, u, e_list

# 📏 평가 지표 계산 함수
def calculate_metrics(y, u, ref, tolerance=0.05):
    y = np.array(y)
    t = np.arange(len(y))
    rise_time = next((i for i, val in enumerate(y) if val >= ref), None)
    overshoot = np.max(y) - ref
    settling_idx = next((i for i in range(len(y)-1, -1, -1)
                         if abs(y[i] - ref) > ref * tolerance), 0)
    settling_time = settling_idx + 1
    steady_state_error = abs(y[-1] - ref)
    mean_effort = np.mean(np.abs(u))
    return rise_time, overshoot, settling_time, steady_state_error, mean_effort

# 📊 Streamlit UI
st.title("🌾 FGS-PID vs 고정 PID 제어 성능 비교 (환경 외란 없음)")

# 🔘 변수 선택
var = st.selectbox("분석할 생육 변수", list(TARGET_VALUES.keys()))
ref = TARGET_VALUES[var]

# 데이터 평균값으로 초기화
df = pd.read_csv("Advanced Soybean new.csv")
y0 = df[var].mean()

# 시뮬레이션 실행
y_fgs, u_fgs, e_fgs = simulate_pid(y0, ref, method="FGS-PID")
y_pid, u_pid, e_pid = simulate_pid(y0, ref, method="Fixed")

# 평가 지표 계산
metrics_fgs = calculate_metrics(y_fgs, u_fgs, ref)
metrics_pid = calculate_metrics(y_pid, u_pid, ref)

# 🧾 표로 출력
cols = ['Rise Time', 'Overshoot', 'Settling Time', 'Steady State Error', 'Mean Control Effort']
st.subheader("📋 평가 지표 비교")
df_metrics = pd.DataFrame([metrics_pid, metrics_fgs], columns=cols, index=["Fixed PID", "FGS-PID"])
st.dataframe(df_metrics)

# 📈 시각화
st.subheader("📈 제어 출력 비교")
fig, ax = plt.subplots()
ax.plot(y_pid, label='Fixed PID', linestyle='--', marker='x')
ax.plot(y_fgs, label='FGS-PID', linestyle='-', marker='o')
ax.axhline(ref, color='gray', linestyle=':', label='Target')
ax.set_title(f"{var} 시뮬레이션 결과")
ax.set_ylabel(var)
ax.set_xlabel("Time Step")
ax.legend()
st.pyplot(fig)

