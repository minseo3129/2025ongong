import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------
# 데이터 불러오기
# ---------------------
df = pd.read_csv("Advanced Soybean new.csv")
rwcl_data = df["Relative Water Content in Leaves (RWCL)"].values.astype(float)

# ---------------------
# 목표값 및 시간 설정
# ---------------------
target_rwcl = 750.0
time = np.linspace(0, 80, len(rwcl_data))

# ---------------------
# 고정 PID 제어기
# ---------------------
def fixed_pid_controller(y, target, Kp=0.015, Ki=0.001, Kd=0.005):
    e_prev = 0
    integral = 0
    output = []
    for i in range(len(y)):
        e = target - y[i]
        integral += e
        derivative = e - e_prev
        control = Kp * e + Ki * integral + Kd * derivative
        y[i] += control
        output.append(y[i])
        e_prev = e
    return np.array(output)

# ---------------------
# FGS-PID 제어기 (퍼지 게인 가변)
# ---------------------
def fgs_pid_controller(y, target, Kp=0.015, Ki=0.001, Kd=0.005):
    e_prev = 0
    integral = 0
    output = []
    for i in range(len(y)):
        e = target - y[i]
        de = e - e_prev
        integral += e

        # 퍼지 기반 게인 조절 (단순 모델)
        Kp_fuzzy = Kp * (1 + 0.5 * np.tanh(abs(e) / 100))
        Ki_fuzzy = Ki * (1 + 0.3 * np.tanh(abs(integral) / 1000))
        Kd_fuzzy = Kd * (1 + 0.4 * np.tanh(abs(de) / 100))

        control = Kp_fuzzy * e + Ki_fuzzy * integral + Kd_fuzzy * de
        y[i] += control
        output.append(y[i])
        e_prev = e
    return np.array(output)

# ---------------------
# 시뮬레이션 실행
# ---------------------
rwcl_initial = rwcl_data.copy()
rwcl_pid = fixed_pid_controller(rwcl_data.copy(), target_rwcl)
rwcl_fgs = fgs_pid_controller(rwcl_initial.copy(), target_rwcl)

# ---------------------
# Streamlit 시각화
# ---------------------
st.title("🌱 Case 1: 고정 PID vs FGS-PID 제어 시뮬레이션")
st.subheader("목표 RWCL: 750.0")

fig, ax = plt.subplots()
ax.plot(time, rwcl_data, label="초기 RWCL", linestyle="--", alpha=0.4)
ax.plot(time, rwcl_pid, label="PID", linestyle="--")
ax.plot(time, rwcl_fgs, label="FGS-PID", linestyle="-")
ax.axhline(y=target_rwcl, color='gray', linestyle=':', label="목표 RWCL")
ax.set_xlabel("Time (s)")
ax.set_ylabel("RWCL")
ax.set_title("Step Response (CASE 1)")
ax.legend()
st.pyplot(fig)

# ---------------------
# 성능 비교
# ---------------------
def compute_metrics(signal, target):
    rise_time_idx = next((i for i, val in enumerate(signal) if val >= target), len(signal)-1)
    overshoot = max(signal) - target
    settling_idx = next((i for i, val in enumerate(signal[::-1]) if abs(val - target) > 5), 0)
    settling_time = len(signal) - settling_idx
    mean_control = np.mean(np.abs(np.diff(signal)))
    return rise_time_idx, overshoot, settling_time, mean_control

r_pid = compute_metrics(rwcl_pid, target_rwcl)
r_fgs = compute_metrics(rwcl_fgs, target_rwcl)

st.subheader("📊 제어 성능 비교 (Case 1)")
perf_df = pd.DataFrame({
    "지표": ["Rise Time", "Overshoot", "Settling Time", "Mean Control Effort"],
    "PID": [f"{r_pid[0]} s", f"{r_pid[1]:.2f}", f"{r_pid[2]} s", f"{r_pid[3]:.2f}"],
    "FGS-PID": [f"{r_fgs[0]} s", f"{r_fgs[1]:.2f}", f"{r_fgs[2]} s", f"{r_fgs[3]:.2f}"]
})
st.dataframe(perf_df)
