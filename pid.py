import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint

# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("Advanced Soybean Agricultural Dataset.csv")
    df = df.rename(columns=lambda x: x.strip())  # 공백 제거
    return df

data = load_data()
rwcl_values = data["Relative Water Content in Leaves (RWCL)"].values

# ---- PID Controller ----
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral = 0
        self.prev_error = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# ---- FGS-PID Controller ----
class FGSPIDController(PIDController):
    def compute(self, error, dt):
        de = (error - self.prev_error) / dt
        kp_gain = 1 + 0.4 * np.tanh(error)
        ki_gain = 1 + 0.2 * np.tanh(error)
        kd_gain = 1 + 0.2 * np.tanh(de)

        u = (self.Kp * kp_gain) * error
        self.integral += error * dt
        u += (self.Ki * ki_gain) * self.integral
        u += (self.Kd * kd_gain) * de

        self.prev_error = error
        return u

# ---- Dynamics Function ----
def rwcl_dynamics(y, t, controller, setpoint):
    y_val, y_dot = y
    error = setpoint - y_val
    u = controller.compute(error, dt=0.1)
    dydt = [y_dot, -2*y_dot - 5*y_val + u]
    return dydt

# ---- Simulation ----
def simulate(controller_class, label, y0_val):
    t = np.linspace(0, 80, 800)
    y0 = [y0_val, 0.0]
    controller = controller_class(Kp=1.5, Ki=0.5, Kd=0.1)
    y = odeint(rwcl_dynamics, y0, t, args=(controller, 0.75))
    return t, y[:, 0], label

# ---- 성능 지표 ----
def evaluate_performance(t, y, setpoint=0.75):
    rise_time = t[next(i for i, val in enumerate(y) if val >= 0.9 * setpoint)]
    overshoot = (np.max(y) - setpoint) / setpoint * 100
    settling_idx = next(i for i in range(len(y) - 1, 0, -1) if abs(y[i] - setpoint) > 0.05)
    settling_time = t[settling_idx]
    return rise_time, overshoot, settling_time

# ---- Streamlit 앱 시작 ----
st.title("FGS-PID vs PID RWCL 제어 시뮬레이션")
st.subheader("🎯 목표 수분 함량 (RWCL) = 0.75")

initial_rwcl = float(np.median(rwcl_values))
st.write(f"📌 평균 초기 RWCL 값: {initial_rwcl:.3f}")

# 시뮬레이션 실행
t1, y1, label1 = simulate(PIDController, "PID", initial_rwcl)
t2, y2, label2 = simulate(FGSPIDController, "FGS-PID", initial_rwcl)

# ---- 시각화 ----
fig, ax = plt.subplots()
ax.plot(t1, y1, 'r--', label=label1)
ax.plot(t2, y2, 'b-', label=label2)
ax.set_title("Step Response (CASE 1)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("RWCL")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ---- 성능 지표 비교 ----
r1, o1, s1 = evaluate_performance(t1, y1)
r2, o2, s2 = evaluate_performance(t2, y2)

st.markdown("### 📈 성능 지표 비교")
st.table({
    "Metric": ["Rise Time", "Overshoot (%)", "Settling Time (s)"],
    "PID": [f"{r1:.2f}", f"{o1:.1f}", f"{s1:.1f}"],
    "FGS-PID": [f"{r2:.2f}", f"{o2:.1f}", f"{s2:.1f}"]
})
