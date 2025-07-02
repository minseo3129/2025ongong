import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 데이터 로딩
df = pd.read_csv("Advanced Soybean Agricultural Dataset.csv")
rwcl_data = df["Relative Water Content in Leaves (RWCL)"].dropna().values

# ---- PID Controller ----
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral = 0
        self.prev_error = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt != 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# ---- FGS-PID Controller ----
class FGSPIDController(PIDController):
    def compute(self, error, dt):
        de = (error - self.prev_error) / dt if dt != 0 else 0
        kp_gain = 1 + 0.2 * np.tanh(error)
        ki_gain = 1 + 0.1 * np.tanh(error)
        kd_gain = 1 + 0.1 * np.tanh(de)
        u = (self.Kp * kp_gain) * error
        self.integral += error * dt
        u += (self.Ki * ki_gain) * self.integral
        u += (self.Kd * kd_gain) * de
        self.prev_error = error
        return u

# ---- Plant dynamics ----
def plant_dynamics(y, t, controller, setpoint):
    y_val, y_dot = y
    error = setpoint - y_val
    u = controller.compute(error, dt=0.1)
    dydt = [y_dot, -2.0*y_dot - 5.0*y_val + u]
    return dydt

# ---- 시뮬레이션 실행 ----
def run_simulation(controller_class, y0_value):
    t = np.linspace(0, 50, 500)
    y0 = [y0_value, 0.0]
    controller = controller_class(Kp=2.0, Ki=1.0, Kd=0.3)
    sol = odeint(plant_dynamics, y0, t, args=(controller, 0.75))
    return t, sol[:, 0]

# ---- Streamlit UI ----
st.title("CASE 1: 고정 PID vs FGS-PID 제어 성능 비교")
st.subheader("목표 RWCL: 0.75 (외란 없음)")

# 초기값 설정
init_rwcl = float(st.slider("초기 RWCL (0~1)", min_value=0.3, max_value=0.8, value=0.55, step=0.01))

# 시뮬레이션
t, pid_output = run_simulation(PIDController, init_rwcl)
_, fgs_pid_output = run_simulation(FGSPIDController, init_rwcl)

# 결과 시각화
fig, ax = plt.subplots()
ax.plot(t, pid_output, 'r--', label="고정 PID")
ax.plot(t, fgs_pid_output, 'b-', label="FGS-PID")
ax.axhline(0.75, color='gray', linestyle=':', label="목표 RWCL = 0.75")
ax.set_xlabel("시간 (s)")
ax.set_ylabel("RWCL (%)")
ax.set_title("RWCL 제어 시뮬레이션 (Case 1)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# 성능 비교 지표
def compute_metrics(y, t, setpoint=0.75):
    rise_time = t[next(i for i, val in enumerate(y) if val >= 0.9 * setpoint)]
    overshoot = (np.max(y) - setpoint) / setpoint * 100
    settling_idx = next(i for i in reversed(range(len(y))) if abs(y[i] - setpoint) > 0.05)
    settling_time = t[settling_idx]
    return rise_time, overshoot, settling_time

r1, o1, s1 = compute_metrics(pid_output, t)
r2, o2, s2 = compute_metrics(fgs_pid_output, t)

st.subheader("📊 제어 성능 비교")
st.table({
    "지표": ["Rise Time", "Overshoot (%)", "Settling Time"],
    "고정 PID": [f"{r1:.2f}", f"{o1:.1f}", f"{s1:.1f}"],
    "FGS-PID": [f"{r2:.2f}", f"{o2:.1f}", f"{s2:.1f}"]
})
