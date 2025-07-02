# streamlit_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import find_peaks

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

# ---- FGS-PID Controller ---- (simple fuzzified gain adjustment)
class FGSPIDController(PIDController):
    def compute(self, error, dt):
        de = (error - self.prev_error) / dt if dt != 0 else 0

        # Fuzzy Gain Tuning - very basic linear mapping
        kp_gain = 1 + 0.2 * np.tanh(error)
        ki_gain = 1 + 0.1 * np.tanh(error)
        kd_gain = 1 + 0.1 * np.tanh(de)

        u = (self.Kp * kp_gain) * error
        self.integral += error * dt
        u += (self.Ki * ki_gain) * self.integral
        u += (self.Kd * kd_gain) * de

        self.prev_error = error
        return u

# ---- 2nd-order system dynamics (e.g. RWCL response) ----
def system_dynamics(y, t, controller, setpoint):
    y_val, y_dot = y
    error = setpoint - y_val
    u = controller.compute(error, dt=0.1)
    dydt = [y_dot, -2.0*y_dot - 5.0*y_val + u]
    return dydt

# ---- Run Simulation ----
def simulate(controller_class, label):
    t = np.linspace(0, 80, 800)
    y0 = [0.0, 0.0]
    controller = controller_class(Kp=1.5, Ki=0.5, Kd=0.1)
    y = odeint(system_dynamics, y0, t, args=(controller, 1.0))
    return t, y[:, 0], label

# ---- Performance Metrics ----
def compute_metrics(t, y, setpoint=1.0):
    rise_time = t[next(i for i, val in enumerate(y) if val >= 0.9 * setpoint)]
    overshoot = (np.max(y) - setpoint) / setpoint * 100
    settling_idx = next(i for i in range(len(y)-1, 0, -1) if abs(y[i] - setpoint) > 0.05)
    settling_time = t[settling_idx]
    return rise_time, overshoot, settling_time

# ---- Streamlit UI ----
st.title("FGS-PID vs 고정 PID 시뮬레이션 (CASE 1 - 외란 없음)")
st.subheader("📊 Step Response: RWCL 목표 수분 함량 제어")

# 시뮬레이션
t1, y1, label1 = simulate(PIDController, "PID")
t2, y2, label2 = simulate(FGSPIDController, "FGS-PID")

# 그래프 시각화
fig, ax = plt.subplots()
ax.plot(t1, y1, '--', label=label1)
ax.plot(t2, y2, '-', label=label2)
ax.set_title("Step Response (CASE 1)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("RWCL (%)")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# 성능 비교
r1, o1, s1 = compute_metrics(t1, y1)
r2, o2, s2 = compute_metrics(t2, y2)

st.markdown("### 📈 성능 비교 지표")
st.table({
    "Metric": ["Rise Time", "Overshoot (%)", "Settling Time (s)"],
    "PID": [f"{r1:.2f}", f"{o1:.1f}", f"{s1:.1f}"],
    "FGS-PID": [f"{r2:.2f}", f"{o2:.1f}", f"{s2:.1f}"]
})
