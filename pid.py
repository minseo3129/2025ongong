import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("Advanced Soybean Agricultural Dataset.csv")
    return df['Relative Water Content in Leaves (RWCL)'].dropna().values

data = load_data()

# --- Controllers ---
class PID:
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

class FGSPID(PID):
    def compute(self, error, dt):
        de = (error - self.prev_error) / dt
        kp_gain = 1 + 0.2 * np.tanh(error)
        ki_gain = 1 + 0.1 * np.tanh(error)
        kd_gain = 1 + 0.1 * np.tanh(de)
        self.integral += error * dt
        output = (self.Kp * kp_gain) * error + (self.Ki * ki_gain) * self.integral + (self.Kd * kd_gain) * de
        self.prev_error = error
        return output

# --- System Model (RWCL Response) ---
def model(y, t, controller, setpoint):
    y_val, y_dot = y
    error = setpoint - y_val
    u = controller.compute(error, dt=0.1)
    dydt = [y_dot, -2 * y_dot - 5 * y_val + u]
    return dydt

def simulate(controller_type, label, setpoint=0.75):
    t = np.linspace(0, 80, 800)
    y0 = [data[0], 0]
    controller = controller_type(1.5, 0.4, 0.1)
    y = odeint(model, y0, t, args=(controller, setpoint))
    return t, y[:, 0], label

# --- Performance Metrics ---
def compute_metrics(t, y, setpoint=0.75):
    rise_time = t[next(i for i, val in enumerate(y) if val >= 0.9 * setpoint)]
    overshoot = (np.max(y) - setpoint) / setpoint * 100
    settling_idx = next(i for i in reversed(range(len(y))) if abs(y[i] - setpoint) > 0.05)
    settling_time = t[settling_idx]
    return rise_time, overshoot, settling_time

# --- Streamlit UI ---
st.title("FGS-PID vs 고정 PID 제어 시뮬레이션 (RWCL 목표: 0.75)")
st.subheader("📊 Case 1: 외란 없음 – Step Response 비교")

t1, y1, label1 = simulate(PID, "PID")
t2, y2, label2 = simulate(FGSPID, "FGS-PID")

# --- Plotting ---
fig, ax = plt.subplots()
ax.plot(t1, y1, 'r--', label=label1)
ax.plot(t2, y2, 'b-', label=label2)
ax.set_xlabel("Time (s)")
ax.set_ylabel("RWCL (%)")
ax.set_title("Step Response (CASE 1)")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# --- Metrics Table ---
r1, o1, s1 = compute_metrics(t1, y1)
r2, o2, s2 = compute_metrics(t2, y2)

st.markdown("### 📈 제어 성능 비교 (Step Response Metrics)")
st.table({
    "Metric": ["Rise Time (s)", "Overshoot (%)", "Settling Time (s)"],
    "PID": [f"{r1:.1f}", f"{o1:.1f}", f"{s1:.1f}"],
    "FGS-PID": [f"{r2:.1f}", f"{o2:.1f}", f"{s2:.1f}"]
})
