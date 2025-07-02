# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# -------------------------------
# 📌 PID 및 FGS-PID 제어기 클래스 정의
# -------------------------------

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral = 0
        self.prev_error = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

class FGSPIDController(PIDController):
    def compute(self, error, dt):
        de = (error - self.prev_error) / dt if dt > 0 else 0

        # 간단한 퍼지 계수 조정 (tanh로 게인 보정)
        kp_adj = 1.0 + 0.3 * np.tanh(error)
        ki_adj = 1.0 + 0.2 * np.tanh(error)
        kd_adj = 1.0 + 0.2 * np.tanh(de)

        self.integral += error * dt
        output = (self.Kp * kp_adj) * error + (self.Ki * ki_adj) * self.integral + (self.Kd * kd_adj) * de
        self.prev_error = error
        return output

# -------------------------------
# 📌 시스템 동역학 정의 (2차 시스템 예시)
# -------------------------------

def leaf_rwcl_dynamics(y, t, controller, setpoint):
    y_pos, y_vel = y
    error = setpoint - y_pos
    u = controller.compute(error, dt=0.1)
    dydt = [y_vel, -2.0 * y_vel - 5.0 * y_pos + u]
    return dydt

# -------------------------------
# 📌 시뮬레이션 함수
# -------------------------------

def run_simulation(controller_type, label, setpoint):
    t = np.linspace(0, 50, 1000)
    y0 = [0.6, 0]  # 초기 RWCL = 0.6 (예: 60%)
    controller = controller_type(Kp=2.0, Ki=0.4, Kd=0.1)
    y = odeint(leaf_rwcl_dynamics, y0, t, args=(controller, setpoint))
    return t, y[:, 0], label

# -------------------------------
# 📌 Streamlit UI 시작
# -------------------------------

st.title("🌿 RWCL 제어 시뮬레이션: PID vs FGS-PID (CASE 1 - 외란 없음)")
st.markdown("**목표 RWCL: 0.75 (75%)**, 초기 RWCL: 0.60 (60%)")

setpoint = 0.75  # 목표 수분 함량 (75%)

# 시뮬레이션 실행
t_pid, y_pid, label_pid = run_simulation(PIDController, "PID", setpoint)
t_fgs, y_fgs, label_fgs = run_simulation(FGSPIDController, "FGS-PID", setpoint)

# -------------------------------
# 📈 시각화
# -------------------------------

fig, ax = plt.subplots()
ax.plot(t_pid, y_pid, '--', label='PID Controller')
ax.plot(t_fgs, y_fgs, '-', label='FGS-PID Controller')
ax.axhline(y=setpoint, color='gray', linestyle=':', label='Target RWCL (0.75)')
ax.set_xlabel("Time (s)")
ax.set_ylabel("RWCL (Relative Water Content in Leaves)")
ax.set_title("Step Response: RWCL Control (CASE 1)")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# -------------------------------
# 📊 성능 비교 지표
# -------------------------------

def get_step_metrics(t, y, setpoint):
    rise_time = t[next(i for i, v in enumerate(y) if v >= 0.9 * setpoint)]
    overshoot = (np.max(y) - setpoint) / setpoint * 100
    settling_idx = next((i for i in reversed(range(len(y))) if abs(y[i] - setpoint) > 0.02), None)
    settling_time = t[settling_idx] if settling_idx else t[-1]
    return rise_time, overshoot, settling_time

r1, o1, s1 = get_step_metrics(t_pid, y_pid, setpoint)
r2, o2, s2 = get_step_metrics(t_fgs, y_fgs, setpoint)

st.subheader("📊 Step Response 성능 비교")
st.table({
    "Metric": ["Rise Time (s)", "Overshoot (%)", "Settling Time (s)"],
    "PID": [f"{r1:.2f}", f"{o1:.1f}", f"{s1:.1f}"],
    "FGS-PID": [f"{r2:.2f}", f"{o2:.1f}", f"{s2:.1f}"]
})
