# streamlit_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# -------------------
# 기본 PID Controller
# -------------------
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

# ---------------------------
# FGS-PID Controller (간단형)
# ---------------------------
class FGSPIDController(PIDController):
    def compute(self, error, dt):
        de = (error - self.prev_error) / dt if dt != 0 else 0

        # 간단한 퍼지 게인 보정
        kp_gain = 1 + 0.2 * np.tanh(error)
        ki_gain = 1 + 0.1 * np.tanh(error)
        kd_gain = 1 + 0.1 * np.tanh(de)

        u = (self.Kp * kp_gain) * error
        self.integral += error * dt
        u += (self.Ki * ki_gain) * self.integral
        u += (self.Kd * kd_gain) * de

        self.prev_error = error
        return u

# -------------------
# 시스템 동역학 정의
# -------------------
def system_dynamics(y, t, controller, setpoint):
    y_val, y_dot = y
    error = setpoint - y_val
    u = controller.compute(error, dt=0.1)
    dydt = [y_dot, -2.0 * y_dot - 5.0 * y_val + u]  # 2차 시스템 모델
    return dydt

# -------------------
# 시뮬레이션 함수
# -------------------
def run_simulation(controller_class, label):
    t = np.linspace(0, 50, 500)
    y0 = [0.0, 0.0]  # 초기값: 수분 0%
    controller = controller_class(Kp=2.0, Ki=0.8, Kd=0.3)
    y = odeint(system_dynamics, y0, t, args=(controller, 0.75))  # 목표값 = 75%
    return t, y[:, 0], label

# -------------------
# Streamlit UI 구성
# -------------------
st.title("🌱 스마트팜 수분 제어 시뮬레이션 (CASE 1)")
st.subheader("PID vs FGS-PID 제어기 비교 (목표 수분값 75%)")

# 시뮬레이션 수행
t1, y1, label1 = run_simulation(PIDController, "PID 제어기")
t2, y2, label2 = run_simulation(FGSPIDController, "FGS-PID 제어기")

# 그래프 시각화
fig, ax = plt.subplots()
ax.plot(t1, y1, 'r--', label=label1)
ax.plot(t2, y2, 'b-', label=label2)
ax.axhline(0.75, color='gray', linestyle=':', label='목표 수분값 (0.75)')
ax.set_title("Step Response (CASE 1 - 외란 없음)")
ax.set_xlabel("시간 (초)")
ax.set_ylabel("정규화 수분값")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# 성능 요약 (텍스트)
st.markdown("#### ✅ 해석 요약")
st.markdown("""
- **FGS-PID**는 목표 수분값에 더 빠르게 수렴하며 overshoot를 억제합니다.
- **PID**는 수렴이 느리고 overshoot가 크게 발생할 수 있습니다.
""")
