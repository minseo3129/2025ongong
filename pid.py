# case1_pid_vs_fgs_pid.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 목표 RWCL
RWCL_target = 75.0

# 고정 PID 제어기 파라미터
Kp_fixed = 0.6
Ki_fixed = 0.2
Kd_fixed = 0.05

# FGS-PID 보정 함수 (간단화된 rule)
def fuzzy_gain_modifiers(e, de_dt):
    return (1 + 0.5 * np.tanh(e)), (1 + 0.3 * np.tanh(de_dt)), (1 + 0.2 * np.tanh(e * de_dt))

# PID 제어기
def pid_controller(e, e_sum, e_diff, Kp, Ki, Kd):
    return Kp * e + Ki * e_sum + Kd * e_diff

# 시스템 모델 (RWCL 응답 모델: 간단한 1차 지연 시스템 가정)
def rwcl_system(y, t, u):
    tau = 10.0
    dydt = (-y + u) / tau
    return dydt

# 시뮬레이션 함수
def run_simulation(mode="PID"):
    t = np.linspace(0, 80, 800)
    y = np.zeros_like(t)
    u = np.zeros_like(t)
    e_sum = 0.0
    e_prev = 0.0
    RWCL_init = 65.0
    y[0] = RWCL_init

    for i in range(1, len(t)):
        e = RWCL_target - y[i-1]
        de = (e - e_prev)
        e_sum += e * (t[i] - t[i-1])
        e_prev = e

        if mode == "PID":
            u[i] = pid_controller(e, e_sum, de, Kp_fixed, Ki_fixed, Kd_fixed)
        elif mode == "FGS-PID":
            Kp_fz, Ki_fz, Kd_fz = fuzzy_gain_modifiers(e, de)
            u[i] = pid_controller(e, e_sum, de,
                                  Kp_fixed * Kp_fz,
                                  Ki_fixed * Ki_fz,
                                  Kd_fixed * Kd_fz)

        dy = rwcl_system(y[i-1], t[i-1], u[i])
        y[i] = y[i-1] + dy * (t[i] - t[i-1])

    return t, y, u

# Streamlit 시각화
st.title("Case 1: 고정 PID vs FGS-PID 제어 비교 시뮬레이션")
st.markdown("목표 잎 수분 함량 (RWCL): **75.0%**, 초기값: 65.0%")

t, y_pid, u_pid = run_simulation(mode="PID")
t, y_fgs, u_fgs = run_simulation(mode="FGS-PID")

fig, ax = plt.subplots()
ax.plot(t, y_pid, label="Fixed PID", linestyle="--")
ax.plot(t, y_fgs, label="FGS-PID", linestyle="-")
ax.axhline(RWCL_target, color='gray', linestyle=":", label="Target RWCL")
ax.set_xlabel("Time (s)")
ax.set_ylabel("RWCL (%)")
ax.set_title("RWCL Response: PID vs FGS-PID")
ax.legend()
st.pyplot(fig)


