import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 제목
st.title("Case 1: 외란 없음 – FGS-PID vs PID 제어 성능 비교")

# 1. Step Response 그래프 시각화
st.subheader("📈 Step Response (x-position vs Time)")

# 예시 데이터 (논문 기반)
time = np.linspace(0, 80, 100)
pid_response = 1.6 * np.exp(-0.05 * time) * np.sin(0.2 * time) + 0.75
fgs_pid_response = 1.3 * np.exp(-0.07 * time) * np.sin(0.2 * time) + 0.75

# 그래프
fig, ax = plt.subplots()
ax.plot(time, pid_response, label="PID", linestyle='--')
ax.plot(time, fgs_pid_response, label="FGS-PID")
ax.set_xlabel("Time (s)")
ax.set_ylabel("x position (m)")
ax.set_title("Step Response (CASE 1)")
ax.legend()
st.pyplot(fig)

# 2. 성능 비교표
st.subheader("📊 Step Response 성능 비교 지표")

performance_data = {
    "항목": ["Rise time", "Overshoot", "Settling time", "Mean power consumed"],
    "FGS-PID": ["3.1 s", "36 %", "19.8 s", "1.04E+02 kN·m/s"],
    "PID": ["3.3 s", "62 %", "52.8 s", "1.09E+02 kN·m/s"]
}
df = pd.DataFrame(performance_data)
st.table(df)
