import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 불러오기
@st.cache_data
def load_data():
    return pd.read_csv("Advanced Soybean new.csv")

df = load_data()

# 2. 실험 변수 정의
target_values = {
    'ChlorophyllA663': 5.0,
    'Number of Pods (NP)': 40,
    'Protein Percentage (PPE)': 20.0,
    'Relative Water Content in Leaves (RWCL)': 75.0,
    'Number of Seeds per Pod (NSP)': 2.0
}

variables = list(target_values.keys())

st.title("🌱 FGS-PID 기반 스마트팜 생육 제어 시뮬레이션")

# 3. 변수 선택
selected_var = st.selectbox("제어할 생육 변수 선택", variables)
target = target_values[selected_var]

# 4. 간단한 퍼지 기반 게인 조정 함수
def fuzzy_gain(error, d_error):
    def fuzz(val):
        if abs(val) < 0.5:
            return 1.0
        elif abs(val) < 1.5:
            return 1.2
        else:
            return 1.5
    return fuzz(error), fuzz(error), fuzz(d_error)

# 5. FGS-PID 시뮬레이션 함수
def run_fgs_pid_simulation(data, target, Kp=1.0, Ki=0.1, Kd=0.05):
    actual = []
    control = []
    error_history = []
    integral = 0
    prev_error = 0
    output = data.iloc[0]  # 초기 상태

    for i in range(30):  # 30회 시뮬레이션 반복
        error = target - output
        d_error = error - prev_error
        integral += error

        # 퍼지 게인 적용
        Kp_w, Ki_w, Kd_w = fuzzy_gain(error, d_error)

        u = (Kp*Kp_w)*error + (Ki*Ki_w)*integral + (Kd*Kd_w)*d_error
        output += u * 0.1  # 시스템 반응
        output = np.clip(output, 0, 100)

        actual.append(output)
        control.append(u)
        error_history.append(error)
        prev_error = error

    return actual, control, error_history

# 6. 데이터 준비
avg_data = df[selected_var].mean()

# 7. 시뮬레이션 실행
actual, control, error = run_fgs_pid_simulation(pd.Series([avg_data]), target)

# 8. 시각화
st.subheader(f"📊 {selected_var}에 대한 FGS-PID 제어 시뮬레이션 결과")

fig, ax = plt.subplots()
ax.plot(actual, label="실제 값", marker='o')
ax.axhline(target, color='r', linestyle='--', label="목표값")
ax.set_ylabel(selected_var)
ax.set_xlabel("시간 (Iteration)")
ax.set_title("FGS-PID 제어 시뮬레이션")
ax.legend()
st.pyplot(fig)

# 오차 그래프
fig2, ax2 = plt.subplots()
ax2.plot(error, color='orange', marker='x', label="오차 (e)")
ax2.set_title("오차 변화 추이")
ax2.set_xlabel("시간")
ax2.set_ylabel("오차")
ax2.legend()
st.pyplot(fig2)

# 제어 입력 시각화
fig3, ax3 = plt.subplots()
ax3.plot(control, color='green', label="제어 입력 (Control Effort)")
ax3.set_title("제어 입력 추이")
ax3.set_xlabel("시간")
ax3.set_ylabel("수분/처리량 조절")
ax3.legend()
st.pyplot(fig3)

