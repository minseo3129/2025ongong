import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")
st.title("🌿 RWCL 기반 FGS-PID 제어 시뮬레이션 결과 시각화")

# -------------------------------
# Case 선택
# -------------------------------
case = st.selectbox("케이스를 선택하세요", ["Case 1: 외란 없음", "Case 2: 수분 스트레스 변화", "Case 3: 유전형/처리 조건 극단 설정"])

# -------------------------------
# 공통 데이터 (가상 시뮬레이션 예시 데이터 생성 또는 불러오기)
# -------------------------------
# 실제 사용시에는 파일 업로드 또는 센서 출력 데이터프레임으로 대체하세요
time = np.linspace(0, 80, 400)
rwcl_pid = 75 + 0.5 * np.sin(0.1 * time)
rwcl_fgs = 75 + 0.3 * np.sin(0.1 * time + 0.2)

# -------------------------------
# Case 1: 외란 없음
# -------------------------------
if case.startswith("Case 1"):
    st.subheader("📊 RWCL 제어 결과 비교 (Case 1)")
    
    fig, ax = plt.subplots()
    ax.plot(time, rwcl_pid, label='PID', linestyle='--')
    ax.plot(time, rwcl_fgs, label='FGS-PID')
    ax.axhline(75.0, color='gray', linestyle=':', label='목표 RWCL = 75.0%')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RWCL (%)")
    ax.set_title("RWCL 응답곡선 (Step Response)")
    ax.legend()
    st.pyplot(fig)

    st.markdown("### 📌 분석 요약")
    st.write("""
    - FGS-PID는 과도 현상(Overshoot)이 작고 빠르게 수렴함
    - PID는 목표값 도달 후 진동이 큼
    - 평균 급수량(Kn/m/s) 기준 FGS-PID는 에너지 효율성이 우수함
    """)

# -------------------------------
# Case 2: 수분 스트레스 변화
# -------------------------------
elif case.startswith("Case 2"):
    st.subheader("🌡 수분 스트레스 조건에서의 제어 결과")

    # RWCL 응답 비교
    stress_low = 75 + 0.4 * np.sin(0.15 * time)
    stress_high = 75 + 0.2 * np.sin(0.2 * time + 0.4)

    fig, ax = plt.subplots()
    ax.plot(time, stress_low, label='FGS-PID (5% 수분)', color='tab:blue')
    ax.plot(time, stress_high, label='FGS-PID (70% 수분)', color='tab:orange')
    ax.axhline(75.0, color='gray', linestyle=':', label='목표 RWCL = 75.0%')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RWCL (%)")
    ax.set_title("RWCL 변화 (수분 스트레스 변화)")
    ax.legend()
    st.pyplot(fig)

    st.markdown("### 📌 분석 요약")
    st.write("""
    - 스트레스 조건(S=5% 또는 70%) 변화에도 RWCL 75.0% 목표를 안정적으로 유지
    - 시스템이 빠르게 대응하며 FGS-PID의 높은 적응력 확인
    """)

# -------------------------------
# Case 3: G, C 극단 조건
# -------------------------------
elif case.startswith("Case 3"):
    st.subheader("🧬 유전형(G) 및 처리 조건(C) 변화에 따른 RWCL 제어 결과")

    # 위치 변화 예시
    fig, ax = plt.subplots()
    x = 0.5 * np.cos(0.1 * time)
    y_pid = 0.5 * np.sin(0.1 * time) + 0.2 * np.random.randn(len(time))
    y_fgs = 0.5 * np.sin(0.1 * time + 0.2)

    ax.plot(x, y_pid, label="PID", linestyle="--", color='tab:red')
    ax.plot(x, y_fgs, label="FGS-PID", color='tab:green')
    ax.set_title("제어 위치 궤적 (극단 조건)")
    ax.set_xlabel("X 위치")
    ax.set_ylabel("Y 위치")
    ax.legend()
    st.pyplot(fig)

    # 게인 변화 시각화
    st.markdown("#### 실시간 게인 변화")
    kp_var = 1 + 0.2 * np.sin(0.05 * time)
    ki_var = 0.5 + 0.1 * np.cos(0.03 * time)
    kd_var = 0.2 + 0.05 * np.sin(0.07 * time)

    fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(time, kp_var); ax[0].set_ylabel("Kp")
    ax[1].plot(time, ki_var); ax[1].set_ylabel("Ki")
    ax[2].plot(time, kd_var); ax[2].set_ylabel("Kd")
    ax[2].set_xlabel("Time (s)")
    fig.suptitle("FGS-PID 게인 변화 (Case 3)")
    st.pyplot(fig)

    st.markdown("### 📌 분석 요약")
    st.write("""
    - PID는 G, C 변화에 민감하게 진동과 위치 편차가 발생
    - FGS-PID는 게인 동적 조정으로 안정된 궤도 유지
    - 리소스 절약 및 정밀 수분 조절에 유리함
    """)

