import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === 데이터 로딩 ===
@st.cache_data
def load_data():
    df = pd.read_csv("moniflora-backup-rtdb.csv")
    df = df[["time", "moisture", "temperature", "conductivity", "light"]].dropna()
    df["moisture"] = df["moisture"] / 100  # 정규화
    return df

df = load_data()
t = np.arange(len(df))
setpoint = 0.75  # 목표 수분값

# === PID 제어기 ===
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

# === FGS-PID 제어기 ===
class FGSPIDController(PIDController):
    def compute(self, error, dt):
        de = (error - self.prev_error) / dt
        # 퍼지 보정 (간단한 조정)
        kp_gain = 1 + 0.3 * np.tanh(error)
        ki_gain = 1 + 0.2 * np.tanh(error)
        kd_gain = 1 + 0.1 * np.tanh(de)
        self.integral += error * dt
        output = (self.Kp * kp_gain) * error + (self.Ki * ki_gain) * self.integral + (self.Kd * kd_gain) * de
        self.prev_error = error
        return output

# === 시뮬레이션 함수 ===
def simulate(controller_class, label, disturbances=None):
    dt = 1.0
    y = df["moisture"].iloc[0]
    outputs = []
    control_effort = []
    controller = controller_class(Kp=1.2, Ki=0.4, Kd=0.05)

    for i in range(len(df)):
        error = setpoint - y
        u = controller.compute(error, dt)
        d = 0
        if disturbances is not None:
            d = 0.005 * disturbances["temperature"].iloc[i] + \
                0.003 * disturbances["conductivity"].iloc[i] + \
                0.001 * disturbances["light"].iloc[i]
        y += 0.1 * u - d
        y = np.clip(y, 0, 1.2)
        outputs.append(y)
        control_effort.append(abs(u))
    return outputs, control_effort

# === 성능 지표 계산 ===
def compute_metrics(y, control_effort):
    y = np.array(y)
    try:
        rise_time = np.argmax(y >= 0.75)
    except:
        rise_time = -1
    overshoot = (np.max(y) - 0.75) / 0.75 * 100
    try:
        stable_idx = np.where(np.abs(y - 0.75) < 0.02)[0]
        settling_time = stable_idx[-1] if len(stable_idx) > 0 else -1
    except:
        settling_time = -1
    steady_state_error = abs(y[-1] - 0.75)
    mean_effort = np.mean(control_effort)
    return rise_time, overshoot, settling_time, steady_state_error, mean_effort

# === 시뮬레이션 실행 ===
def run_case(case_name, disturbances=None):
    pid_y, pid_u = simulate(PIDController, "PID", disturbances)
    fgs_y, fgs_u = simulate(FGSPIDController, "FGS-PID", disturbances)

    # 시각화
    fig, ax = plt.subplots()
    ax.plot(t, pid_y, "--", label="PID 제어기")
    ax.plot(t, fgs_y, "-", label="FGS-PID 제어기")
    ax.axhline(setpoint, color="gray", linestyle=":", label="목표 수분값 (0.75)")
    ax.set_title(f"{case_name} – 수분 제어 응답")
    ax.set_xlabel("시간 (tick)")
    ax.set_ylabel("수분값 (정규화)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # 성능 지표 테이블
    metrics = {
        "지표": ["Rise Time", "Overshoot (%)", "Settling Time", "Steady-State Error", "Mean Effort"],
        "PID": compute_metrics(pid_y, pid_u),
        "FGS-PID": compute_metrics(fgs_y, fgs_u)
    }
    st.markdown(f"### 📊 {case_name} 성능 비교 지표")
    st.table(pd.DataFrame(metrics))

# === Streamlit 레이아웃 ===
st.title("💧 스마트팜 수분 제어 시뮬레이션 (Case 1~3)")
case = st.selectbox("실험 케이스 선택", ["Case 1: 외란 없음", "Case 2: 환경 외란 적용", "Case 3: 유전형/처리조건 외란"])

if case == "Case 1: 외란 없음":
    st.markdown("**외란 없이 고정 PID vs FGS-PID 비교**")
    run_case("Case 1")
elif case == "Case 2: 환경 외란 적용":
    st.markdown("**환경 센서 외란 (온도, 광량 등) 추가**")
    disturbance = df[["temperature", "conductivity", "light"]]
    run_case("Case 2", disturbances=disturbance)
elif case == "Case 3: 유전형/처리조건 외란":
    st.markdown("**극단 외란: 높은 온도 + 전도도 조합**")
    temp = df["temperature"].copy()
    cond = df["conductivity"].copy()
    light = df["light"].copy()
    temp.iloc[:50] += 5   # G 조건 가정
    cond.iloc[:50] += 10  # C 조건 가정
    disturbances = pd.DataFrame({"temperature": temp, "conductivity": cond, "light": light})
    run_case("Case 3", disturbances=disturbances)
