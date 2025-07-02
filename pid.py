import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("Advanced Soybean new.csv")
    return df

df = load_data()

st.title("🌱 FGS-PID 제어 기반 콩 생육 제어 시뮬레이션")
st.markdown("스마트팜 환경에서 생육 변수(PID 오차)를 자동 제어하는 퍼지 기반 제어 구조 시뮬레이션")

# 선택 가능한 생육 변수
target_vars = ['Number of Pods (NP)', 'Biological Weight (BW)', 'Sugars (Su)',
               'Relative Water Content in Leaves (RWCL)', 'ChlorophyllA663',
               'Chlorophyllb649', 'Protein Percentage (PPE)', 'Weight of 300 Seeds (W3S)',
               'Leaf Area Index (LAI)', 'Number of Seeds per Pod (NSP)']

# 사용자 입력
target_var = st.selectbox("🎯 제어할 생육 변수 선택", target_vars)
target_value = st.slider(f"📍 목표 {target_var} 값", 
                         min_value=float(df[target_var].min()), 
                         max_value=float(df[target_var].max()), 
                         value=float(df[target_var].mean()))

st.write(f"목표: {target_var} = {target_value:.2f}")

# 오차 및 오차 변화율 계산
df["Error"] = target_value - df[target_var]
df["dError"] = df["Error"].diff().fillna(0)

# 간단한 퍼지 규칙 기반 Kp 설정
def fuzzy_kp(error, derror):
    if abs(error) > 1.5:
        return 0.8
    elif abs(error) > 0.5:
        return 0.5
    else:
        return 0.2

df["Kp"] = df.apply(lambda row: fuzzy_kp(row["Error"], row["dError"]), axis=1)
df["Control_Signal"] = df["Kp"] * df["Error"]
df["Adjusted_Water_Input"] = np.clip(0.5 + df["Control_Signal"] / 10, 0, 1)

# 시각화
st.subheader("📈 생육 변수 vs 제어 효과 시각화")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df[target_var], label="실제 생육값", color='green')
ax.axhline(target_value, color='red', linestyle='--', label="목표값")
ax.set_ylabel(target_var)
ax.set_title("생육 변수 제어 시뮬레이션 결과")
ax.legend()
st.pyplot(fig)

# 제어 입력 시각화
st.subheader("💧 조정된 수분 투입량")
fig2, ax2 = plt.subplots(figsize=(10, 3))
ax2.plot(df["Adjusted_Water_Input"], label="Adjusted Water Input", color='blue')
ax2.set_ylabel("Water Input (0 ~ 1)")
ax2.set_xlabel("샘플 번호")
ax2.set_title("퍼지 제어에 따른 급수량 조정")
st.pyplot(fig2)

# 결과 요약
st.success("✅ 제어 시뮬레이션 완료")
st.write("간단한 퍼지 규칙을 기반으로 FGS-PID 제어 시뮬레이션이 수행되었습니다.")
