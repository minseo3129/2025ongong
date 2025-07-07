# streamlit_energy_analysis.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_excel("통계 인포그래픽 설문지.xlsx", sheet_name="설문지 응답 시트1")

st.title("🌎 신재생에너지 인식 설문조사 결과 분석")

# Q1. 연령대 분포
st.header("Q1. 연령대 분포")
age_counts = df['귀하의 연령대를 표시해주세요.'].value_counts().sort_index()
st.bar_chart(age_counts)

# Q2. 신재생에너지 전력 비중 인식
st.header("Q2. 신재생에너지 비중 인식 (정답 vs 오답)")

def check_answer(ans):
    return '정답' if '10~20%' in ans else '오답'

df['Q2 정답 여부'] = df['현재 대한민국 전체 전력 생산량에서 신재생에너지가 차지하는 비율은 어느 정도라고 생각하시나요?'].apply(check_answer)
q2_counts = df['Q2 정답 여부'].value_counts()
st.bar_chart(q2_counts)
