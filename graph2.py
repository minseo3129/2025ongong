# 📁 파일명 예: streamlit_energy_analysis.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_excel("통계 인포그래픽 설문지.xlsx", sheet_name="설문지 응답 시트1")

st.title("🌎 신재생에너지 인식 설문조사 결과 분석")

# Q1. 연령대 분석
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

# Q3. 에너지 종류 순서 인식 분석
st.header("Q3. 신재생에너지 종류 인식 (정확한 순서 비율)")

# 정답 순서: 태양광, 바이오, 수력, 풍력, 지열
answer_order = ['태양광', '바이오', '수력', '풍력', '지열']

# 순서 칼럼 추출
order_cols = [col for col in df.columns if '다음 신재생에너지 종류' in col]

# 중복 응답 제외
df_unique = df[df[order_cols].nunique(axis=1) == 5]

# 정확히 일치한 순서 개수 세기
df_unique['정답개수'] = df_unique[order_cols].apply(lambda row: sum(row.values == answer_order), axis=1)
correct_counts = df_unique['정답개수'].value_counts().sort_index()

st.bar_chart(correct_counts)
st.markdown("💡 5개 전부 정답인 사람은 거의 없음 → 인식 부족 확인!")

# Q4. 단가 높아도 사용할 의향
st.header("Q4. 발전 단가가 높아도 신재생에너지 사용할 의향?")
q4_counts = df['신재생에너지의 발전 단가가 기존 에너지원(석탄, 석유, 가스 등)보다 높더라도 환경 보호를 위해 이를 적극적으로 사용할 의향이 있으신가요?'].value_counts()
st.bar_chart(q4_counts)

# Q5. 정보 접한 경험
st.header("Q5. 신재생에너지 관련 정보 접한 경험")
q5_counts = df['최근 1년간 정부와 지자체의 신재생에너지 정책, 에너지원의 장단점, 국내외 에너지 현황 등에 대한 정보를 접한 경험이 얼마나 있었나요?'].value_counts()
st.bar_chart(q5_counts)

# Q6. 접한 정보 매체
st.header("Q6. 정보를 접한 매체")
media_raw = df['접한 경험이 있다면, 주로 정보를 접한 매체는 무엇인가요?'].dropna()
media_split = media_raw.str.split(',|, ', expand=True).stack().str.strip().value_counts()
st.bar_chart(media_split)
