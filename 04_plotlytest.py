import streamlit as st
import pandas as pd
import plotly.express as px
import re

st.set_page_config(layout="wide")  # 화면 넓게

# ✅ 데이터 불러오기 (GitHub CSV, CP949 인코딩)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/minseo3129/2025ongong/main/people_gender.csv"
    return pd.read_csv(url, encoding="cp949")

gender_df = load_data()

# ▶ 행정구역 이름 정리 ('서울특별시  (1100000000)' → '서울특별시')
gender_df['행정구역명'] = gender_df['행정구역'].apply(lambda x: x.split('  ')[0])

# ▶ 지역 선택
regions = gender_df['행정구역명'].unique().tolist()
selected_region = st.selectbox("📍 지역을 선택하세요", regions[1:])  # '전국' 제외

# ▶ 남/여 연령 컬럼 추출
male_cols = [col for col in gender_df.columns if '남_' in col and '세' in col]
female_cols = [col for col in gender_df.columns if '여_' in col and '세' in col]

# ✅ 연령 숫자 추출 함수 (정규표현식 기반)
def extract_age(col_name):
    match = re.search(r'(\d+)', col_name)
    return int(match.group(1)) if match else None

# ▶ 연령 리스트 생성
age_list = sorted(set(
    age for age i
