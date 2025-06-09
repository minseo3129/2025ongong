import streamlit as st
import pandas as pd
import plotly.express as px
import re

# ✅ GitHub에서 데이터 불러오기 (CP949 인코딩)
gender_df = pd.read_csv(
    "https://raw.githubusercontent.com/minseo3129/2025ongong/main/people_gender.csv",
    encoding="cp949"
)

# ▶ 행정구역 이름만 추출 ('서울특별시  (1100000000)' → '서울특별시')
gender_df['행정구역명'] = gender_df['행정구역'].apply(lambda x: x.split('  ')[0])

# ▶ 지역 선택 위젯
regions = gender_df['행정구역명'].unique().tolist()
selected_region = st.selectbox("지역을 선택하세요", regions[1:])  # '전국' 제외

# ▶ 연령 관련 컬럼 추출
male_cols = [col for col in gender_df.columns if '남_' in col and '세' in col]
female_cols = [col for col in gender_df.columns if '여_' in col and '세' in col]

# ✅ 정규표현식으로 연령 숫자만 안전하게 추출
def extract_age(col_name):
    match = re.search(r'(\d+)', col_name)
    return int(match.group(1)) if match else None

# ✅ 연령 숫자만 모아서 중복 제거 후 정렬
ages = sorted(list(set(
    [extract_age(col) for col in male_cols if extract_age(col) is not None]
)))

# ▶ 연령대 슬라이더 위젯 (유효성 검사)
if ages:
    min_age, max_age = st.slider("📊 연령 범위를 선택하세요!", min(ages), max(ages), (min(ages), max(ages)))
else:
    st.error("⚠️ 연령 데이터를 불러오는 데 실패했습니다.")
    st.stop()

# ▶ 인구 피라미드 생성 함수
def create_pyramid(region_name, age_min, age_max):
    row = gender_df[gender_df['행정구역명'] == region_name]
    if row.empty:
        return None

    male_values = row[male_cols].iloc[0].apply(lambda x: int(str(x).replace(',', '')))
    female_values = row[female_cols].iloc[0].apply(lambda x: int(str(x).replace(',', '')))

    # 선택한 연령대 범위에 해당하는 데이터만 추출
    age_filtered = [(a, m, f) for a, m, f in zip(ages, male_values, female_values) if age_min <= a <= age_max]
    if not age_filtered:
        return None

    age_labels = [a for a, _, _ in age_filtered]
    male_pop = [-]()_

