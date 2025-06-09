import streamlit as st
import pandas as pd
import plotly.express as px
import re

gender_df = pd.read_csv(
    "https://raw.githubusercontent.com/minseo3129/2025ongong/main/people_gender.csv",
    encoding="cp949"
)



# ▶ 행정구역 이름만 추출 ('서울특별시  (1100000000)' → '서울특별시')
gender_df['행정구역명'] = gender_df['행정구역'].apply(lambda x: x.split('  ')[0])

# ▶ 지역 선택 위젯
regions = gender_df['행정구역명'].unique().tolist()
selected_region = st.selectbox("지역을 선택하세요", regions[1:])  # '전국' 제외

# ▶ 연령 추출 및 필터링
male_cols = [col for col in gender_df.columns if '남_' in col and '세' in col]
female_cols = [col for col in gender_df.columns if '여_' in col and '세' in col]

# ✅ 정규표현식으로 안전하게 연령 숫자 추출
def extract_age(col_name):
    match = re.search(r'(\d+)', col_name)
    return int(match.group(1)) if match else 100

ages = [extract_age(col) for col in male_cols]

# ▶ 연령대 선택 위젯
# 연령 숫자 추출 (중복 제거 + 정렬)
ages = sorted(list(set([extract_age(col) for col in male_cols if extract_age(col) is not None])))

# 슬라이더 (데이터 없으면 종료)
if ages:
    min_age, max_age = st.slider("📊 연령 범위를 선택하세요!", min(ages), max(ages), (min(ages), max(ages)))
else:
    st.error("연령 정보를 불러오지 못했습니다.")
    st.stop()


# ▶ 인구 피라미드 생성 함수
def create_pyramid(region_name, age_min, age_max):
    row = gender_df[gender_df['행정구역명'] == region_name]
    if row.empty:
        return None

    male_values = row[male_cols].iloc[0].apply(lambda x: int(str(x).replace(',', '')))
    female_values = row[female_cols].iloc[0].apply(lambda x: int(str(x).replace(',', '')))

    age_filtered = [(a, m, f) for a, m, f in zip(ages, male_values, female_values) if age_min <= a <= age_max]
    if not age_filtered:
        return None

    age_labels = [a for a, _, _ in age_filtered]
    male_pop = [-m for _, m, _ in age_filtered]  # 음수: 왼쪽
    female_pop = [f for _, _, f in age_filtered]

    df_plot = pd.DataFrame({
        '연령': age_labels + age_labels,
        '인구수': male_pop + female_pop,
        '성별': ['남자'] * len(age_labels) + ['여자'] * len(age_labels)
    })

    fig = px.bar(
        df_plot,
        x='인구수',
        y='연령',
        color='성별',
        orientation='h',
        title=f"{region_name} 인구 피라미드 ({age_min}세 ~ {age_max}세)",
        height=600
    )
    return fig

# ▶ 실행
fig = create_pyramid(selected_region, min_age, max_age)
if fig:
    st.plotly_chart(fig)
else:
    st.warning("해당 조건에 맞는 인구 데이터를 찾을 수 없습니다.")

