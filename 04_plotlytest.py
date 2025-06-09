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
)))import streamlit as st
import pandas as pd
import plotly.express as px
import re

# ✅ GitHub에서 CSV 데이터 불러오기
gender_df = pd.read_csv(
    "https://raw.githubusercontent.com/minseo3129/2025ongong/main/people_gender.csv",
    encoding="cp949"
)

# ▶ 행정구역 이름만 추출
gender_df['행정구역명'] = gender_df['행정구역'].apply(lambda x: x.split('  ')[0])

# ▶ 지역 선택 위젯
regions = gender_df['행정구역명'].unique().tolist()
selected_region = st.selectbox("지역을 선택하세요", regions[1:])  # '전국' 제외

# ▶ 남/여 연령 컬럼 추출
male_cols = [col for col in gender_df.columns if '남_' in col and '세' in col]
female_cols = [col for col in gender_df.columns if '여_' in col and '세' in col]

# ✅ 연령 숫자 안전 추출 함수
def extract_age(col_name):
    match = re.search(r'(\d+)', col_name)
    try:
        return int(match.group(1)) if match else None
    except:
        return None

# ▶ 연령 리스트 만들기 (None 제거 + 정수형만 유지 + 정렬)
age_list = sorted(set(
    age for age in [extract_age(col) for col in male_cols] if isinstance(age, int)
))

# ▶ 슬라이더 생성 (안전 검사)
if len(age_list) >= 2:
    min_age_val = min(age_list)
    max_age_val = max(age_list)
    default_range = (min_age_val, max_age_val)

    min_age, max_age = st.slider(
        "🎚️ 연령 범위를 선택하세요",
        min_value=min_age_val,
        max_value=max_age_val,
        value=default_range
    )
else:
    st.error("⚠️ 연령 데이터를 불러오는 데 실패했습니다.")
    st.stop()

# ▶ 인구 피라미드 시각화 함수
def create_pyramid(region_name, age_min, age_max):
    row = gender_df[gender_df['행정구역명'] == region_name]
    if row.empty:
        return None

    male_values = row[male_cols].iloc[0].apply(lambda x: int(str(x).replace(',', '')))
    female_values = row[female_cols].iloc[0].apply(lambda x: int(str(x).replace(',', '')))

    filtered = [(a, m, f) for a, m, f in zip(age_list, male_values, female_values) if age_min <= a <= age_max]
    if not filtered:
        return None

    age_labels = [a for a, _, _ in filtered]
    male_pop = [-m for _, m, _ in filtered]
    female_pop = [f for _, _, f in filtered]

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
    st.warning("⚠️ 해당 조건에 맞는 인구 데이터를 찾을 수 없습니다.")

