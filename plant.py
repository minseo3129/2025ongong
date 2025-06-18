import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ 가장 먼저 페이지 설정
st.set_page_config(layout="wide")
st.title("🌱 식물 생장 분포 분석을 통한 스마트팜의 리스크 기반 작물관리 전략 : 30129 김민서")

# ✅ 폰트 및 스타일 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

# ✅ 데이터 불러오기
df = pd.read_csv("plant_growth_data.csv")
df["Failure"] = 1 - df["Growth_Milestone"]

# ✅ 변수 이름 매핑
name_map = {
    "Sunlight_Hours": "햇빛 노출 시간",
    "Temperature": "온도",
    "Humidity": "습도"
}

# 📊 1. 생장 성공/실패군의 주요 변수 분포
st.subheader("1. 생장 성공/실패군의 주요 변수 분포")
for feature in name_map:
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="Failure", y=feature, palette="pastel", ax=ax)
    ax.set_title(f"{name_map[feature]}에 따른 생장 성공/실패 분포", fontsize=14)
    ax.set_xlabel("성공(0) / 실패(1)", fontsize=12)
    ax.set_ylabel(name_map[feature], fontsize=12)
    st.pyplot(fig)

# 📊 2. 조건별 생장 결과의 분산 분석
st.subheader("2. 조건별 생장 결과의 분산 분석")

# 조건 조합 문자열 생성
df["조건조합"] = df["Soil_Type"] + " | " + df["Water_Frequency"] + " | " + df["Fertilizer_Type"]

# 그룹별 통계량 계산
group_stats = df.groupby("조건조합")["Growth_Milestone"].agg(['mean', 'var', 'std', 'count']).reset_index()
group_stats = group_stats.rename(columns={
    '조건조합': '조건 조합',
    'mean': '평균 생장값',
    'var': '분산',
    'std': '표준편차',
    'count': '샘플 수'
})
# 샘플 수가 충분한 조건만 필터링
filtered = group_stats[group_stats['샘플 수'] >= 3].sort_values(by='분산', ascending=False)

# 📋 분산값 기준 상위 5개 조건 테이블
st.markdown("### 🔍 분산값 기준 상위 불안정 조건")
st.dataframe(filtered.head(5), use_container_width=True)

# 📈 상위 조건의 생장률 분포 박스플롯
st.markdown("### 📊 상위 불안정 조건의 생장 결과 분포")
top_conditions = filtered.head(5)['조건 조합'].tolist()
subset = df[df["조건조합"].isin(top_conditions)]

fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=subset, x="조건조합", y="Growth_Milestone", palette="coolwarm", ax=ax)
ax.set_title("조건 조합별 생장 결과 분포 (Top 5 분산)", fontsize=14)
ax.set_xlabel("조건 조합", fontsize=12)
ax.set_ylabel("Growth_Milestone (생장률)", fontsize=12)
plt.xticks(rotation=45)
st.pyplot(fig)

# ℹ️ 분석 요약 메시지
st.info("👆 분산이 클수록 동일 조건에서도 생장 성공/실패의 결과 편차가 큽니다. 불안정한 조건으로 관리 우선이 필요합니다.")