import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 🌱 기본 설정
st.set_page_config(page_title="식물 생장 분산 분석", layout="wide")
st.title("🌱 같은 조건, 다른 성장")
st.markdown("**식물 생장 결과의 분산 분석을 통한 스마트팜의 불안정성 해소 방안 탐색**")


# 내장 데이터 로드
df = pd.read_csv("plant_growth_data.csv")
features = df.columns[:-1]

# 🔧 전처리
df['Condition_Group'] = df['Soil_Type'] + "_" + df['Water_Frequency'] + "_" + df['Fertilizer_Type']
group_stats = df.groupby('Condition_Group')['Growth_Milestone'].agg(['mean', 'var', 'count']).reset_index()
group_stats = group_stats[group_stats['count'] > 2]
df = df.merge(group_stats[['Condition_Group', 'var']], on='Condition_Group', how='left')

# 🔍 상위 분산 그룹
top_var_groups = group_stats.sort_values('var', ascending=False).head(5)
st.subheader("📊 생장 결과 분산이 큰 상위 조건 그룹")
st.dataframe(top_var_groups)

# 📈 분포 시각화
st.subheader("🎯 Growth_Milestone 분포 (Top 5 High Variance Groups)")
unstable_df = df[df['Condition_Group'].isin(top_var_groups['Condition_Group'])]
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=unstable_df, x='Condition_Group', y='Growth_Milestone', ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# 🔍 변수 비교
st.subheader("📋 불안정 vs 안정 그룹의 환경 변수 비교")
low_var_groups = group_stats.sort_values('var').head(5)
compare_groups = pd.concat([
    df[df['Condition_Group'].isin(top_var_groups['Condition_Group'])].assign(Var_Level='High Variance'),
    df[df['Condition_Group'].isin(low_var_groups['Condition_Group'])].assign(Var_Level='Low Variance')
])

for col in ['Sunlight_Hours', 'Temperature', 'Humidity']:
    st.markdown(f"**{col} 비교**")
    fig, ax = plt.subplots()
    sns.boxplot(data=compare_groups, x='Var_Level', y=col, ax=ax)
    st.pyplot(fig)

# 🧠 변수 중요도 분석
st.subheader("📌 변수 중요도 분석 (Random Forest)")
X = df[['Sunlight_Hours', 'Temperature', 'Humidity']]
y = df['Growth_Milestone']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
st.dataframe(importance)

st.markdown("**➡️ 위 분석을 통해 불안정한 그룹에서 조절 우선순위가 높은 환경 변수를 식별할 수 있습니다.**")
