import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import squarify

# CSV 불러오기
df = pd.read_csv("통계 인포그래픽 설문지 - 설문지 응답 시트1.csv")

st.set_page_config(page_title="신재생에너지 설문 결과", layout="wide")
st.title("🔋 신재생에너지 인식 실태조사 결과 시각화")

# Q2 - 신재생에너지 비중 인식 (정답: 10~20%)
st.header("Q2. 신재생에너지가 전력 생산에서 차지하는 비율 인식")
q2 = df["현재 대한민국 전체 전력 생산량에서 신재생에너지가 차지하는 비율은 어느 정도라고 생각하시나요?"]
correct_range = ['10~20%', '10~15%', '15~20%']
correct = q2.isin(correct_range).sum()
incorrect = len(q2) - correct
fig1, ax1 = plt.subplots()
ax1.pie([correct, incorrect], labels=["정답", "오답"], autopct="%1.1f%%", startangle=90)
ax1.axis("equal")
st.pyplot(fig1)

# Q3 - 신재생에너지 사용 순위
st.header("Q3. 많이 사용된다고 생각하는 신재생에너지 순위")
order_cols = [col for col in df.columns if "많이 사용된다고 생각하는 순서대로" in col]
correct_order = ['태양광', '바이오', '수력', '풍력', '지열']
position_scores = {source: 0 for source in correct_order}

for i, col in enumerate(order_cols):
    for val in df[col].dropna():
        val = val.strip()
        if val in position_scores:
            position_scores[val] += (5 - i)  # 가중치 부여

sorted_positions = dict(sorted(position_scores.items(), key=lambda x: -x[1]))
fig2, ax2 = plt.subplots()
ax2.bar(sorted_positions.keys(), sorted_positions.values())
ax2.set_ylabel("선호 순위 점수")
st.pyplot(fig2)

# Q4 - 신재생에너지 사용 의향
st.subheader("Q4. 환경 보호를 위해 신재생에너지를 사용할 의향이 있나요?")
q4 = df["신재생에너지의 발전 단가가 기존 에너지원(석탄, 석유, 가스 등)보다 높더라도 환경 보호를 위해 이를 적극적으로 사용할 의향이 있으신가요?"]
q4_counts = q4.value_counts()
fig3, ax3 = plt.subplots()
ax3.plot(q4_counts.index, q4_counts.values, marker="o")
ax3.set_ylabel("응답 수")
st.pyplot(fig3)

# Q5 - 정보 접촉 경험
st.header("Q5. 1년간 관련 정보 접촉 경험")
q5 = df["최근 1년간 정부와 지자체의 신재생에너지 정책, 에너지원의 장단점, 국내외 에너지 현황 등에 대한 정보를 접한 경험이 얼마나 있었나요?"]
q5_counts = q5.value_counts()
fig4, ax4 = plt.subplots()
ax4.pie(q5_counts.values, labels=q5_counts.index, autopct="%1.1f%%")
ax4.axis("equal")
st.pyplot(fig4)

# Q6 - 접촉 매체 (트리맵)
st.header("Q6. 정보를 접한 주요 매체")
media_raw = df["접한 경험이 있다면, 주로 정보를 접한 매체는 무엇인가요?"].dropna()
media_counts = {}
for entry in media_raw:
    for item in entry.split(","):
        key = item.strip()
        media_counts[key] = media_counts.get(key, 0) + 1

fig5, ax5 = plt.subplots()
squarify.plot(sizes=media_counts.values(), label=media_counts.keys(), alpha=0.8)
ax5.axis("off")
st.pyplot(fig5)

# Q7 - 교육 효과
st.header("Q7. 학교 교육이 인식에 도움이 되었는가?")
q7 = df["학교 교육과정에서 배운 신재생에너지 관련 내용이 신재생에너지에 대한 인식을 높이는 데 도움이 되었다고 생각하시나요?"]
fig6, ax6 = plt.subplots()
ax6.bar(q7.value_counts().index, q7.value_counts().values)
ax6.set_ylabel("응답 수")
st.pyplot(fig6)

# Q8 - 캠페인 참여 경험
st.header("Q8. 캠페인이나 제도 참여 여부")
q8 = df["신재생에너지 확대를 위한 참여형 캠페인이나 인센티브 제도(예: ‘에너지의 날‘ 소등, 소형 태양광 발전 설치 시 지원금 지급 등)에 지속적으로 참여하시나요?"]
fig7, ax7 = plt.subplots()
ax7.pie(q8.value_counts().values, labels=q8.value_counts().index, autopct="%1.1f%%")
ax7.axis("equal")
st.pyplot(fig7)

# Q9-1 - 참여 이유
st.header("Q9-1. 참여 지속 이유 (중복 응답)")
q9_1 = df["참여형 캠페인이나 인센티브 제도에 지속적으로 참여하게 된 가장 큰 이유는 무엇인가요? (복수 응답 가능)"].dropna()
reason_counts = {}
for entry in q9_1:
    for item in entry.split(","):
        key = item.strip()
        reason_counts[key] = reason_counts.get(key, 0) + 1

fig8, ax8 = plt.subplots()
ax8.barh(list(reason_counts.keys()), list(reason_counts.values()))
st.pyplot(fig8)

# Q9-2 - 인식 향상 여부
st.header("Q9-2. 참여 후 인식 향상 여부")
q9_2 = df["위 제도에 참여하여 신재생에너지에 대한 인식이 향상되었다고 생각하시나요?"].dropna()
fig9, ax9 = plt.subplots()
ax9.bar(q9_2.value_counts().index, q9_2.value_counts().values)
st.pyplot(fig9)

# Q10-1 - 불참 이유
st.header("Q10-1. 캠페인 불참 이유")
q10_1 = df["참여형 캠페인이나 인센티브 제도 참여를 중단했거나 참여하지 않은 이유는 무엇인가요? (복수 응답 가능)"].dropna()
nonreason_counts = {}
for entry in q10_1:
    for item in entry.split(","):
        key = item.strip()
        nonreason_counts[key] = nonreason_counts.get(key, 0) + 1

fig10, ax10 = plt.subplots()
ax10.barh(list(nonreason_counts.keys()), list(nonreason_counts.values()))
st.pyplot(fig10)

# Q10-2 - 유도 요소 (트리맵)
st.header("Q10-2. 참여를 유도할 수 있는 요소")
q10_2 = df["어떤 요소가 캠페인 및 제도 참여를 이끌 수 있을 것이라고 생각하시나요?"].dropna()
induce_counts = {}
for entry in q10_2:
    for item in entry.split(","):
        key = item.strip()
        induce_counts[key] = induce_counts.get(key, 0) + 1

fig11, ax11 = plt.subplots()
squarify.plot(sizes=induce_counts.values(), label=induce_counts.keys(), alpha=0.8)
ax11.axis("off")
st.pyplot(fig11)
