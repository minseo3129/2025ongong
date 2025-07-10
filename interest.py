# markov_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1단계: 데이터 불러오기 및 전처리
df = pd.read_csv("index.csv")

# 날짜 통합
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

# 필요한 열만 선택 + 결측 제거
df = df[['Date', 'Effective Federal Funds Rate']].dropna()

# 금리 변화율 계산
df['Rate Change'] = df['Effective Federal Funds Rate'].diff()

# 상태 분류 함수 정의
def classify_state(change):
    if pd.isna(change):
        return None
    elif change > 0.05:
        return '상승'
    elif change < -0.05:
        return '하락'
    else:
        return '보합'

df['State'] = df['Rate Change'].apply(classify_state)
df = df.dropna(subset=['State']).reset_index(drop=True)

print("📌 총 데이터 개수:", len(df))

# 2단계: 데이터 분할 (시계열 순서대로 70% 학습, 30% 테스트)
split_index = int(len(df) * 0.7)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

# 전이확률행렬 계산 함수
def compute_transition_matrix(state_list):
    states = ['상승', '보합', '하락']
    counts = defaultdict(lambda: defaultdict(int))
    
    for (s1, s2) in zip(state_list[:-1], state_list[1:]):
        counts[s1][s2] += 1

    matrix = pd.DataFrame(index=states, columns=states).fillna(0)
    for from_state in states:
        total = sum(counts[from_state].values())
        if total == 0:
            continue
        for to_state in states:
            matrix.loc[from_state, to_state] = counts[from_state][to_state] / total

    return matrix

# 학습 데이터에서 마르코프 행렬 추출
train_matrix = compute_transition_matrix(train_df['State'].tolist())
print("\n🔍 학습 데이터 기반 전이확률 행렬:")
print(train_matrix.round(3))

# 테스트 데이터에서도 전이확률행렬 추출 (성능 비교용)
test_matrix = compute_transition_matrix(test_df['State'].tolist())
print("\n🔍 테스트 데이터 기반 전이확률 행렬:")
print(test_matrix.round(3))

# 3단계: 예측 및 정확도 평가
actual_states = test_df['State'].tolist()
predicted_states = []
current_state = train_df['State'].iloc[-1]  # 예측 시작 지점은 학습 마지막 상태

for _ in actual_states:
    if current_state in train_matrix.index:
        next_state = train_matrix.loc[current_state].astype(float).idxmax()
    else:
        next_state = '보합'
    predicted_states.append(next_state)
    current_state = next_state  # 다음 상태에 반영

# 정확도 계산
accuracy = np.mean([a == p for a, p in zip(actual_states, predicted_states)])
print(f"\n📊 예측 정확도: {accuracy:.2%}")

# 4단계: 시각화
state_to_num = {'상승': 1, '보합': 0, '하락': -1}
actual_numeric = [state_to_num[s] for s in actual_states]
predicted_numeric = [state_to_num[s] for s in predicted_states]
dates = test_df['Date']

# 시계열 비교 그래프
plt.figure(figsize=(14, 6))
plt.plot(dates, actual_numeric, label='실제 상태', marker='o', linewidth=2)
plt.plot(dates, predicted_numeric, label='예측 상태', linestyle='--', marker='x', linewidth=2)
plt.yticks([-1, 0, 1], ['하락', '보합', '상승'])
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.title('📈 금리 상태: 마르코프 모델 예측 vs 실제', fontsize=14)
plt.xlabel('날짜')
plt.ylabel('상태')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 혼동 행렬 시각화
cm = confusion_matrix(actual_states, predicted_states, labels=['상승', '보합', '하락'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['상승', '보합', '하락'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues')
plt.title("🔍 혼동 행렬")
plt.tight_layout()
plt.show()