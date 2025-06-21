import streamlit as st
import pandas as pd

# 기준값 설정
THRESHOLDS = {
    "Turbidity": 5.0,                # NTU
    "Residual Free Chlorine": 0.2,   # mg/L
    "Coliform (MPN/100mL)": 0,       # 이상 없음
    "Fluoride": 1.5                  # mg/L
}

def calculate_risk_score(values):
    risk_score = 0
    explanations = []

    for key, value in values.items():
        threshold = THRESHOLDS[key]
        if key == "Coliform (MPN/100mL)":
            if value > threshold:
                risk_score += 3
                explanations.append(f"- **Coliform 검출됨**: {value} MPN → 🚫 위험")
            else:
                explanations.append(f"- **Coliform 없음** → ✅ 안전")
        elif key == "Residual Free Chlorine":
            if value < threshold:
                risk_score += 2
                explanations.append(f"- **염소 부족**: {value} mg/L < {threshold} → ⚠️ 소독 미흡")
            else:
                explanations.append(f"- **염소 적정**: {value} mg/L → ✅")
        else:
            if value > threshold:
                risk_score += 1
                explanations.append(f"- **{key} 기준 초과**: {value} > {threshold} → ⚠️")
            else:
                explanations.append(f"- **{key} 적정 수준** → ✅")
    
    return risk_score, explanations

# Streamlit UI
st.set_page_config(page_title="NYC 수질 위험도 평가", layout="centered")
st.title("💧 NYC 수돗물 수질 위험도 평가 시스템")
st.markdown("사용자 입력 기반으로 NYC 수질 위험도를 분석합니다. 기준은 WHO 및 NYC 환경보건 기준에 기반합니다.")

st.header("🔎 수질 데이터 입력")
user_input = {
    "Turbidity": st.number_input("탁도 (NTU)", min_value=0.0, step=0.1),
    "Residual Free Chlorine": st.number_input("잔류 염소 (mg/L)", min_value=0.0, step=0.1),
    "Coliform (MPN/100mL)": st.number_input("대장균 수 (MPN/100mL)", min_value=0, step=1),
    "Fluoride": st.number_input("불소 농도 (mg/L)", min_value=0.0, step=0.1),
}

if st.button("📈 위험도 평가 실행"):
    score, messages = calculate_risk_score(user_input)

    st.subheader("📋 평가 결과")
    for msg in messages:
        st.markdown(msg)

    st.markdown("---")
    if score >= 5:
        st.error(f"총 위험 점수: {score}점 → 🚨 **고위험 상태**")
    elif score >= 3:
        st.warning(f"총 위험 점수: {score}점 → ⚠️ **주의 필요**")
    else:
        st.success(f"총 위험 점수: {score}점 → ✅ **안전**")
