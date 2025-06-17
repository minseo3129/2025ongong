import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.neighbors import KNeighborsClassifier


st.set_page_config(layout="wide")
st.title("🌱 스마트팜 생장 데이터 분석 및 조건 기반 작물 재배 매뉴얼")

# 데이터 로드 (로컬 파일로 수정)
df = pd.read_csv("plant_growth_data.csv")
df["Failure"] = 1 - df["Growth_Milestone"]

# 📊 1. 박스플롯
st.subheader("📊 1. 생장 성공/실패군의 주요 변수 분포 (Boxplot)")
for feature in ["Sunlight_Hours", "Temperature", "Humidity"]:
    fig = px.box(df, x="Failure", y=feature, color="Failure",
                 title=f"{feature}에 따른 생장 성공/실패 분포",
                 labels={"Failure": "성공(0)/실패(1)"})
    st.plotly_chart(fig, use_container_width=True)
