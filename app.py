import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# 페이지 설정
st.set_page_config(page_title="머신러닝 모델 비교기", layout="wide")

st.title("📊 통계적 머신러닝 모델 비교기")
st.markdown("로지스틱 회귀, LASSO, Ridge 모델의 성능을 비교해보세요!")

# 데이터 로드
@st.cache_data
def load_data():
    data = load_breast_cancer()
    return pd.DataFrame(data.data, columns=data.feature_names), data.target

X, y = load_data()

# 사용자 입력
model_type = st.radio("모델 선택", ["Logistic Regression", "LASSO", "Ridge"])
C = st.slider("정규화 강도 (C)", 0.01, 10.0, 1.0)
test_size = st.slider("테스트 데이터 비율", 0.1, 0.5, 0.3)

# 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

penalty = 'none'
solver = 'lbfgs'

if model_type == "LASSO":
    penalty = 'l1'
    solver = 'saga'
elif model_type == "Ridge":
    penalty = 'l2'
    solver = 'saga'

model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 결과 출력
st.success(f"✅ 정확도: **{accuracy:.2f}**")

# 계수 시각화
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values("Coefficient", key=abs, ascending=False)

st.subheader("📌 모델 계수 (Feature Importance)")
st.dataframe(coefficients)

st.bar_chart(coefficients.set_index("Feature"))

