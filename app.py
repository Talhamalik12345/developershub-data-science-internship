import streamlit as st

st.set_page_config(
    page_title="DevelopersHub – Data Science Internship",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 DevelopersHub Corporation")
st.subheader("Data Science & Analytics – Advanced Internship Tasks")
st.caption("Due: 9 June 2026  ·  Completed: All 5 Tasks")

st.markdown("---")

st.markdown(
    "Use the **sidebar** to navigate between tasks. "
    "Each task contains a full pipeline: data exploration, modelling, evaluation, and insights."
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.info(
        "### 🏦 Task 1\n"
        "**Term Deposit Prediction**\n\n"
        "Classification · SHAP explanations · ROC · F1"
    )
with col2:
    st.success(
        "### 🛍️ Task 2\n"
        "**Customer Segmentation**\n\n"
        "K-Means · PCA · t-SNE · Marketing strategies"
    )
with col3:
    st.warning(
        "### ⚡ Task 3\n"
        "**Energy Forecasting**\n\n"
        "ARIMA · XGBoost · Seasonal decomposition · MAE/RMSE"
    )
with col4:
    st.error(
        "### 💳 Task 4\n"
        "**Loan Default Risk**\n\n"
        "Binary classification · Cost-threshold optimisation"
    )
with col5:
    st.info(
        "### 📊 Task 5\n"
        "**Business Dashboard**\n\n"
        "Plotly KPIs · Filters · Global Superstore data"
    )

st.markdown("---")

st.markdown("### Stack")

cols = st.columns(4)
stack = [
    ("Data", "`pandas`  `numpy`  `scipy`"),
    ("ML / AI", "`scikit-learn`  `xgboost`  `shap`"),
    ("Forecasting", "`statsmodels` (ARIMA)  `XGBoost`"),
    ("Visualisation", "`matplotlib`  `seaborn`  `plotly`  `streamlit`"),
]
for col, (label, libs) in zip(cols, stack):
    col.markdown(f"**{label}**\n\n{libs}")
