# DevelopersHub Corporation
## Data Science & Analytics – Advanced Internship Tasks

**Intern:** Talha Malik  
**Due Date:** 9 June 2026  
**Status:** ✅ All 5 Tasks Completed  
**Live App:** https://422b08be-2113-4ec1-a5a3-662175a5892d-00-1g3eyyrhmtfq7.pike.replit.dev

---

## Overview

This repository contains complete solutions to all 5 advanced data science tasks assigned as part of the DevelopersHub Corporation internship programme. Each task covers a real-world data science problem using industry-standard tools and libraries.

---

## Tasks

### Task 1 – Term Deposit Subscription Prediction (Bank Marketing)
**Objective:** Predict whether a bank customer will subscribe to a term deposit.

**Approach:**
- Loaded and explored the Bank Marketing dataset (UCI structure, 4,521 records)
- Encoded all categorical features using LabelEncoder
- Applied StandardScaler for feature normalisation
- Trained Logistic Regression and Random Forest classifiers
- Evaluated using Confusion Matrix, F1-Score, and ROC-AUC Curve
- Used **SHAP (TreeExplainer)** to explain 5 individual predictions with waterfall plots

**Results:**
- Random Forest outperformed Logistic Regression on both F1 and AUC
- Call duration and previous campaign outcome were the strongest predictors
- SHAP revealed feature-level reasoning for individual predictions

---

### Task 2 – Customer Segmentation Using Unsupervised Learning
**Objective:** Cluster customers based on spending habits and propose marketing strategies.

**Approach:**
- Conducted EDA on Mall Customers dataset (200 records)
- Used Elbow Method and Silhouette Score to determine optimal K = 5
- Applied K-Means Clustering on Annual Income and Spending Score
- Visualised clusters using **PCA** (2 components) and **t-SNE**
- Proposed tailored marketing strategies for each of the 5 segments

**Results:**
- 5 distinct customer segments identified
- Premium Spenders (high income, high score) and Conservative Affluent (high income, low score) identified as highest commercial opportunity
- t-SNE and PCA both confirmed clean cluster separation

---

### Task 3 – Energy Consumption Time Series Forecasting
**Objective:** Forecast short-term household energy usage using historical patterns.

**Approach:**
- Generated one year of hourly energy data with daily and weekly seasonality
- Resampled to daily frequency for stable forecasting
- Performed seasonal decomposition (trend, seasonality, residuals)
- Engineered time-based features: lag values, rolling means, day of week, month
- Compared three models on a 7-day held-out test set:
  - **ARIMA(2,1,2)(1,1,1)[7]** – statsmodels
  - **XGBoost** – with lag and rolling features
  - **Seasonal Naïve** – Prophet-style weekly baseline

**Results (MAE / RMSE):**
- XGBoost achieved lowest forecasting error
- ARIMA strong on stationary components
- Seasonal Naïve competitive baseline leveraging weekly periodicity

---

### Task 4 – Loan Default Risk with Business Cost Optimisation
**Objective:** Predict loan default and optimise decision threshold based on cost-benefit analysis.

**Approach:**
- Cleaned and preprocessed Home Credit-style dataset (10,000 applicants)
- Trained Logistic Regression and XGBoost binary classifiers (class-weight balanced)
- Defined business cost values: False Positive (missed profit) and False Negative (default loss)
- Swept decision thresholds from 0.01 to 0.99, computed total business cost at each point
- Selected threshold that minimises total cost

**Results:**
- XGBoost achieved higher AUC-ROC than Logistic Regression
- Default threshold of 0.5 proved sub-optimal; cost-based tuning improved business outcome
- Prior defaults and debt-to-income ratio were the strongest risk signals

---

### Task 5 – Interactive Business Dashboard (Global Superstore)
**Objective:** Develop an interactive dashboard for sales, profit, and segment-wise performance.

**Approach:**
- Cleaned and prepared Global Superstore dataset (~10,000 orders, 2021–2024)
- Built a multi-filter Streamlit dashboard (Region, Category, Sub-Category, Segment, Year)
- Displayed KPIs: Total Sales, Total Profit, Profit Margin, Orders, Avg Order Value
- Charts: Monthly Sales & Profit trend, Regional Sales, Category Profit, Sub-Category breakdown, Top 5 Customers, Segment Bubble Chart, Discount Impact scatter, Regional Heatmap

**Key Insights:**
- Technology has the best profit margins; Furniture Tables run at a loss
- Heavy discounting (>30%) strongly correlates with negative profit
- Corporate segment has better margin than Consumer despite lower order volume

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python -m streamlit run app.py
```

Open browser at: `http://localhost:8080`

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data | `pandas`, `numpy`, `scipy` |
| Machine Learning | `scikit-learn`, `xgboost` |
| Explainable AI | `shap` |
| Time Series | `statsmodels` |
| Visualisation | `matplotlib`, `seaborn`, `plotly` |
| Dashboard | `streamlit` |

---

## Repository Structure

```
data-science/
├── app.py                          # Home page
├── requirements.txt                # Dependencies
├── .streamlit/
│   └── config.toml                 # Streamlit config & theme
└── pages/
    ├── 1_🏦_Bank_Marketing.py      # Task 1
    ├── 2_🛍️_Customer_Segmentation.py  # Task 2
    ├── 3_⚡_Energy_Forecasting.py  # Task 3
    ├── 4_💳_Loan_Default_Risk.py   # Task 4
    └── 5_📊_Business_Dashboard.py  # Task 5
```
