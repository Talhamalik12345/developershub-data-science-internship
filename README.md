# DevelopersHub Corporation — Data Science & Analytics Internship Tasks

**Intern:** [TALHA AKRAM]
**Due Date:** 15th May, 2026

---

## Tasks Completed

| # | Task | Dataset | Model Used | Status |
|---|------|---------|------------|--------|
| 1 | Exploring & Visualizing Iris Dataset | Iris (via seaborn) | Visualization only | ✅ Done |
| 2 | Credit Risk Prediction | Loan Prediction Dataset | Logistic Regression, Decision Tree | ✅ Done |
| 3 | Customer Churn Prediction | Churn Modelling Dataset | Random Forest, Logistic Regression, Decision Tree | ✅ Done |
| 4 | Predicting Insurance Claim Amounts | Medical Cost Personal Dataset | Linear Regression, Random Forest | ✅ Done |
| 5 | Personal Loan Acceptance Prediction | Bank Marketing Dataset (UCI) | Logistic Regression, Decision Tree, Random Forest | ✅ Done |

---

## Task 1: Exploring and Visualizing the Iris Dataset

### Objective
Understand how to read, summarize, and visualize a dataset.

### Approach
- Loaded the Iris dataset using `seaborn.load_dataset('iris')`
- Inspected shape, columns, data types, and missing values
- Created scatter plots, histograms, box plots, pair plots, and a correlation heatmap

### Results & Insights
- Dataset has 150 samples, 4 features, 0 missing values, 3 balanced species (50 each)
- Petal length and petal width show the clearest species separation
- Setosa is completely separable; Versicolor and Virginica overlap slightly
- Petal length & width are highly correlated (r ≈ 0.96)

---

## Task 2: Credit Risk Prediction

### Objective
Predict whether a loan applicant is likely to default on a loan.

### Approach
- Handled missing values: categorical → mode, numerical → median
- Engineered features: Total Income, log-transformed Loan Amount and Income
- Label-encoded categorical variables
- Trained Logistic Regression and Decision Tree classifiers
- Evaluated using accuracy, confusion matrix, and classification report

### Results & Insights
- **Credit History** is the most important predictor by far
- **Total Income** and **Loan Amount** also significantly impact approval
- Decision Tree provided interpretable feature importance rankings
- Graduates with positive credit history have the highest approval rates

---

## Task 3: Customer Churn Prediction (Bank Customers)

### Objective
Identify customers who are likely to leave the bank.

### Approach
- Dropped non-predictive columns (CustomerID, RowNumber, Surname)
- Label-encoded Gender; one-hot encoded Geography
- Trained Logistic Regression, Decision Tree, and Random Forest
- Evaluated with accuracy, ROC-AUC, confusion matrix, and classification report

### Results & Insights
- ~20% churn rate in the dataset
- **Age**, **Germany geography**, and **inactive membership** are top churn drivers
- Customers with exactly 2 products churn the least
- Random Forest achieved the best ROC-AUC score

---

## Task 4: Predicting Insurance Claim Amounts

### Objective
Estimate medical insurance claim amounts based on personal data.

### Approach
- Encoded sex, smoker (label encoding), and region (one-hot encoding)
- Trained Linear Regression and Random Forest Regressor
- Evaluated using MAE, RMSE, and R² score
- Visualized residuals and actual vs predicted plots

### Results & Insights
- **Smoking** is by far the strongest predictor — smokers pay 3–4× more
- **Age** and **BMI** also strongly influence charges
- Linear Regression explains ~75–80% of variance (R² ≈ 0.75–0.80)
- Random Forest achieves ~85–90% (R² ≈ 0.85–0.90)

---

## Task 5: Personal Loan Acceptance Prediction

### Objective
Predict which customers are likely to accept a personal loan offer.

### Approach
- Label-encoded all categorical features
- Trained Logistic Regression, Decision Tree, and Random Forest
- Analyzed feature importance and customer segment acceptance rates
- Evaluated with accuracy, ROC-AUC, confusion matrix

### Results & Insights
- **Call duration** is the strongest predictor — longer calls = higher acceptance
- **Students** and **retired** customers show highest acceptance rates
- Best months for campaigns: March, September, October, December
- Cellular contact outperforms telephone contact significantly
- Model can reduce marketing costs by 70% by targeting top 30% most likely to accept

---

## Libraries Used

```
pandas, numpy, matplotlib, seaborn, scikit-learn
```

## How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

2. Open Jupyter:
   ```bash
   jupyter notebook
   ```

3. Open each `.ipynb` file and run all cells (`Kernel → Restart & Run All`)
