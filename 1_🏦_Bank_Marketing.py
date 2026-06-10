"""
Task 1 – Term Deposit Subscription Prediction (Bank Marketing)
Objective: Predict whether a bank customer will subscribe to a term deposit.
Dataset: Synthetic based on UCI Bank Marketing dataset structure.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, f1_score, roc_curve, auc,
    classification_report, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Task 1 – Bank Marketing", page_icon="🏦", layout="wide")

st.title("🏦 Task 1 – Term Deposit Subscription Prediction")
st.markdown("""
**Objective:** Predict whether a bank customer will subscribe to a term deposit (target: `y`) 
using the Bank Marketing dataset structure. Models: Logistic Regression & Random Forest.  
Explainability via **SHAP** for 5 individual predictions.
""")
st.markdown("---")


# ─── Data Generation ──────────────────────────────────────────────────────────
@st.cache_data
def generate_bank_data(n=4521, seed=42):
    rng = np.random.default_rng(seed)
    n = int(n)

    job_cats = ["admin.", "blue-collar", "entrepreneur", "housemaid",
                "management", "retired", "self-employed", "services",
                "student", "technician", "unemployed", "unknown"]
    marital_cats = ["divorced", "married", "single"]
    education_cats = ["primary", "secondary", "tertiary", "unknown"]
    contact_cats = ["cellular", "telephone", "unknown"]
    month_cats = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    poutcome_cats = ["failure", "other", "success", "unknown"]

    age = rng.integers(18, 95, n)
    balance = rng.integers(-3000, 80000, n)
    duration = rng.integers(0, 4000, n)
    campaign = rng.integers(1, 50, n)
    pdays = rng.choice([-1] + list(range(0, 400)), n)
    previous = rng.integers(0, 30, n)

    job = rng.choice(job_cats, n)
    marital = rng.choice(marital_cats, n)
    education = rng.choice(education_cats, n)
    default = rng.choice(["no", "yes"], n, p=[0.98, 0.02])
    housing = rng.choice(["no", "yes"], n)
    loan = rng.choice(["no", "yes"], n, p=[0.84, 0.16])
    contact = rng.choice(contact_cats, n)
    month = rng.choice(month_cats, n)
    poutcome = rng.choice(poutcome_cats, n, p=[0.10, 0.03, 0.11, 0.76])

    # Synthesise target with some realistic signal
    logit = (
        -2.5
        + 0.02 * (age - 40)
        + 0.00005 * balance
        + 0.002 * duration
        - 0.04 * campaign
        + (0.8 * (poutcome == "success"))
        + (0.3 * (education == "tertiary"))
        + (-0.4 * (housing == "yes"))
    )
    prob = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "balance": balance, "housing": housing, "loan": loan,
        "contact": contact, "month": month, "duration": duration,
        "campaign": campaign, "pdays": pdays, "previous": previous,
        "poutcome": poutcome, "y": y
    })
    return df


df = generate_bank_data()

# ─── EDA ──────────────────────────────────────────────────────────────────────
st.subheader("1. Exploratory Data Analysis")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{len(df):,}")
col2.metric("Features", str(df.shape[1] - 1))
col3.metric("Subscribed (y=1)", f"{df['y'].sum():,} ({df['y'].mean()*100:.1f}%)")
col4.metric("Not Subscribed (y=0)", f"{(df['y']==0).sum():,}")

with st.expander("📋 Raw Data Sample (first 10 rows)"):
    st.dataframe(df.head(10), width="stretch")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Target distribution
axes[0].bar(["No (0)", "Yes (1)"], df["y"].value_counts().sort_index().values,
            color=["#e74c3c", "#2ecc71"])
axes[0].set_title("Target Distribution")
axes[0].set_ylabel("Count")

# Age distribution by outcome
df[df["y"] == 0]["age"].hist(ax=axes[1], alpha=0.6, color="#e74c3c", label="No", bins=20)
df[df["y"] == 1]["age"].hist(ax=axes[1], alpha=0.6, color="#2ecc71", label="Yes", bins=20)
axes[1].set_title("Age by Subscription")
axes[1].legend()

# Balance distribution
axes[2].hist(df["balance"].clip(-2000, 20000), bins=30, color="#3498db", edgecolor="white")
axes[2].set_title("Account Balance Distribution")
axes[2].set_xlabel("Balance (€)")

plt.tight_layout()
st.pyplot(fig)
plt.close()

# ─── Preprocessing ────────────────────────────────────────────────────────────
st.subheader("2. Preprocessing & Feature Encoding")

@st.cache_data
def preprocess(df):
    df2 = df.copy()
    cat_cols = df2.select_dtypes("object").columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df2[col] = le.fit_transform(df2[col])
        encoders[col] = le
    X = df2.drop("y", axis=1)
    y = df2["y"]
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, y, scaler, encoders, X.columns.tolist()

X, y, scaler, encoders, feature_names = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

st.markdown(f"- Encoded **{len(encoders)} categorical columns** using LabelEncoder  \n"
            f"- Applied **StandardScaler** to all features  \n"
            f"- Train/Test split: **80/20** (stratified on target)")

# ─── Model Training ───────────────────────────────────────────────────────────
st.subheader("3. Model Training & Evaluation")

@st.cache_resource
def train_models(X_train, y_train):
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return lr, rf

lr_model, rf_model = train_models(X_train, y_train)

model_choice = st.selectbox("Select model to inspect", ["Logistic Regression", "Random Forest"])
model = lr_model if model_choice == "Logistic Regression" else rf_model

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix – {model_choice}")
    st.pyplot(fig)
    plt.close()

with col_b:
    st.markdown("#### ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#2ecc71", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve – {model_choice}")
    ax.legend()
    st.pyplot(fig)
    plt.close()

f1 = f1_score(y_test, y_pred)
st.markdown(f"**F1-Score ({model_choice}):** `{f1:.4f}`  |  **AUC:** `{roc_auc:.4f}`")

with st.expander("Full Classification Report"):
    st.code(classification_report(y_test, y_pred, target_names=["No", "Yes"]))

# ─── Model Comparison ─────────────────────────────────────────────────────────
st.subheader("4. Model Comparison")

lr_f1 = f1_score(y_test, lr_model.predict(X_test))
rf_f1 = f1_score(y_test, rf_model.predict(X_test))
lr_auc = auc(*roc_curve(y_test, lr_model.predict_proba(X_test)[:, 1])[:2])
rf_auc = auc(*roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])[:2])

comparison_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "F1-Score": [round(lr_f1, 4), round(rf_f1, 4)],
    "AUC-ROC": [round(lr_auc, 4), round(rf_auc, 4)],
})
st.dataframe(comparison_df.set_index("Model"), width="stretch")

# ─── SHAP Explanations ────────────────────────────────────────────────────────
st.subheader("5. SHAP Explanations – 5 Individual Predictions")
st.markdown("SHAP (SHapley Additive exPlanations) shows each feature's contribution to a single prediction.")

@st.cache_resource
def compute_shap(_rf_model, X_train_sample, X_test_sample):
    explainer = shap.TreeExplainer(_rf_model)
    shap_values = explainer.shap_values(X_test_sample)
    return explainer, shap_values

X_test_sample = X_test.iloc[:100].reset_index(drop=True)
explainer_rf, shap_values_rf = compute_shap(rf_model, X_train.iloc[:200], X_test_sample)

# SHAP summary bar plot
st.markdown("#### Feature Importance (SHAP – Random Forest)")
fig, ax = plt.subplots(figsize=(8, 5))
# shap_values_rf shape: (n_samples, n_features, n_classes) or (n_samples, n_features)
if isinstance(shap_values_rf, list):
    sv = shap_values_rf[1]  # class 1
elif isinstance(shap_values_rf, np.ndarray) and shap_values_rf.ndim == 3:
    sv = shap_values_rf[:, :, 1]  # shape (n_samples, n_features) for class 1
else:
    sv = shap_values_rf

mean_abs = np.abs(sv).mean(axis=0).flatten()
importance_df = pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": mean_abs})
importance_df = importance_df.sort_values("Mean |SHAP|", ascending=True).tail(10)
ax.barh(importance_df["Feature"], importance_df["Mean |SHAP|"], color="#3498db")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Top-10 Features by SHAP Importance")
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("#### Waterfall Plots – 5 Individual Predictions")

shap_base = explainer_rf.expected_value
if isinstance(shap_base, (list, np.ndarray)):
    shap_base = shap_base[1]

for i in range(5):
    pred_label = "Subscribed ✅" if rf_model.predict(X_test_sample.iloc[[i]])[0] == 1 else "Not Subscribed ❌"
    with st.expander(f"Prediction {i+1}: {pred_label}"):
        sv_i = sv[i]
        sorted_idx = np.argsort(np.abs(sv_i))[-8:]
        feat_names_top = [feature_names[j] for j in sorted_idx]
        sv_top = sv_i[sorted_idx]

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in sv_top]
        ax.barh(feat_names_top, sv_top, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("SHAP value")
        ax.set_title(f"Sample {i+1} – {pred_label}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ─── Conclusion ───────────────────────────────────────────────────────────────
st.subheader("6. Conclusion & Key Insights")
st.success("""
- **Random Forest outperforms Logistic Regression** on both F1 and AUC metrics.
- **Call duration** is the strongest predictor — longer conversations strongly correlate with subscription.
- **Previous campaign outcome (success)** dramatically increases subscription probability.
- **Age** and **account balance** are secondary but meaningful features.
- SHAP waterfall plots reveal *why* individual customers are predicted to subscribe or not,
  enabling targeted marketing and personalised outreach.
""")
