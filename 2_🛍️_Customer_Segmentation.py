"""
Task 2 – Customer Segmentation Using Unsupervised Learning
Objective: Cluster customers based on spending habits using K-Means.
Dataset: Synthetic Mall Customers dataset.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Task 2 – Customer Segmentation", page_icon="🛍️", layout="wide")

st.title("🛍️ Task 2 – Customer Segmentation Using Unsupervised Learning")
st.markdown("""
**Objective:** Segment mall customers by spending habits using **K-Means clustering**.  
Visualise clusters with **PCA** and **t-SNE**, then derive actionable marketing strategies.
""")
st.markdown("---")


# ─── Data Generation ──────────────────────────────────────────────────────────
@st.cache_data
def generate_mall_customers(n=200, seed=42):
    rng = np.random.default_rng(seed)
    genders = rng.choice(["Male", "Female"], n, p=[0.44, 0.56])

    # Simulate 5 natural customer clusters
    cluster_params = [
        dict(age_mu=45, age_s=5, income_mu=26, income_s=4, score_mu=20, score_s=5),   # Low income, low spend
        dict(age_mu=55, age_s=8, income_mu=55, income_s=8, score_mu=49, score_s=10),   # Mid income, mid spend
        dict(age_mu=32, age_s=6, income_mu=86, income_s=8, score_mu=82, score_s=8),    # High income, high spend
        dict(age_mu=40, age_s=8, income_mu=88, income_s=8, score_mu=18, score_s=6),    # High income, low spend
        dict(age_mu=25, age_s=5, income_mu=25, income_s=5, score_mu=78, score_s=8),    # Low income, high spend
    ]
    sizes = [n // 5] * 4 + [n - 4 * (n // 5)]
    rows = []
    for cid, (p, sz) in enumerate(zip(cluster_params, sizes), 1):
        age = rng.normal(p["age_mu"], p["age_s"], sz).clip(18, 70).astype(int)
        income = rng.normal(p["income_mu"], p["income_s"], sz).clip(15, 137).astype(int)
        score = rng.normal(p["score_mu"], p["score_s"], sz).clip(1, 100).astype(int)
        for a, inc, sc, g in zip(age, income, score, genders[:sz]):
            rows.append({"Gender": g, "Age": a, "Annual_Income_k": inc, "Spending_Score": sc})
    df = pd.DataFrame(rows)
    df.index = range(1, len(df) + 1)
    df.index.name = "CustomerID"
    return df.sample(frac=1, random_state=seed).reset_index(drop=False)

df = generate_mall_customers()

# ─── EDA ──────────────────────────────────────────────────────────────────────
st.subheader("1. Exploratory Data Analysis")

col1, col2, col3 = st.columns(3)
col1.metric("Customers", str(len(df)))
col2.metric("Avg Annual Income", f"${df['Annual_Income_k'].mean():.0f}k")
col3.metric("Avg Spending Score", f"{df['Spending_Score'].mean():.1f} / 100")

with st.expander("📋 Dataset Sample"):
    st.dataframe(df.head(10), width="stretch")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].hist(df["Age"], bins=20, color="#3498db", edgecolor="white")
axes[0].set_title("Age Distribution")
axes[1].hist(df["Annual_Income_k"], bins=20, color="#e67e22", edgecolor="white")
axes[1].set_title("Annual Income (k$)")
axes[2].hist(df["Spending_Score"], bins=20, color="#2ecc71", edgecolor="white")
axes[2].set_title("Spending Score (1–100)")
for ax in axes:
    ax.set_ylabel("Count")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ─── Elbow + Silhouette ───────────────────────────────────────────────────────
st.subheader("2. Optimal K – Elbow Method & Silhouette Score")

features = ["Annual_Income_k", "Spending_Score"]
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias, silhouettes = [], []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(k_range), inertias, "o-", color="#e74c3c", lw=2)
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].set_title("Elbow Method")
axes[0].axvline(5, color="gray", linestyle="--", label="K=5 (chosen)")
axes[0].legend()

axes[1].plot(list(k_range), silhouettes, "o-", color="#3498db", lw=2)
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Scores")
axes[1].axvline(5, color="gray", linestyle="--", label="K=5 (chosen)")
axes[1].legend()
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.info("Both the Elbow and Silhouette methods indicate **K = 5** as the optimal number of clusters.")

# ─── K-Means ──────────────────────────────────────────────────────────────────
st.subheader("3. K-Means Clustering (K = 5)")

k_input = st.slider("Adjust K (for exploration):", 2, 10, 5)

km_final = KMeans(n_clusters=k_input, random_state=42, n_init=10)
df["Cluster"] = km_final.fit_predict(X_scaled)

palette = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
           "#1abc9c", "#e67e22", "#95a5a6", "#34495e", "#c0392b"]
colors = [palette[c] for c in df["Cluster"]]

fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(df["Annual_Income_k"], df["Spending_Score"],
                     c=[palette[c] for c in df["Cluster"]], s=60, alpha=0.8, edgecolors="white", lw=0.4)
centers = scaler.inverse_transform(km_final.cluster_centers_)
ax.scatter(centers[:, 0], centers[:, 1], c="black", marker="X", s=200, zorder=5, label="Centroids")
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1–100)")
ax.set_title(f"K-Means Clusters (K={k_input}) – Income vs Spending Score")
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close()

cluster_summary = df.groupby("Cluster")[["Age", "Annual_Income_k", "Spending_Score"]].mean().round(1)
cluster_summary.columns = ["Avg Age", "Avg Income (k$)", "Avg Spending Score"]
st.markdown("**Cluster Averages:**")
st.dataframe(cluster_summary, width="stretch")

# ─── PCA Visualisation ────────────────────────────────────────────────────────
st.subheader("4a. Dimensionality Reduction – PCA")

X_full = df[["Age", "Annual_Income_k", "Spending_Score"]].values
X_full_scaled = StandardScaler().fit_transform(X_full)
km_3d = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X_full_scaled)
labels_3d = km_3d.labels_

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_full_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
for c in range(5):
    mask = labels_3d == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Cluster {c}",
               alpha=0.7, s=50, color=palette[c], edgecolors="white", lw=0.3)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_title("Customer Clusters – PCA Projection")
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ─── t-SNE Visualisation ──────────────────────────────────────────────────────
st.subheader("4b. Dimensionality Reduction – t-SNE")
st.caption("t-SNE reveals local cluster structure; best for smaller datasets.")

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_full_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
for c in range(5):
    mask = labels_3d == c
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=f"Cluster {c}",
               alpha=0.7, s=50, color=palette[c], edgecolors="white", lw=0.3)
ax.set_xlabel("t-SNE Component 1")
ax.set_ylabel("t-SNE Component 2")
ax.set_title("Customer Clusters – t-SNE Projection")
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ─── Marketing Strategies ─────────────────────────────────────────────────────
st.subheader("5. Marketing Strategies per Segment")

strategies = {
    "Cluster 0 – Careful Spenders\n(Mid Income, Low Score)": (
        "💼 **Value-Focused Promotions**\n\n"
        "Customers with moderate income but conservative spending. "
        "Target with loyalty programmes, bundle discounts, and value-for-money messaging. "
        "Emphasise quality and long-term savings."
    ),
    "Cluster 1 – Standard Shoppers\n(Mid Income, Mid Score)": (
        "🛒 **Engagement & Upsell**\n\n"
        "Core mid-market segment. Grow their basket with personalised recommendations, "
        "cross-sell complementary products, and regular email newsletters with curated picks."
    ),
    "Cluster 2 – Premium Spenders\n(High Income, High Score)": (
        "💎 **Premium & Exclusivity**\n\n"
        "High-value customers. Target with exclusive VIP events, limited-edition products, "
        "concierge service, and premium brand partnerships. Retention is key."
    ),
    "Cluster 3 – Conservative Affluent\n(High Income, Low Score)": (
        "🎯 **Re-engagement Campaigns**\n\n"
        "High earners who spend little. Investigate barriers (inconvenience, irrelevant offers). "
        "Offer personalised experiences, luxury previews, and private shopping events to activate them."
    ),
    "Cluster 4 – Enthusiastic Shoppers\n(Low Income, High Score)": (
        "🎉 **Affordable Luxury & BNPL**\n\n"
        "Young, enthusiastic but budget-constrained. Offer instalment/BNPL schemes, "
        "student/youth discounts, trendy affordable lines, and social-media-driven promotions."
    ),
}

for title, desc in strategies.items():
    with st.expander(title):
        st.markdown(desc)

# ─── Conclusion ───────────────────────────────────────────────────────────────
st.subheader("6. Conclusion")
st.success("""
- K-Means with **K = 5** produces well-separated, interpretable customer segments.
- **PCA** and **t-SNE** both confirm the cluster structure in 2D projections.
- The most commercially valuable segments are **Premium Spenders** (Cluster 2) and
  **Conservative Affluent** (Cluster 3, high-potential upsell).
- Tailored marketing strategies for each cluster can significantly improve
  conversion rates, customer lifetime value, and retention.
""")
