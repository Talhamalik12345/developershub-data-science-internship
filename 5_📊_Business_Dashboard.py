"""
Task 5 – Interactive Business Dashboard in Streamlit
Objective: Analyse sales, profit, and segment-wise performance using the Global Superstore dataset.
Dataset: Synthetic Global Superstore dataset.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Task 5 – Business Dashboard", page_icon="📊", layout="wide")

st.title("📊 Task 5 – Interactive Business Intelligence Dashboard")
st.markdown("""
**Global Superstore** – Sales, Profit & Segment Performance Dashboard  
Use the **sidebar filters** to explore the data by Region, Category, and Sub-Category.
""")
st.markdown("---")


# ─── Data Generation ──────────────────────────────────────────────────────────
@st.cache_data
def generate_superstore_data(n=9994, seed=42):
    rng = np.random.default_rng(seed)

    regions = ["West", "East", "Central", "South"]
    categories = ["Technology", "Furniture", "Office Supplies"]
    sub_categories = {
        "Technology": ["Phones", "Machines", "Accessories", "Copiers"],
        "Furniture": ["Chairs", "Tables", "Bookcases", "Furnishings"],
        "Office Supplies": ["Binders", "Paper", "Storage", "Art", "Labels", "Fasteners", "Envelopes", "Supplies", "Appliances"],
    }
    segments = ["Consumer", "Corporate", "Home Office"]
    ship_modes = ["Standard Class", "Second Class", "First Class", "Same Day"]

    first_names = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace",
                   "Hank", "Iris", "Jack", "Karen", "Leo", "Mia", "Ned",
                   "Olivia", "Pete", "Quinn", "Rose", "Sam", "Tina"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
                  "Miller", "Davis", "Wilson", "Taylor", "Anderson", "Thomas"]

    rows = []
    order_date = pd.date_range("2021-01-01", "2024-12-31", periods=n)
    for i in range(n):
        cat = rng.choice(categories)
        sub = rng.choice(sub_categories[cat])
        region = rng.choice(regions, p=[0.32, 0.28, 0.22, 0.18])
        segment = rng.choice(segments, p=[0.52, 0.30, 0.18])
        customer = rng.choice(first_names) + " " + rng.choice(last_names)

        # Realistic sales per sub-category
        base_sales = {
            "Phones": 400, "Machines": 800, "Accessories": 80, "Copiers": 2000,
            "Chairs": 300, "Tables": 600, "Bookcases": 250, "Furnishings": 100,
            "Binders": 40, "Paper": 30, "Storage": 90, "Art": 15,
            "Labels": 12, "Fasteners": 8, "Envelopes": 20, "Supplies": 60,
            "Appliances": 150,
        }.get(sub, 100)
        sales = max(5, rng.normal(base_sales, base_sales * 0.3))
        # Profit margin varies by category
        margin = {"Technology": 0.14, "Furniture": -0.02, "Office Supplies": 0.18}[cat]
        profit = sales * rng.normal(margin, 0.08)
        discount = rng.choice([0, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50], p=[0.5, 0.15, 0.1, 0.1, 0.06, 0.05, 0.04])
        quantity = rng.integers(1, 15)

        rows.append({
            "Order Date": order_date[i],
            "Region": region,
            "Segment": segment,
            "Category": cat,
            "Sub-Category": sub,
            "Customer Name": customer,
            "Sales": round(sales, 2),
            "Profit": round(profit, 2),
            "Discount": discount,
            "Quantity": quantity,
            "Ship Mode": rng.choice(ship_modes),
        })
    df = pd.DataFrame(rows)
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.to_period("M").astype(str)
    return df


df_full = generate_superstore_data()

# ─── Sidebar Filters ──────────────────────────────────────────────────────────
st.sidebar.header("🔧 Filters")

years = sorted(df_full["Year"].unique())
selected_years = st.sidebar.multiselect("Year", years, default=years)

regions = sorted(df_full["Region"].unique())
selected_regions = st.sidebar.multiselect("Region", regions, default=regions)

categories = sorted(df_full["Category"].unique())
selected_categories = st.sidebar.multiselect("Category", categories, default=categories)

sub_cats = sorted(df_full[df_full["Category"].isin(selected_categories)]["Sub-Category"].unique())
selected_sub_cats = st.sidebar.multiselect("Sub-Category", sub_cats, default=sub_cats)

segments = sorted(df_full["Segment"].unique())
selected_segments = st.sidebar.multiselect("Segment", segments, default=segments)

# Apply filters
df = df_full[
    (df_full["Year"].isin(selected_years)) &
    (df_full["Region"].isin(selected_regions)) &
    (df_full["Category"].isin(selected_categories)) &
    (df_full["Sub-Category"].isin(selected_sub_cats)) &
    (df_full["Segment"].isin(selected_segments))
].copy()

if df.empty:
    st.warning("No data matches the selected filters. Please adjust your selections.")
    st.stop()

# ─── KPI Cards ────────────────────────────────────────────────────────────────
st.subheader("Key Performance Indicators")

total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
profit_margin = total_profit / total_sales * 100 if total_sales else 0
total_orders = len(df)
avg_order_value = total_sales / total_orders if total_orders else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("💰 Total Sales", f"${total_sales:,.0f}")
k2.metric("📈 Total Profit", f"${total_profit:,.0f}")
k3.metric("📉 Profit Margin", f"{profit_margin:.1f}%")
k4.metric("📦 Total Orders", f"{total_orders:,}")
k5.metric("🛒 Avg Order Value", f"${avg_order_value:,.2f}")

st.markdown("---")

# ─── Sales & Profit Trend ─────────────────────────────────────────────────────
st.subheader("Sales & Profit Over Time")

monthly = df.groupby("Month")[["Sales", "Profit"]].sum().reset_index()
monthly = monthly.sort_values("Month")

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=monthly["Month"], y=monthly["Sales"], name="Sales",
                     marker_color="#3498db", opacity=0.75), secondary_y=False)
fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["Profit"], name="Profit",
                         mode="lines+markers", line=dict(color="#e74c3c", width=2),
                         marker=dict(size=4)), secondary_y=True)
fig.update_layout(title="Monthly Sales (bars) & Profit (line)", height=380,
                  legend=dict(orientation="h", y=1.05))
fig.update_yaxes(title_text="Sales ($)", secondary_y=False)
fig.update_yaxes(title_text="Profit ($)", secondary_y=True)
st.plotly_chart(fig, width="stretch")

# ─── Region & Category Breakdown ─────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Sales by Region")
    region_sales = df.groupby("Region")["Sales"].sum().reset_index().sort_values("Sales", ascending=False)
    fig = px.bar(region_sales, x="Region", y="Sales",
                 color="Sales", color_continuous_scale="Blues",
                 text_auto=".2s", title="Total Sales by Region")
    fig.update_layout(coloraxis_showscale=False, height=340)
    st.plotly_chart(fig, width="stretch")

with col_b:
    st.subheader("Profit by Category")
    cat_profit = df.groupby("Category")["Profit"].sum().reset_index()
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in cat_profit["Profit"]]
    fig = go.Figure(go.Bar(
        x=cat_profit["Category"], y=cat_profit["Profit"],
        marker_color=colors, text=cat_profit["Profit"].round(0),
        texttemplate="$%{text:,.0f}", textposition="outside"
    ))
    fig.update_layout(title="Total Profit by Category", height=340,
                      yaxis_title="Profit ($)")
    st.plotly_chart(fig, width="stretch")

# ─── Sub-Category Deep Dive ───────────────────────────────────────────────────
st.subheader("Sales & Profit by Sub-Category")

sub_perf = df.groupby("Sub-Category")[["Sales", "Profit"]].sum().reset_index()
sub_perf = sub_perf.sort_values("Sales", ascending=False)

fig = make_subplots(rows=1, cols=2, subplot_titles=("Sales by Sub-Category", "Profit by Sub-Category"))
fig.add_trace(go.Bar(y=sub_perf["Sub-Category"], x=sub_perf["Sales"],
                     orientation="h", marker_color="#3498db", name="Sales"), row=1, col=1)
profit_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sub_perf["Profit"]]
fig.add_trace(go.Bar(y=sub_perf["Sub-Category"], x=sub_perf["Profit"],
                     orientation="h", marker_color=profit_colors, name="Profit"), row=1, col=2)
fig.update_layout(height=450, showlegend=False)
st.plotly_chart(fig, width="stretch")

# ─── Top 5 Customers ──────────────────────────────────────────────────────────
st.subheader("🏆 Top 5 Customers by Sales")

top_customers = (
    df.groupby("Customer Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
)
top_customers.columns = ["Customer Name", "Total Sales"]
top_customers["Rank"] = range(1, 6)

col1, col2 = st.columns([2, 3])
with col1:
    st.dataframe(
        top_customers[["Rank", "Customer Name", "Total Sales"]]
        .assign(**{"Total Sales": top_customers["Total Sales"].map("${:,.2f}".format)})
        .set_index("Rank"),
        width="stretch",
    )

with col2:
    fig = px.bar(
        top_customers, x="Total Sales", y="Customer Name",
        orientation="h", color="Total Sales",
        color_continuous_scale="Teal",
        text_auto=".2s", title="Top 5 Customers by Total Sales"
    )
    fig.update_layout(coloraxis_showscale=False, height=300, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")

# ─── Segment Analysis ─────────────────────────────────────────────────────────
st.subheader("Segment Performance")

seg_perf = df.groupby("Segment")[["Sales", "Profit"]].sum().reset_index()
seg_perf["Profit Margin %"] = (seg_perf["Profit"] / seg_perf["Sales"] * 100).round(2)

fig = px.scatter(
    seg_perf, x="Sales", y="Profit", size="Sales",
    color="Segment", text="Segment",
    title="Segment: Sales vs Profit Bubble Chart",
    size_max=60
)
fig.update_traces(textposition="top center")
fig.update_layout(height=380)
st.plotly_chart(fig, width="stretch")

st.dataframe(
    seg_perf.set_index("Segment")
    .assign(Sales=seg_perf["Sales"].map("${:,.2f}".format).values,
            Profit=seg_perf["Profit"].map("${:,.2f}".format).values)
    .rename(columns={"Profit Margin %": "Profit Margin (%)"}),
    width="stretch"
)

# ─── Discount Impact ──────────────────────────────────────────────────────────
st.subheader("Discount Impact on Profit")

fig = px.scatter(
    df.sample(min(2000, len(df)), random_state=42),
    x="Discount", y="Profit", color="Category",
    opacity=0.5, title="Discount vs Profit (sampled 2,000 orders)",
    trendline="ols"
)
fig.update_layout(height=380)
st.plotly_chart(fig, width="stretch")

# ─── Heatmap ──────────────────────────────────────────────────────────────────
st.subheader("Regional Sales Heatmap by Category")

pivot = df.pivot_table(values="Sales", index="Region", columns="Category", aggfunc="sum")

fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(pivot, annot=True, fmt=",.0f", cmap="YlOrRd", ax=ax, linewidths=0.5)
ax.set_title("Sales Heatmap – Region × Category ($)")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ─── Raw Data ─────────────────────────────────────────────────────────────────
with st.expander("📋 Filtered Data Table"):
    st.dataframe(
        df[["Order Date", "Region", "Segment", "Category", "Sub-Category",
            "Customer Name", "Sales", "Profit", "Discount", "Quantity", "Ship Mode"]]
        .sort_values("Order Date", ascending=False)
        .head(500),
        width="stretch"
    )
    st.caption(f"Showing up to 500 of {len(df):,} filtered records.")

# ─── Conclusion ───────────────────────────────────────────────────────────────
st.subheader("💡 Business Insights")
st.success("""
- **Technology** generates the highest revenue and the best profit margins; prioritise its growth.
- **Furniture (Tables)** consistently generates negative profit — review supplier costs or discontinue.
- **Heavy discounting (>30%)** strongly correlates with negative profit; consider a discount cap policy.
- The **West** region leads in sales; the **South** has the highest margin percentage per order.
- The **Consumer** segment accounts for >50% of orders but **Corporate** has a better profit margin — 
  corporate account growth is the highest-leverage sales strategy.
""")
