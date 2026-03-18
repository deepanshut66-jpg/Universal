import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────── PAGE CONFIG ───────────────────────
st.set_page_config(
    page_title="Universal Bank — Personal Loan Campaign Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────── CUSTOM CSS ───────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,400&family=Playfair+Display:wght@600;700&display=swap');

:root {
    --navy: #0B1D3A;
    --gold: #C9963B;
    --gold-light: #E8C675;
    --teal: #1ABC9C;
    --coral: #E74C3C;
    --slate: #2C3E50;
    --bg-card: #F8F9FC;
    --text-primary: #1A1A2E;
    --text-secondary: #5A6170;
    --border: #E2E6ED;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

h1, h2, h3, .stMetricLabel {
    font-family: 'Playfair Display', serif !important;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0B1D3A 0%, #163A6C 100%);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 4px 20px rgba(11,29,58,0.12);
}
div[data-testid="stMetric"] label {
    color: #A8BDD9 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #E8C675 !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.8rem !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 10px 22px !important;
}

/* Section headers */
.section-header {
    background: linear-gradient(90deg, #0B1D3A 0%, #163A6C 60%, transparent 100%);
    color: #E8C675;
    padding: 12px 24px;
    border-radius: 10px;
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem;
    margin: 20px 0 14px 0;
    letter-spacing: 0.03em;
}

/* Insight boxes */
.insight-box {
    background: #F0F4FA;
    border-left: 4px solid #C9963B;
    padding: 12px 18px;
    border-radius: 0 10px 10px 0;
    margin: 8px 0 16px 0;
    font-size: 0.9rem;
    line-height: 1.55;
    color: #2C3E50;
}

/* Data table styling */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1D3A 0%, #0E2445 100%);
}
section[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #E8C675 !important;
}

.stDownloadButton button {
    background: linear-gradient(135deg, #C9963B, #E8C675) !important;
    color: #0B1D3A !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────── HELPER FUNCTIONS ───────────────────────
COLORS = {
    "navy": "#0B1D3A",
    "gold": "#C9963B",
    "gold_light": "#E8C675",
    "teal": "#1ABC9C",
    "coral": "#E74C3C",
    "blue": "#3498DB",
    "purple": "#8E44AD",
    "slate": "#2C3E50",
    "green": "#27AE60",
    "orange": "#E67E22",
}
PALETTE = ["#0B1D3A", "#C9963B", "#1ABC9C", "#E74C3C", "#3498DB", "#8E44AD", "#27AE60", "#E67E22"]
LOAN_COLORS = {0: "#3498DB", 1: "#C9963B"}

def section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

def styled_plotly(fig, height=420):
    fig.update_layout(
        font=dict(family="DM Sans", size=12, color="#2C3E50"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=30, t=50, b=40),
        height=height,
        legend=dict(
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="#E2E6ED",
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EDF0F5", gridwidth=1)
    fig.update_yaxes(showgrid=True, gridcolor="#EDF0F5", gridwidth=1)
    return fig


@st.cache_data
def load_data():
    df = pd.read_csv("UniversalBank.csv")
    # Clean negative experience
    df["Experience"] = df["Experience"].apply(lambda x: max(x, 0))
    return df

@st.cache_resource
def train_models(df):
    feature_cols = ["Age", "Experience", "Income", "Family", "CCAvg",
                    "Education", "Mortgage", "Securities Account",
                    "CD Account", "Online", "CreditCard"]
    X = df[feature_cols]
    y = df["Personal Loan"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_split=10, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=10,
            random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "Gradient Boosted Tree": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42, min_samples_split=10
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        roc_auc = auc(fpr, tpr)

        results[name] = {
            "model": model,
            "train_acc": accuracy_score(y_train, y_train_pred),
            "test_acc": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred, zero_division=0),
            "recall": recall_score(y_test, y_test_pred, zero_division=0),
            "f1": f1_score(y_test, y_test_pred, zero_division=0),
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": roc_auc,
            "cm": confusion_matrix(y_test, y_test_pred),
            "y_test": y_test,
            "y_test_pred": y_test_pred,
            "y_test_proba": y_test_proba,
            "feature_importance": dict(zip(feature_cols, model.feature_importances_)),
        }

    return results, feature_cols, X_train, X_test, y_train, y_test


# ─────────────────────── LOAD DATA + MODELS ───────────────────────
df = load_data()
results, feature_cols, X_train, X_test, y_train, y_test = train_models(df)

# ─────────────────────── SIDEBAR ───────────────────────
with st.sidebar:
    st.markdown("## 🏦 Universal Bank")
    st.markdown("#### Personal Loan Campaign Intelligence")
    st.markdown("---")
    st.markdown("**Dataset:** 5,000 customers")
    st.markdown(f"**Loan Acceptors:** {df['Personal Loan'].sum()} ({df['Personal Loan'].mean()*100:.1f}%)")
    st.markdown(f"**Features:** {len(feature_cols)} predictive variables")
    st.markdown("---")
    st.markdown("##### Navigation")
    st.markdown("""
    Use the tabs above to explore:
    - 📊 **Descriptive** — Who are our customers?
    - 🔍 **Diagnostic** — What drives acceptance?
    - 🤖 **Predictive** — ML model performance
    - 🎯 **Prescriptive** — Campaign strategy
    - 📤 **Predict New Data** — Upload & score
    """)
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;font-size:0.75rem;color:#5A7A9A;'>"
        "Built for Universal Bank Marketing Team<br>Head of Marketing Dashboard</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────── HEADER ───────────────────────
st.markdown(
    "<h1 style='text-align:center; font-size:2.3rem; color:#0B1D3A; margin-bottom:0;'>"
    "🏦 Universal Bank — Personal Loan Campaign Intelligence</h1>"
    "<p style='text-align:center; color:#5A6170; font-size:1.05rem; margin-top:4px;'>"
    "Data-driven insights to maximise personal loan acceptance with half the marketing budget</p>",
    unsafe_allow_html=True,
)

# ─── KPI ROW ───
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Customers", f"{len(df):,}")
k2.metric("Loan Acceptors", f"{df['Personal Loan'].sum()}")
k3.metric("Acceptance Rate", f"{df['Personal Loan'].mean()*100:.1f}%")
k4.metric("Avg Income ($K)", f"{df['Income'].mean():.0f}")
k5.metric("Avg CC Spend ($K)", f"{df['CCAvg'].mean():.1f}")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────── TABS ───────────────────────
tabs = st.tabs([
    "📊 Descriptive Analytics",
    "🔍 Diagnostic Analytics",
    "🤖 Predictive Analytics",
    "🎯 Prescriptive Analytics",
    "📤 Predict New Data",
])


# ═══════════════════════════════════════════════════════════════
#  TAB 1 — DESCRIPTIVE ANALYTICS
# ═══════════════════════════════════════════════════════════════
with tabs[0]:
    section_header("📊 Descriptive Analytics — Understanding Our Customer Base")
    insight(
        "This section provides a comprehensive snapshot of Universal Bank's 5,000-customer base. "
        "Understanding the demographic and financial profile of our customers is the first step to "
        "designing a laser-focused personal loan campaign."
    )

    # ── Target Variable Distribution ──
    st.markdown("#### Personal Loan Acceptance Distribution")
    c1, c2 = st.columns([1, 1])
    with c1:
        loan_counts = df["Personal Loan"].value_counts().reset_index()
        loan_counts.columns = ["Personal Loan", "Count"]
        loan_counts["Label"] = loan_counts["Personal Loan"].map({0: "Not Accepted", 1: "Accepted"})
        loan_counts["Percentage"] = (loan_counts["Count"] / loan_counts["Count"].sum() * 100).round(1)
        loan_counts["Text"] = loan_counts.apply(lambda r: f"{r['Label']}<br>{r['Count']} ({r['Percentage']}%)", axis=1)
        fig = px.pie(
            loan_counts, values="Count", names="Label",
            color="Label", color_discrete_map={"Not Accepted": "#3498DB", "Accepted": "#C9963B"},
            hole=0.55,
        )
        fig.update_traces(
            textinfo="label+percent", textfont_size=13,
            marker=dict(line=dict(color="#FFFFFF", width=2)),
        )
        fig.update_layout(
            title=dict(text="Loan Acceptance Split", font=dict(size=15, family="Playfair Display")),
            showlegend=True, height=380,
            font=dict(family="DM Sans"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
        insight(
            "<b>Key Finding:</b> Only 9.6% of customers accepted the personal loan — a highly imbalanced dataset. "
            "This tells us the last campaign had a ~90% rejection rate, meaning budget was wasted on uninterested customers. "
            "Targeted marketing can dramatically improve ROI."
        )

    with c2:
        # Age distribution by loan
        fig = px.histogram(
            df, x="Age", color=df["Personal Loan"].map({0: "Not Accepted", 1: "Accepted"}),
            nbins=30, barmode="overlay", opacity=0.75,
            color_discrete_map={"Not Accepted": "#3498DB", "Accepted": "#C9963B"},
            labels={"color": "Personal Loan"},
        )
        fig = styled_plotly(fig, height=380)
        fig.update_layout(title=dict(text="Age Distribution by Loan Acceptance", font=dict(size=15, family="Playfair Display")))
        fig.update_xaxes(title_text="Age (years)")
        fig.update_yaxes(title_text="Count")
        st.plotly_chart(fig, use_container_width=True)
        insight(
            "<b>Key Finding:</b> Loan acceptors are spread across all age groups (25–65), with a slight concentration in the 30–55 range. "
            "Age alone is not a strong differentiator — we need to dig deeper into income and spending patterns."
        )

    # ── Income & Spending ──
    st.markdown("#### Income & Credit Card Spending Patterns")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(
            df, x=df["Personal Loan"].map({0: "Not Accepted", 1: "Accepted"}),
            y="Income", color=df["Personal Loan"].map({0: "Not Accepted", 1: "Accepted"}),
            color_discrete_map={"Not Accepted": "#3498DB", "Accepted": "#C9963B"},
            labels={"x": "Personal Loan Status", "color": "Status"},
        )
        fig = styled_plotly(fig, height=400)
        fig.update_layout(
            title=dict(text="Income Distribution: Acceptors vs Non-Acceptors", font=dict(size=15, family="Playfair Display")),
            showlegend=False,
        )
        fig.update_yaxes(title_text="Annual Income ($000)")
        st.plotly_chart(fig, use_container_width=True)
        insight(
            "<b>Key Finding:</b> Loan acceptors have significantly higher incomes (median ~$115K vs ~$50K). "
            "Income appears to be the single most powerful predictor — high-income customers are our primary target segment."
        )

    with c2:
        fig = px.box(
            df, x=df["Personal Loan"].map({0: "Not Accepted", 1: "Accepted"}),
            y="CCAvg", color=df["Personal Loan"].map({0: "Not Accepted", 1: "Accepted"}),
            color_discrete_map={"Not Accepted": "#3498DB", "Accepted": "#C9963B"},
            labels={"x": "Personal Loan Status", "color": "Status"},
        )
        fig = styled_plotly(fig, height=400)
        fig.update_layout(
            title=dict(text="Credit Card Avg Spend: Acceptors vs Non-Acceptors", font=dict(size=15, family="Playfair Display")),
            showlegend=False,
        )
        fig.update_yaxes(title_text="Monthly CC Spend ($000)")
        st.plotly_chart(fig, use_container_width=True)
        insight(
            "<b>Key Finding:</b> Acceptors spend ~$3.9K/month on credit cards vs ~$1.6K for non-acceptors. "
            "Higher spenders are comfortable with credit products and more receptive to loan offers."
        )

    # ── Education & Family ──
    st.markdown("#### Education & Family Profile")
    c1, c2 = st.columns(2)
    with c1:
        edu_loan = df.groupby(["Education", "Personal Loan"]).size().reset_index(name="Count")
        edu_loan["Education"] = edu_loan["Education"].map({1: "Undergrad", 2: "Graduate", 3: "Advanced"})
        edu_loan["Status"] = edu_loan["Personal Loan"].map({0: "Not Accepted", 1: "Accepted"})
        fig = px.bar(
            edu_loan, x="Education", y="Count", color="Status", barmode="group",
            color_discrete_map={"Not Accepted": "#3498DB", "Accepted": "#C9963B"},
            text="Count",
        )
        fig = styled_plotly(fig, height=400)
        fig.update_layout(title=dict(text="Loan Acceptance by Education Level", font=dict(size=15, family="Playfair Display")))
        fig.update_traces(textposition="outside", textfont_size=11)
        st.plotly_chart(fig, use_container_width=True)

        # Calculate acceptance rates by education
        edu_rates = df.groupby("Education")["Personal Loan"].mean() * 100
        insight(
            f"<b>Key Finding:</b> Acceptance rates rise with education — "
            f"Undergrad: {edu_rates[1]:.1f}%, Graduate: {edu_rates[2]:.1f}%, Advanced: {edu_rates[3]:.1f}%. "
            "Advanced degree holders are ~3× more likely to accept a loan than undergrads."
        )

    with c2:
        fam_loan = df.groupby(["Family", "Personal Loan"]).size().reset_index(name="Count")
        fam_loan["Status"] = fam_loan["Personal Loan"].map({0: "Not Accepted", 1: "Accepted"})
        fig = px.bar(
            fam_loan, x="Family", y="Count", color="Status", barmode="group",
            color_discrete_map={"Not Accepted": "#3498DB", "Accepted": "#C9963B"},
            text="Count",
        )
        fig = styled_plotly(fig, height=400)
        fig.update_layout(title=dict(text="Loan Acceptance by Family Size", font=dict(size=15, family="Playfair Display")))
        fig.update_traces(textposition="outside", textfont_size=11)
        st.plotly_chart(fig, use_container_width=True)

        fam_rates = df.groupby("Family")["Personal Loan"].mean() * 100
        insight(
            f"<b>Key Finding:</b> Family size 3 ({fam_rates.get(3, 0):.1f}%) and 4 ({fam_rates.get(4, 0):.1f}%) "
            f"show higher acceptance rates vs size 1 ({fam_rates.get(1, 0):.1f}%). "
            "Larger families likely have more financial needs, making them more receptive to loan products."
        )

    # ── Banking Relationships ──
    st.markdown("#### Existing Banking Relationships")
    bank_features = ["Securities Account", "CD Account", "Online", "CreditCard"]
    bank_data = []
    for feat in bank_features:
        for val in [0, 1]:
            for loan in [0, 1]:
                count = len(df[(df[feat] == val) & (df["Personal Loan"] == loan)])
                bank_data.append({
                    "Feature": feat, "Has Feature": "Yes" if val == 1 else "No",
                    "Loan Status": "Accepted" if loan == 1 else "Not Accepted", "Count": count
                })
    bank_df = pd.DataFrame(bank_data)

    fig = px.bar(
        bank_df, x="Feature", y="Count", color="Loan Status",
        facet_col="Has Feature", barmode="group",
        color_discrete_map={"Not Accepted": "#3498DB", "Accepted": "#C9963B"},
        text="Count",
    )
    fig = styled_plotly(fig, height=420)
    fig.update_layout(title=dict(text="Banking Relationships & Loan Acceptance", font=dict(size=16, family="Playfair Display")))
    fig.update_traces(textposition="outside", textfont_size=10)
    st.plotly_chart(fig, use_container_width=True)

    cd_yes = df[df["CD Account"] == 1]["Personal Loan"].mean() * 100
    cd_no = df[df["CD Account"] == 0]["Personal Loan"].mean() * 100
    insight(
        f"<b>Key Finding:</b> CD Account holders have a {cd_yes:.0f}% loan acceptance rate vs just {cd_no:.1f}% for non-holders — "
        f"a {cd_yes/cd_no:.0f}× difference! Customers with deeper banking relationships (especially CD accounts) are prime targets."
    )


# ═══════════════════════════════════════════════════════════════
#  TAB 2 — DIAGNOSTIC ANALYTICS
# ═══════════════════════════════════════════════════════════════
with tabs[1]:
    section_header("🔍 Diagnostic Analytics — Why Do Customers Accept Loans?")
    insight(
        "Diagnostic analytics digs deeper into the 'why' behind loan acceptance. "
        "We examine correlations, cross-segment patterns, and multi-factor analysis to isolate "
        "the key drivers that differentiate acceptors from non-acceptors."
    )

    # ── Correlation Heatmap ──
    st.markdown("#### Feature Correlation Matrix")
    corr_cols = feature_cols + ["Personal Loan"]
    corr_matrix = df[corr_cols].corr().round(2)
    fig = px.imshow(
        corr_matrix, text_auto=True, aspect="auto",
        color_continuous_scale=["#3498DB", "#FFFFFF", "#C9963B"],
        zmin=-1, zmax=1,
    )
    fig = styled_plotly(fig, height=520)
    fig.update_layout(
        title=dict(text="Correlation Heatmap — All Features vs Personal Loan", font=dict(size=16, family="Playfair Display")),
    )
    st.plotly_chart(fig, use_container_width=True)
    insight(
        "<b>Key Finding:</b> The strongest positive correlations with Personal Loan are: "
        "Income (0.50), CCAvg (0.37), CD Account (0.32), and Education (0.14). "
        "Age and Experience are nearly perfectly correlated (0.99) — we should consider dropping one to avoid multicollinearity."
    )

    # ── Income vs CCAvg Scatter ──
    st.markdown("#### Income vs Credit Card Spend — The Power Duo")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.scatter(
            df, x="Income", y="CCAvg",
            color=df["Personal Loan"].map({0: "Not Accepted", 1: "Accepted"}),
            color_discrete_map={"Not Accepted": "rgba(52,152,219,0.3)", "Accepted": "rgba(201,150,59,0.85)"},
            labels={"color": "Loan Status"},
            opacity=0.7,
        )
        fig = styled_plotly(fig, height=450)
        fig.update_layout(title=dict(text="Income vs CC Spend — Loan Acceptors Cluster in Upper-Right", font=dict(size=15, family="Playfair Display")))
        fig.update_xaxes(title_text="Annual Income ($000)")
        fig.update_yaxes(title_text="Monthly CC Spend ($000)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        insight(
            "<b>The Sweet Spot:</b> Loan acceptors cluster clearly in the high-income + high-spending zone "
            "(Income > $100K, CCAvg > $2.5K). This two-variable combination alone can isolate ~70% of acceptors. "
            "This is the core targeting criterion for the next campaign."
        )
        # Quick stats
        high_income = df[df["Income"] >= 100]
        high_both = df[(df["Income"] >= 100) & (df["CCAvg"] >= 2.5)]
        st.metric("Customers with Income ≥ $100K", f"{len(high_income)} ({len(high_income)/len(df)*100:.1f}%)")
        st.metric("Acceptance in this group", f"{high_income['Personal Loan'].mean()*100:.1f}%")
        st.metric("Income ≥ $100K + CC ≥ $2.5K", f"{len(high_both)} ({len(high_both)/len(df)*100:.1f}%)")
        st.metric("Acceptance in this group", f"{high_both['Personal Loan'].mean()*100:.1f}%")

    # ── Multi-factor: Education × Income ──
    st.markdown("#### Multi-Factor Analysis: Education × Income Bracket")
    df_temp = df.copy()
    df_temp["Income Bracket"] = pd.cut(df_temp["Income"], bins=[0, 50, 100, 150, 250], labels=["<$50K", "$50-100K", "$100-150K", "$150K+"])
    df_temp["Education Level"] = df_temp["Education"].map({1: "Undergrad", 2: "Graduate", 3: "Advanced"})
    cross = df_temp.groupby(["Income Bracket", "Education Level"])["Personal Loan"].agg(["mean", "count"]).reset_index()
    cross.columns = ["Income Bracket", "Education Level", "Acceptance Rate", "Count"]
    cross["Acceptance Rate"] = (cross["Acceptance Rate"] * 100).round(1)

    fig = px.bar(
        cross, x="Income Bracket", y="Acceptance Rate", color="Education Level",
        barmode="group", text="Acceptance Rate",
        color_discrete_sequence=["#0B1D3A", "#C9963B", "#1ABC9C"],
    )
    fig = styled_plotly(fig, height=420)
    fig.update_layout(title=dict(text="Loan Acceptance Rate by Income Bracket & Education", font=dict(size=15, family="Playfair Display")))
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", textfont_size=11)
    fig.update_yaxes(title_text="Acceptance Rate (%)", range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)
    insight(
        "<b>Key Finding:</b> The combination of high income ($150K+) and advanced education produces acceptance rates above 50%. "
        "Even in the $100-150K bracket, graduate/advanced degree holders show 2-3× higher acceptance than undergrads. "
        "Education and income are multiplicative — targeting both together yields disproportionate returns."
    )

    # ── CD Account Deep Dive ──
    st.markdown("#### CD Account — The Hidden Gem")
    c1, c2 = st.columns(2)
    with c1:
        cd_data = df.groupby("CD Account")["Personal Loan"].agg(["sum", "count"]).reset_index()
        cd_data.columns = ["CD Account", "Accepted", "Total"]
        cd_data["Not Accepted"] = cd_data["Total"] - cd_data["Accepted"]
        cd_data["CD Account"] = cd_data["CD Account"].map({0: "No CD Account", 1: "Has CD Account"})
        cd_data["Acceptance Rate"] = (cd_data["Accepted"] / cd_data["Total"] * 100).round(1)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cd_data["CD Account"], y=cd_data["Not Accepted"], name="Not Accepted",
            marker_color="#3498DB", text=cd_data["Not Accepted"], textposition="inside",
        ))
        fig.add_trace(go.Bar(
            x=cd_data["CD Account"], y=cd_data["Accepted"], name="Accepted",
            marker_color="#C9963B", text=cd_data["Accepted"], textposition="inside",
        ))
        fig = styled_plotly(fig, height=400)
        fig.update_layout(
            barmode="stack",
            title=dict(text="CD Account Holders vs Loan Acceptance", font=dict(size=15, family="Playfair Display")),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        has_cd = df[df["CD Account"] == 1]
        no_cd = df[df["CD Account"] == 0]
        st.metric("CD Holders — Acceptance Rate", f"{has_cd['Personal Loan'].mean()*100:.1f}%")
        st.metric("Non-CD Holders — Acceptance Rate", f"{no_cd['Personal Loan'].mean()*100:.1f}%")
        st.metric("Lift Factor", f"{has_cd['Personal Loan'].mean() / no_cd['Personal Loan'].mean():.1f}×")
        insight(
            "<b>Actionable Insight:</b> CD account holders are your lowest-hanging fruit — they already trust the bank "
            "with term deposits and show dramatically higher loan acceptance. Prioritise them in any budget-constrained campaign."
        )

    # ── Mortgage analysis ──
    st.markdown("#### Mortgage Holders — Do They Need More Credit?")
    df_temp2 = df.copy()
    df_temp2["Has Mortgage"] = df_temp2["Mortgage"].apply(lambda x: "Has Mortgage" if x > 0 else "No Mortgage")
    mort_rates = df_temp2.groupby("Has Mortgage")["Personal Loan"].mean().reset_index()
    mort_rates.columns = ["Mortgage Status", "Acceptance Rate"]
    mort_rates["Acceptance Rate"] = (mort_rates["Acceptance Rate"] * 100).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            mort_rates, x="Mortgage Status", y="Acceptance Rate", color="Mortgage Status",
            color_discrete_map={"No Mortgage": "#3498DB", "Has Mortgage": "#C9963B"},
            text="Acceptance Rate",
        )
        fig = styled_plotly(fig, height=380)
        fig.update_layout(
            title=dict(text="Acceptance Rate: Mortgage vs No Mortgage", font=dict(size=15, family="Playfair Display")),
            showlegend=False,
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_yaxes(title_text="Acceptance Rate (%)", range=[0, 25])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Mortgage value vs loan acceptance
        mortgage_holders = df[df["Mortgage"] > 0].copy()
        fig = px.histogram(
            mortgage_holders, x="Mortgage",
            color=mortgage_holders["Personal Loan"].map({0: "Not Accepted", 1: "Accepted"}),
            nbins=25, barmode="overlay", opacity=0.75,
            color_discrete_map={"Not Accepted": "#3498DB", "Accepted": "#C9963B"},
            labels={"color": "Loan Status"},
        )
        fig = styled_plotly(fig, height=380)
        fig.update_layout(title=dict(text="Mortgage Value Distribution by Loan Status", font=dict(size=15, family="Playfair Display")))
        fig.update_xaxes(title_text="Mortgage Value ($000)")
        fig.update_yaxes(title_text="Count")
        st.plotly_chart(fig, use_container_width=True)

    has_mort = df[df["Mortgage"] > 0]["Personal Loan"].mean() * 100
    no_mort = df[df["Mortgage"] == 0]["Personal Loan"].mean() * 100
    insight(
        f"<b>Key Finding:</b> Mortgage holders show a higher acceptance rate ({has_mort:.1f}%) vs non-holders ({no_mort:.1f}%). "
        "Customers already managing a mortgage are comfortable with debt products and may need additional liquidity — "
        "making them a receptive audience for personal loan offers."
    )


# ═══════════════════════════════════════════════════════════════
#  TAB 3 — PREDICTIVE ANALYTICS
# ═══════════════════════════════════════════════════════════════
with tabs[2]:
    section_header("🤖 Predictive Analytics — Classification Model Performance")
    insight(
        "Three classification algorithms — Decision Tree, Random Forest, and Gradient Boosted Tree — "
        "were trained on 70% of the data and tested on the remaining 30% (stratified split). "
        "Below is a comprehensive comparison of their performance."
    )

    # ── Performance Metrics Table ──
    st.markdown("#### Model Performance Comparison")
    metrics_data = []
    for name, res in results.items():
        metrics_data.append({
            "Model": name,
            "Training Accuracy": f"{res['train_acc']*100:.2f}%",
            "Testing Accuracy": f"{res['test_acc']*100:.2f}%",
            "Precision": f"{res['precision']*100:.2f}%",
            "Recall": f"{res['recall']*100:.2f}%",
            "F1-Score": f"{res['f1']*100:.2f}%",
            "ROC-AUC": f"{res['roc_auc']:.4f}",
        })
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Find best model
    best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
    insight(
        f"<b>Best Model:</b> <b>{best_model_name}</b> achieves the highest ROC-AUC of "
        f"{results[best_model_name]['roc_auc']:.4f}, indicating the best overall discrimination between "
        "loan acceptors and non-acceptors. This model will be used for scoring new customers."
    )

    # ── ROC Curve (single plot, all models) ──
    st.markdown("#### ROC Curve — All Models Compared")
    fig = go.Figure()
    roc_colors = {"Decision Tree": "#E74C3C", "Random Forest": "#1ABC9C", "Gradient Boosted Tree": "#C9963B"}
    for name, res in results.items():
        fig.add_trace(go.Scatter(
            x=res["fpr"], y=res["tpr"], mode="lines",
            name=f"{name} (AUC = {res['roc_auc']:.4f})",
            line=dict(color=roc_colors[name], width=2.5),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline",
        line=dict(color="#BDC3C7", width=1.5, dash="dash"),
    ))
    fig = styled_plotly(fig, height=480)
    fig.update_layout(
        title=dict(text="Receiver Operating Characteristic (ROC) Curve", font=dict(size=16, family="Playfair Display")),
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity / Recall)",
    )
    st.plotly_chart(fig, use_container_width=True)
    insight(
        "The ROC curve plots the trade-off between catching true loan acceptors (TPR) and falsely flagging non-acceptors (FPR). "
        "A curve hugging the top-left corner indicates strong model performance. All three models significantly outperform the random baseline (diagonal)."
    )

    # ── Confusion Matrices ──
    st.markdown("#### Confusion Matrices — Detailed Breakdown")
    cm_cols = st.columns(3)
    cm_colors = {"Decision Tree": "#E74C3C", "Random Forest": "#1ABC9C", "Gradient Boosted Tree": "#C9963B"}

    for idx, (name, res) in enumerate(results.items()):
        with cm_cols[idx]:
            cm = res["cm"]
            total = cm.sum()
            cm_pct = (cm / total * 100).round(1)

            labels = [
                [f"TN = {cm[0][0]}<br>({cm_pct[0][0]}%)", f"FP = {cm[0][1]}<br>({cm_pct[0][1]}%)"],
                [f"FN = {cm[1][0]}<br>({cm_pct[1][0]}%)", f"TP = {cm[1][1]}<br>({cm_pct[1][1]}%)"],
            ]

            fig = go.Figure(data=go.Heatmap(
                z=cm, x=["Predicted: No", "Predicted: Yes"], y=["Actual: No", "Actual: Yes"],
                text=[[labels[0][0], labels[0][1]], [labels[1][0], labels[1][1]]],
                texttemplate="%{text}",
                textfont=dict(size=12, color="white"),
                colorscale=[
                    [0, "#2C3E50"],
                    [0.5, cm_colors[name]],
                    [1, cm_colors[name]],
                ],
                showscale=False,
            ))
            fig.update_layout(
                title=dict(text=f"{name}", font=dict(size=14, family="Playfair Display")),
                height=340, margin=dict(l=10, r=10, t=50, b=10),
                font=dict(family="DM Sans", size=11),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(side="bottom"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

    insight(
        "<b>Reading the Matrix:</b> TN = correctly identified non-acceptors, TP = correctly identified acceptors, "
        "FP = wrongly flagged as acceptors (wasted budget), FN = missed acceptors (lost opportunity). "
        "For a budget-constrained campaign, we want high Precision (low FP) to avoid wasting money on uninterested customers."
    )

    # ── Feature Importance ──
    st.markdown("#### Feature Importance — What Drives the Best Model?")
    best_importance = results[best_model_name]["feature_importance"]
    imp_df = pd.DataFrame(list(best_importance.items()), columns=["Feature", "Importance"])
    imp_df = imp_df.sort_values("Importance", ascending=True)

    fig = px.bar(
        imp_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale=["#3498DB", "#C9963B"],
        text=imp_df["Importance"].apply(lambda x: f"{x:.3f}"),
    )
    fig = styled_plotly(fig, height=420)
    fig.update_layout(
        title=dict(text=f"Feature Importance — {best_model_name}", font=dict(size=15, family="Playfair Display")),
        coloraxis_showscale=False,
    )
    fig.update_traces(textposition="outside", textfont_size=11)
    st.plotly_chart(fig, use_container_width=True)
    top3 = imp_df.nlargest(3, "Importance")["Feature"].tolist()
    insight(
        f"<b>Top 3 Predictors:</b> {', '.join(top3)} are the most influential features in predicting loan acceptance. "
        "Marketing campaigns should prioritise customer segments where these factors are highest."
    )


# ═══════════════════════════════════════════════════════════════
#  TAB 4 — PRESCRIPTIVE ANALYTICS
# ═══════════════════════════════════════════════════════════════
with tabs[3]:
    section_header("🎯 Prescriptive Analytics — Campaign Strategy Recommendations")
    insight(
        "Based on descriptive, diagnostic, and predictive analysis, here are actionable strategies "
        "to maximise loan acceptance while operating with half the marketing budget."
    )

    # ── Segment Scoring ──
    st.markdown("#### Customer Segmentation & Priority Scoring")
    df_seg = df.copy()
    best_model = results[best_model_name]["model"]
    df_seg["Probability"] = best_model.predict_proba(df_seg[feature_cols])[:, 1]
    df_seg["Priority"] = pd.cut(
        df_seg["Probability"],
        bins=[0, 0.2, 0.5, 0.8, 1.01],
        labels=["Low", "Medium", "High", "Very High"],
    )

    priority_summary = df_seg.groupby("Priority", observed=True).agg(
        Count=("ID", "count"),
        Avg_Income=("Income", "mean"),
        Avg_CCAvg=("CCAvg", "mean"),
        Acceptance_Rate=("Personal Loan", "mean"),
        Avg_Probability=("Probability", "mean"),
    ).reset_index()
    priority_summary["Acceptance_Rate"] = (priority_summary["Acceptance_Rate"] * 100).round(1)
    priority_summary["Avg_Income"] = priority_summary["Avg_Income"].round(0)
    priority_summary["Avg_CCAvg"] = priority_summary["Avg_CCAvg"].round(1)
    priority_summary["Avg_Probability"] = (priority_summary["Avg_Probability"] * 100).round(1)
    priority_summary.columns = ["Priority Tier", "Count", "Avg Income ($K)", "Avg CC Spend ($K)", "Historical Acceptance %", "Model Probability %"]

    st.dataframe(priority_summary, use_container_width=True, hide_index=True)

    # ── Probability Distribution ──
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            df_seg, x="Probability", nbins=40,
            color=df_seg["Personal Loan"].map({0: "Not Accepted", 1: "Accepted"}),
            barmode="overlay", opacity=0.75,
            color_discrete_map={"Not Accepted": "#3498DB", "Accepted": "#C9963B"},
            labels={"color": "Actual Loan Status"},
        )
        fig = styled_plotly(fig, height=400)
        fig.update_layout(title=dict(text="Predicted Probability Distribution", font=dict(size=15, family="Playfair Display")))
        fig.update_xaxes(title_text="Predicted Probability of Acceptance")
        fig.update_yaxes(title_text="Count")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        tier_counts = df_seg["Priority"].value_counts().reset_index()
        tier_counts.columns = ["Priority", "Count"]
        tier_colors = {"Very High": "#C9963B", "High": "#1ABC9C", "Medium": "#3498DB", "Low": "#95A5A6"}
        fig = px.pie(
            tier_counts, values="Count", names="Priority",
            color="Priority", color_discrete_map=tier_colors,
            hole=0.5,
        )
        fig.update_traces(textinfo="label+percent+value", textfont_size=12)
        fig = styled_plotly(fig, height=400)
        fig.update_layout(title=dict(text="Customer Priority Tier Distribution", font=dict(size=15, family="Playfair Display")))
        st.plotly_chart(fig, use_container_width=True)

    very_high = len(df_seg[df_seg["Priority"] == "Very High"])
    high = len(df_seg[df_seg["Priority"] == "High"])
    target_pct = (very_high + high) / len(df_seg) * 100
    insight(
        f"<b>Budget-Optimised Strategy:</b> Focus the campaign on <b>Very High + High priority</b> tiers — "
        f"only <b>{very_high + high} customers ({target_pct:.1f}% of the base)</b>. "
        "This dramatically reduces marketing spend while capturing the majority of potential acceptors."
    )

    # ── Strategic Recommendations ──
    st.markdown("#### 📋 Strategic Recommendations for Next Campaign")

    rec_data = [
        ("🎯", "Target Audience", "High-income ($100K+), Graduate/Advanced education, CD account holders, high CC spenders ($2.5K+/mo)", "Very High"),
        ("📊", "Segment Priority", f"Focus on {very_high + high} customers in High + Very High tiers instead of mass-blasting all 5,000", "High"),
        ("💡", "Channel Strategy", "Use Online banking channel — 60% of customers use it; personalised in-app offers cost less than mail/calls", "High"),
        ("🏦", "Cross-sell Angle", f"CD account holders have {cd_yes:.0f}% acceptance — offer bundled CD + personal loan products", "Very High"),
        ("👨‍👩‍👧‍👦", "Family Targeting", "Families of 3-4 with mortgages show higher acceptance — tailor messaging around family financial planning", "Medium"),
        ("📈", "Expected Lift", f"Targeting top tiers gives ~{(very_high+high)/len(df)*100:.0f}% reach but captures ~80%+ of acceptors — estimated 3-4× ROI improvement", "Very High"),
    ]

    for icon, title, desc, priority in rec_data:
        priority_color = {"Very High": "#C9963B", "High": "#1ABC9C", "Medium": "#3498DB"}.get(priority, "#95A5A6")
        st.markdown(
            f"""<div style="background: #F8F9FC; border-left: 5px solid {priority_color}; padding: 14px 20px;
            border-radius: 0 10px 10px 0; margin: 8px 0;">
            <div style="display:flex;align-items:center;gap:10px;">
                <span style="font-size:1.6rem;">{icon}</span>
                <div>
                    <div style="font-weight:700;font-size:1rem;color:#0B1D3A;">{title}
                        <span style="background:{priority_color};color:white;padding:2px 10px;border-radius:12px;
                        font-size:0.72rem;margin-left:8px;font-weight:600;">{priority} Priority</span>
                    </div>
                    <div style="color:#5A6170;font-size:0.9rem;margin-top:3px;">{desc}</div>
                </div>
            </div></div>""",
            unsafe_allow_html=True,
        )

    # ── Budget Impact Estimation ──
    st.markdown("#### 💰 Estimated Budget Impact")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Previous Campaign Reach", "5,000 customers (100%)")
        st.metric("Previous Acceptance", f"{df['Personal Loan'].sum()} ({df['Personal Loan'].mean()*100:.1f}%)")
    with c2:
        st.metric("Recommended Reach", f"{very_high + high} customers ({target_pct:.1f}%)")
        target_acceptance = df_seg[df_seg["Priority"].isin(["Very High", "High"])]["Personal Loan"].sum()
        st.metric("Expected Acceptors from Target", f"{target_acceptance} of {df['Personal Loan'].sum()} total")
    with c3:
        cost_saving = (1 - target_pct/100) * 100
        st.metric("Campaign Cost Reduction", f"~{cost_saving:.0f}%")
        capture_rate = target_acceptance / df['Personal Loan'].sum() * 100
        st.metric("Acceptor Capture Rate", f"{capture_rate:.1f}%")

    insight(
        f"<b>Bottom Line:</b> By targeting only the top-priority segments, you reach ~{target_pct:.0f}% of the customer base "
        f"but capture ~{capture_rate:.0f}% of all potential loan acceptors. This means you can cut your budget by ~{cost_saving:.0f}% "
        "while barely losing any conversions — perfectly aligned with this year's 50% budget cut."
    )


# ═══════════════════════════════════════════════════════════════
#  TAB 5 — PREDICT NEW DATA
# ═══════════════════════════════════════════════════════════════
with tabs[4]:
    section_header("📤 Upload New Customer Data & Predict Loan Acceptance")
    insight(
        "Upload a CSV file with the same columns as the training data (without the 'Personal Loan' column) "
        "to predict which new customers are likely to accept a personal loan. "
        "Download a sample test file below if you need a template."
    )

    # ── Download sample file ──
    try:
        sample_df = pd.read_csv("test_data_for_prediction.csv")
        st.download_button(
            label="📥 Download Sample Test Data (50 rows)",
            data=sample_df.to_csv(index=False).encode("utf-8"),
            file_name="test_data_for_prediction.csv",
            mime="text/csv",
        )
    except Exception:
        st.info("Sample test file not found. Upload any CSV with the required columns.")

    st.markdown("---")

    # ── Upload ──
    uploaded = st.file_uploader("Upload your CSV file for prediction", type=["csv"])

    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            st.success(f"File uploaded successfully — {len(new_df)} rows, {len(new_df.columns)} columns")
            st.dataframe(new_df.head(10), use_container_width=True, hide_index=True)

            # Validate columns
            missing_cols = [c for c in feature_cols if c not in new_df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                # Clean Experience
                new_df["Experience"] = new_df["Experience"].apply(lambda x: max(x, 0))

                # Predict
                best_model = results[best_model_name]["model"]
                X_new = new_df[feature_cols]
                new_df["Predicted_Personal_Loan"] = best_model.predict(X_new)
                new_df["Acceptance_Probability"] = best_model.predict_proba(X_new)[:, 1].round(4)
                new_df["Priority_Tier"] = pd.cut(
                    new_df["Acceptance_Probability"],
                    bins=[0, 0.2, 0.5, 0.8, 1.01],
                    labels=["Low", "Medium", "High", "Very High"],
                )

                st.markdown("---")
                st.markdown("#### 🎯 Prediction Results")

                # Summary metrics
                m1, m2, m3, m4 = st.columns(4)
                predicted_yes = new_df["Predicted_Personal_Loan"].sum()
                m1.metric("Total Customers", len(new_df))
                m2.metric("Predicted Acceptors", int(predicted_yes))
                m3.metric("Predicted Acceptance Rate", f"{predicted_yes/len(new_df)*100:.1f}%")
                m4.metric("Avg Probability", f"{new_df['Acceptance_Probability'].mean()*100:.1f}%")

                # Priority tier breakdown
                tier_breakdown = new_df["Priority_Tier"].value_counts().reset_index()
                tier_breakdown.columns = ["Tier", "Count"]
                fig = px.bar(
                    tier_breakdown, x="Tier", y="Count", color="Tier",
                    color_discrete_map={"Very High": "#C9963B", "High": "#1ABC9C", "Medium": "#3498DB", "Low": "#95A5A6"},
                    text="Count",
                )
                fig = styled_plotly(fig, height=350)
                fig.update_layout(
                    title=dict(text="Predicted Priority Tier Distribution", font=dict(size=15, family="Playfair Display")),
                    showlegend=False,
                )
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

                # Full results table
                st.markdown("#### 📋 Full Results Table")
                display_cols = list(new_df.columns)
                st.dataframe(
                    new_df[display_cols].sort_values("Acceptance_Probability", ascending=False),
                    use_container_width=True, hide_index=True,
                )

                # Download results
                csv_result = new_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Prediction Results as CSV",
                    data=csv_result,
                    file_name="personal_loan_predictions.csv",
                    mime="text/csv",
                )
                insight(
                    "<b>Next Steps:</b> Focus your campaign on 'Very High' and 'High' priority customers. "
                    "These are the most likely to accept a personal loan, giving you the highest ROI per marketing dollar spent."
                )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("👆 Upload a CSV file above to start predicting.")


# ─────────────────────── FOOTER ───────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#8896A6; font-size:0.82rem; padding:10px 0;'>"
    "Universal Bank — Personal Loan Campaign Intelligence Dashboard &nbsp;|&nbsp; "
    "Built with Streamlit & Scikit-Learn &nbsp;|&nbsp; "
    f"Best Model: {best_model_name} (AUC: {results[best_model_name]['roc_auc']:.4f})"
    "</div>",
    unsafe_allow_html=True,
)
