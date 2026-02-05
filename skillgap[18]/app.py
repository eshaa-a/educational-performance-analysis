# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model import SkillGapModel
from curriculum import CurriculumOptimizer

st.set_page_config(
    page_title="Skill Gap Detection — Professional",
    layout="wide",
    initial_sidebar_state="expanded"
)

PALETTE = {
    "bg": "#0f172a",
    "panel": "#020617",
    "accent": "#38bdf8",
    "accent2": "#60a5fa",
    "accent3": "#93c5fd",
    "text": "#e5e7eb",
    "muted": "#94a3b8"
}

st.markdown(f"""
<style>
.stApp {{
    background: {PALETTE['bg']};
    color: {PALETTE['text']};
}}

h1, h2, h3 {{
    color: {PALETTE['accent']};
    font-weight: 600;
}}

.section-box {{
    background: {PALETTE['panel']};
    padding: 1.2rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}}

.stButton>button {{
    background-color: {PALETTE['accent']};
    color: {PALETTE['bg']};
    border-radius: 10px;
    font-weight: 600;
}}

[data-testid="metric-container"] {{
    background-color: {PALETTE['panel']};
    border-radius: 12px;
    padding: 14px;
}}
</style>
""", unsafe_allow_html=True)

st.title("Skill Gap Detection & Curriculum Optimization")
st.caption(
    "PCA-based dimensionality reduction, clustering-based gap identification, "
    "and supervised evaluation for employability prediction."
)

st.sidebar.header("Data Input")
mode = st.sidebar.radio(
    "Select input mode",
    ("Default dataset", "Upload CSV", "Manual Entry")
)

DEFAULT_CSV = "data.csv"

model = SkillGapModel(n_components=2, n_clusters=4)
optimizer = CurriculumOptimizer()
df = None

#Datset Loading

if mode == "Default dataset":
    try:
        df = pd.read_csv(DEFAULT_CSV)
        st.sidebar.success("Default dataset loaded")
    except FileNotFoundError:
        st.sidebar.error("Default dataset not found")

elif mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload learner CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.sidebar.success("Dataset uploaded")

elif mode == "Manual Entry":
    st.sidebar.info("Enter skill scores (0–10)")
    manual = {}
    with st.sidebar.form("manual_form"):
        for skill in model.SKILLS:
            manual[skill] = st.number_input(
            skill.replace("_", " ").title(),
            min_value=0.0,
            max_value=10.0,
            value=6.0,
            step=0.5
    )
        submit = st.form_submit_button("Analyze")
    if submit:
        df = pd.DataFrame([manual])

if df is None:
    st.info("Please provide data to continue.")
    st.stop()

#Preview
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

with st.spinner("Running PCA and clustering..."):
    model.load_dataframe(df)
    model.preprocess()
    X = model.preprocess(use_pca=True)
    model.run_kmeans()

st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("Summary Metrics")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Students", len(model.df))
with col2:
    st.metric("Skills", len(model.skill_cols))
with col3:
    metrics = model.supervised_accuracy(threshold=7.0)
    acc = metrics["Accuracy"]
    st.metric("Accuracy", f"{acc*100:.2f}%")

st.caption("Supervised classification using PCA features")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PCA Scatter ----------------
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("PCA Cluster Visualization")

pca_df = pd.DataFrame(model.X_pca[:, :2], columns=["PC1", "PC2"])
pca_df["Cluster"] = model.labels.astype(str)

fig_scatter = px.scatter(
    pca_df, x="PC1", y="PC2",
    color="Cluster",
    opacity=0.85
)
fig_scatter.update_layout(
    plot_bgcolor=PALETTE['bg'],
    paper_bgcolor=PALETTE['bg'],
    font_color=PALETTE['text']
)

st.plotly_chart(fig_scatter, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

#Boxplot
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("Skill Score Distribution (Boxplot)")

skill_columns = model.skill_cols

plot_df = model.df[skill_columns].sample(
    n=min(50, len(model.df)),
    random_state=42
)
long_df = plot_df.melt(var_name="Skill", value_name="Score")
fig_box = px.box(
    long_df,
    x="Skill",
    y="Score",
    points="outliers"
)
fig_box.update_layout(
    plot_bgcolor=PALETTE['bg'],
    paper_bgcolor=PALETTE['bg'],
    font_color=PALETTE['text'],
    xaxis_tickangle=-45
)
st.plotly_chart(fig_box, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

#Histogram
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("Overall Skill Score Distribution")

fig_hist = px.histogram(
    long_df,
    x="Score",
    nbins=20,
    marginal="box"
)
fig_hist.update_layout(
    plot_bgcolor=PALETTE['bg'],
    paper_bgcolor=PALETTE['bg'],
    font_color=PALETTE['text']
)
st.plotly_chart(fig_hist, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

#Heatmap
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("Skill Correlation Heatmap")

corr = df.corr(numeric_only=True)

fig_heat = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="Blues"
)
fig_heat.update_layout(
    plot_bgcolor=PALETTE['bg'],
    paper_bgcolor=PALETTE['bg'],
    font_color=PALETTE['text']
)
st.plotly_chart(fig_heat, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

#Severity
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("Skill Gap Severity")

gap_df = model.compute_skill_gaps(ideal_target=8.0)
gap_display = gap_df.reset_index().rename(columns={"index": "Skill"})

fig_gap = px.bar(
    gap_display.sort_values("gap"),
    x="gap", y="Skill",
    orientation="h",
    text="gap"
)
fig_gap.update_layout(
    plot_bgcolor=PALETTE['bg'],
    paper_bgcolor=PALETTE['bg'],
    font_color=PALETTE['text']
)
st.plotly_chart(fig_gap, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

#Curriculum Recommendations
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("Curriculum Recommendations")

recommendations = optimizer.recommend(gap_df, top_k=6)
for skill, modules in recommendations.items():
    st.markdown(f"**{skill.replace('_',' ').title()}**")
    for m in modules:
        st.write(f"• {m}")

st.markdown('</div>', unsafe_allow_html=True)
