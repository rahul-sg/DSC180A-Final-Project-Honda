# interactive_dashboard.py
"""
Interactive Evaluation Dashboard (Streamlit)

Run with:
    streamlit run src/visualization/interactive_dashboard.py
"""

import streamlit as st
import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# ============================================================
# Utility Functions
# ============================================================

def load_iteration_summaries(iter_dir):
    texts = {}
    for fname in sorted(Path(iter_dir).glob("*.txt")):
        texts[fname.stem] = fname.read_text().strip()
    return texts


def get_embedding(text):
    """Compute OpenAI embedding for semantic drift."""
    if not text.strip():
        return np.zeros(1536)

    emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return np.array(emb.data[0].embedding)


def compute_semantic_drift(iter_texts):
    names = sorted(
        iter_texts.keys(),
        key=lambda x: int(x.split("_")[1]) if x.startswith("iter_") else 999
    )

    embeddings = [get_embedding(iter_texts[n]) for n in names]
    s0_emb = embeddings[0]

    sims = [cosine_similarity([s0_emb], [emb])[0][0] for emb in embeddings]
    drift = [1 - sim for sim in sims]

    return names, drift

# ============================================================
# Streamlit Layout
# ============================================================

st.set_page_config(page_title="Interactive Evaluation Dashboard", layout="wide")

st.title("📊 Interactive Evaluation Dashboard")
st.markdown("---")

# Load lectures
root_dir = Path("data/summaries/refined_iterations")
lectures = sorted([p.name for p in root_dir.glob("*") if p.is_dir()])

if not lectures:
    st.error("No lecture directories found in data/summaries/refined_iterations/")
    st.stop()

lecture_id = st.sidebar.selectbox("Select Lecture:", lectures)

st.subheader(f"Lecture: {lecture_id[-1:]}")

iter_dir = root_dir / lecture_id
result_json = iter_dir / "result.json"

# Load result.json
if not result_json.exists():
    st.error(f"result.json not found for lecture '{lecture_id}'")
    st.stop()

result = json.loads(result_json.read_text())
iter_texts = load_iteration_summaries(iter_dir)

signals = result["signals"]
rubric = result["rubric"]
agreement = result["agreement"]["agreement_1to5"]

# Word counts for each iteration
iter_lengths = {k: len(v.split()) for k, v in iter_texts.items()}
iter_steps = sorted(
    iter_lengths.keys(),
    key=lambda x: int(x.split("_")[1]) if x.startswith("iter_") else 999
)

# ============================================================
# Row 1 — Summary Length + Deterministic Signals
# ============================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("📏 Summary Length Over Refinements")
    fig = px.line(
        x=iter_steps,
        y=[iter_lengths[k] for k in iter_steps],
        markers=True,
        labels={"x": "Iteration", "y": "Word Count"},
    )
    st.plotly_chart(fig, width="stretch")

with col2:
    st.subheader("📐 Deterministic Signals")
    fig = px.bar(
        x=["Coverage", "Glossary Recall", "Hallucination Rate"],
        y=[
            signals["section_coverage_pct"],
            signals["glossary_recall"],
            signals["suspected_hallucination_rate"],
        ],
        range_y=[0, 1],
    )
    st.plotly_chart(fig, width="stretch")

# ============================================================
# Row 2 — Radar + Agreement
# ============================================================

col3, col4 = st.columns(2)

with col3:
    st.subheader("🌐 Rubric Radar Chart")

    categories = ["Coverage", "Faithfulness", "Organization", "Clarity", "Style"]
    values = [
        rubric["coverage"],
        rubric["faithfulness"],
        rubric["organization"],
        rubric["clarity"],
        rubric["style"],
    ]
    values += values[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories + [categories[0]],
        fill='toself',
        line=dict(color="royalblue")
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=False
    )
    st.plotly_chart(fig, width="stretch")

with col4:
    st.subheader("🤝 Agreement Judge Score")
    fig = px.bar(
        x=["Agreement Score"],
        y=[agreement],
        range_y=[0, 5],
    )
    st.plotly_chart(fig, width="stretch")

# ============================================================
# Row 3 — Semantic Drift + Heatmap
# ============================================================

col5, col6 = st.columns(2)

with col5:
    st.subheader("🧠 Semantic Drift Across Iterations")

    names, drifts = compute_semantic_drift(iter_texts)
    fig = px.line(
        x=names,
        y=drifts,
        markers=True,
        labels={"x": "Iteration", "y": "Semantic Drift"},
    )
    st.plotly_chart(fig, width="stretch")

with col6:
    st.subheader("🔥 Judge Disagreement Heatmap")

    rubric_vals = np.array([
        rubric["coverage"],
        rubric["faithfulness"],
        rubric["organization"],
        rubric["clarity"],
        rubric["style"],
    ])
    rubric_norm = rubric_vals / 5.0
    agree_norm = agreement / 5.0

    heat = np.vstack([rubric_norm, [agree_norm] * 5])

    fig = px.imshow(
        heat,
        labels=dict(x="Metric", y="Judge", color="Score"),
        x=["Coverage", "Faithfulness", "Organization", "Clarity", "Style"],
        y=["Rubric", "Agreement"],
        color_continuous_scale="RdBu",
    )
    st.plotly_chart(fig, width="stretch")

# ============================================================
# Row 4 — Full text viewer
# ============================================================

st.markdown("---")
st.subheader("📄 View Iteration Summaries")

for name in iter_steps:
    with st.expander(f"🔍 {name}.txt"):
        st.write(iter_texts[name])
