# src/visualization/dashboard.py

import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# ============================================================
# Load iteration summaries
# ============================================================

def load_iteration_summaries(iter_dir):
    texts = {}
    for fname in Path(iter_dir).glob("*.txt"):
        name = fname.stem
        txt = fname.read_text().strip()
        texts[name] = txt
    return texts


# ============================================================
# Compute embeddings (OpenAI text-embedding-3-large)
# ============================================================

def get_embedding(text):
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
    drifts = [1 - sim for sim in sims]
    return names, drifts


# ============================================================
# Main dashboard plotting function
# ============================================================

def plot_dashboard(iter_dir, result_json_path):
    iter_dir = Path(iter_dir)
    result_json_path = Path(result_json_path)

    # Load summaries
    iter_texts = load_iteration_summaries(iter_dir)
    iter_lengths = {k: len(v.split()) for k, v in iter_texts.items()}

    # Load result.json
    result = json.loads(result_json_path.read_text())
    signals = result["signals"]
    rubric = result["rubric"]
    agreement_score = result["agreement"]["agreement_1to5"]

    # Sort iterations
    iter_steps = sorted(
        iter_lengths.keys(),
        key=lambda x: int(x.split("_")[1]) if x.startswith("iter_") else 999
    )
    word_counts = [iter_lengths[k] for k in iter_steps]

    # ============================================================
    # Create 3x2 dashboard
    # ============================================================
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    fig.suptitle("Evaluation Dashboard", fontsize=20, weight="bold")

    # ============================================================
    # (1) Summary Length Across Iterations
    # ============================================================
    ax = axes[0, 0]
    ax.plot(iter_steps, word_counts, marker='o', linewidth=2)
    ax.set_title("Summary Length Across Refinement Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Word Count")
    ax.grid(True)

    # ============================================================
    # (2) Deterministic Signals
    # ============================================================
    ax = axes[0, 1]
    sig_names = ["Coverage", "Glossary Recall", "Hallucination Rate"]
    sig_vals = [
        signals["section_coverage_pct"],
        signals["glossary_recall"],
        signals["suspected_hallucination_rate"],
    ]

    ax.bar(sig_names, sig_vals, color=["#1f77b4", "#2ca02c", "#d62728"])
    ax.set_ylim(0, 1)
    ax.set_title("Deterministic Signals")
    ax.grid(axis="y")

    # ============================================================
    # (3) Rubric Radar Chart  (fixed)
    # ============================================================
    ax = fig.add_subplot(3, 2, 3, polar=True)

    categories = ["Coverage", "Faithfulness", "Organization", "Clarity", "Style"]
    values = [
        rubric["coverage"],
        rubric["faithfulness"],
        rubric["organization"],
        rubric["clarity"],
        rubric["style"],
    ]
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(values))

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 5)
    ax.set_title("Rubric Radar Chart", pad=20)

    # ============================================================
    # (4) Agreement Judge
    # ============================================================
    ax = axes[1, 1]
    ax.bar(["Final Summary"], [agreement_score], color="#9467bd")
    ax.set_ylim(0, 5)
    ax.set_ylabel("Agreement (1–5)")
    ax.set_title("Agreement Judge")
    ax.grid(axis="y")

    # ============================================================
    # (5) Semantic Drift
    # ============================================================
    ax = axes[2, 0]
    names, drifts = compute_semantic_drift(iter_texts)

    ax.plot(names, drifts, marker="o", color="#ff7f0e", linewidth=2)
    ax.set_title("Semantic Drift Across Iterations\n(1 - cosine similarity to S0)")
    ax.set_ylabel("Semantic Drift")
    ax.set_xlabel("Iteration")
    ax.grid(True)

    # ============================================================
    # (6) Judge Disagreement Heatmap
    # ============================================================
    ax = axes[2, 1]

    rubric_vals = np.array([
        rubric["coverage"],
        rubric["faithfulness"],
        rubric["organization"],
        rubric["clarity"],
        rubric["style"],
    ])

    rubric_norm = rubric_vals / 5.0
    agree_norm = agreement_score / 5.0

    heat_data = np.vstack([rubric_norm, [agree_norm] * 5])

    ax.imshow(heat_data, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(5))
    ax.set_xticklabels(categories)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Rubric", "Agreement"])
    ax.set_title("Judge Disagreement Heatmap (Normalized)")

    for (i, j), val in np.ndenumerate(heat_data):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")

    # ============================================================
    # Fix spacing
    # ============================================================
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(top=0.92, wspace=0.35, hspace=0.55)

    plt.show()


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    if len(sys.argv) > 1:
        lecture_id = sys.argv[1]
    else:
        lecture_id = "lecture1"

    iter_dir = f"data/summaries/refined_iterations/{lecture_id}"
    result_json = f"{iter_dir}/result.json"

    print(f"\n📊 Loading dashboard for {lecture_id}")
    print(f"   Iter dir:   {iter_dir}")
    print(f"   Result JSON: {result_json}\n")

    plot_dashboard(iter_dir, result_json)


if __name__ == "__main__":
    main()
