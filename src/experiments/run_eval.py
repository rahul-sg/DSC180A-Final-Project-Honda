# src/experiments/run_eval.py

"""
Run a full evaluation for a single lecture.

Usage Examples:
    python -m src.experiments.run_eval lecture1
    python -m src.experiments.run_eval lecture2 yes   # force regenerate S0
    python -m src.experiments.run_eval lecture3 no    # reuse existing S0

Defaults:
    lecture_id = "lecture1"
    regenerate_s0 = "no"
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from src.evaluation.pipeline import evaluate_summary
from src.models.llm_client import LLMConfig
from src.models.summarizer import generate_initial_summary

# ================================================================
# Load .env from project root
# ================================================================
ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".env"
load_dotenv(ENV_PATH)


# ================================================================
# MAIN
# ================================================================
def main():

    # ------------------------------------------------------------
    # Parse CLI arguments
    # ------------------------------------------------------------
    if len(sys.argv) > 1:
        lecture_id = sys.argv[1].strip()
    else:
        lecture_id = "lecture1"

    if len(sys.argv) > 2:
        force_regen = sys.argv[2].strip().lower()
    else:
        force_regen = "no"   # default

    print(f"\n📘 Lecture selected: {lecture_id}")
    print(f"🔄 Force regenerate S0: {force_regen}")

    # ------------------------------------------------------------
    # Define paths
    # ------------------------------------------------------------
    SLIDES_PATH = f"data/slides/{lecture_id}.pdf"
    HUMAN_REF_PATH = f"data/references/{lecture_id}_reference.txt"
    INITIAL_SUMMARY_PATH = Path(f"data/summaries/model_s0/{lecture_id}.txt")
    OUT_DIR = Path(f"data/summaries/refined_iterations/{lecture_id}")

    # Validate input files
    if not Path(SLIDES_PATH).exists():
        raise FileNotFoundError(f"❌ Lecture slides not found: {SLIDES_PATH}")

    if not Path(HUMAN_REF_PATH).exists():
        raise FileNotFoundError(f"❌ Reference summary not found: {HUMAN_REF_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Clean evaluation folder
    # ------------------------------------------------------------
    print(f"🧹 Clearing previous evaluation runs for {lecture_id}...")
    for file in OUT_DIR.glob("*"):
        try:
            file.unlink()
        except Exception:
            print(f"⚠️ Could not delete: {file}")

    # ============================================================
    # Load human reference
    # ============================================================
    with open(HUMAN_REF_PATH, "r") as f:
        human_reference = f.read().strip()

    # ============================================================
    # Load / Generate Initial Summary S0
    # ============================================================
    regenerate = (force_regen == "yes")
    initial_summary = ""

    if INITIAL_SUMMARY_PATH.exists() and not regenerate:
        # Try loading existing S0
        with open(INITIAL_SUMMARY_PATH, "r") as f:
            initial_summary = f.read().strip()

        if len(initial_summary.split()) < 50:
            print("[S0] Existing S0 too short — regenerating...")
            initial_summary = ""
        else:
            print(f"[S0] Reusing existing S0 at {INITIAL_SUMMARY_PATH}")

    else:
        if regenerate:
            print("[S0] Force regenerate is ON — creating new S0.")
        else:
            print("[S0] No S0 found — generating new S0.")

        initial_summary = ""

    # Generate new S0 if needed
    if not initial_summary:
        print("[S0] Generating new S0 with gpt-5-chat-latest...")

        cfg_summarizer = LLMConfig(
            model="gpt-5-chat-latest",
            max_completion_tokens=700,
        )

        initial_summary = generate_initial_summary(SLIDES_PATH, cfg_summarizer)

        INITIAL_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INITIAL_SUMMARY_PATH, "w") as f:
            f.write(initial_summary)

        print(f"[S0] Saved new S0 → {INITIAL_SUMMARY_PATH}")

    # ============================================================
    # Judge & Refiner Models
    # ============================================================
    cfg_judge = LLMConfig(
        model="gpt-5-chat-latest",
        max_completion_tokens=512,
    )

    cfg_refine = LLMConfig(
        model="gpt-5-chat-latest",
        max_completion_tokens=800,
    )

    # ============================================================
    # Run Evaluation Pipeline
    # ============================================================
    result = evaluate_summary(
        slide_path=SLIDES_PATH,
        initial_summary=initial_summary,
        human_reference=human_reference,
        cfg_judge=cfg_judge,
        cfg_refine=cfg_refine,
        out_dir=str(OUT_DIR),
        target_words=300,
        refine_iters=3,
    )

    # ============================================================
    # Print Results
    # ============================================================
    print("\n===== FINAL EVALUATION RESULT =====")
    print("Score (0–1):", result["final_score_0to1"])
    print("\nRefined Summary:\n", result["refined_summary"])
    print("\nSignals:", result["signals"])
    print("\nRubric:", result["rubric"])
    print("\nAgreement:", result["agreement"])
    print("\nOutputs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
