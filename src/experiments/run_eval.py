# src/experiments/run_eval.py

"""
Run a full evaluation for a single lecture.

Usage:
    python -m src.experiments.run_eval lecture2
    python -m src.experiments.run_eval lecture3
If no argument is provided, defaults to lecture1.
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
    # Allow CLI selection of lecture
    # ------------------------------------------------------------
    if len(sys.argv) > 1:
        lecture_id = sys.argv[1].strip()
        print(f"\n📘 Using lecture: {lecture_id}")
    else:
        lecture_id = "lecture1"
        print("\n📘 No lecture specified — defaulting to lecture1")

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
    # Clean old evaluation outputs
    # ------------------------------------------------------------
    print(f"🧹 Cleaning old evaluation files in {OUT_DIR} ...")

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
    # Load or generate initial summary S0
    # ============================================================
    initial_summary = ""

    if INITIAL_SUMMARY_PATH.exists():
        with open(INITIAL_SUMMARY_PATH, "r") as f:
            initial_summary = f.read().strip()

        if len(initial_summary.split()) < 50:
            print(f"[S0] Existing S0 summary too short — regenerating.")
            initial_summary = ""

        else:
            print(f"[S0] Loaded existing S0 from {INITIAL_SUMMARY_PATH}")

    if not initial_summary:
        print(f"[S0] Generating S0 using gpt-5-chat-latest...")

        cfg_summarizer = LLMConfig(
            model="gpt-5-chat-latest",
            max_completion_tokens=700,
        )

        initial_summary = generate_initial_summary(SLIDES_PATH, cfg_summarizer)

        INITIAL_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INITIAL_SUMMARY_PATH, "w") as f:
            f.write(initial_summary)

        print(f"[S0] Saved new S0 to {INITIAL_SUMMARY_PATH}")

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
    # Run Evaluation
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
    # Print + Save Results
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
