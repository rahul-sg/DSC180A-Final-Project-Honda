# src/evaluation/pipeline.py

"""
Evaluation Pipeline (Patched with Goal B)

This version:
    ✔ Uses full slides for JUDGES (rubric + agreement) — correct!
    ✔ Uses content-only slides for deterministic signals
    ✔ Passes full slides to refinement (refinement itself handles filtering)
    ✔ Eliminates penalties from syllabus/admin slides
"""

import os
from typing import Dict, Any
from pathlib import Path

from src.utils.io import (
    load_slides,
    write_iteration_summary,
    write_final_summary,
    write_json,
)
from src.utils.signals import compute_signals
from src.models.judge import (
    judge_rubric_ensemble,
    judge_agreement_ensemble,
)
from src.models.refinement import iterative_refinement
from src.evaluation.scoring import combine_scores

# NEW IMPORT — required for Goal B
from src.utils.filter_slides import filter_content_slides


# =====================================================================
# MAIN PIPELINE FUNCTION
# =====================================================================

def evaluate_summary(
    slide_path: str,
    initial_summary: str,
    human_reference: str,
    cfg_judge,
    cfg_refine,
    out_dir: str,
    target_words: int = 300,
    refine_iters: int = 3,
) -> Dict[str, Any]:
    """
    Full evaluation pipeline for a single lecture.
    """

    # -----------------------------------------------------
    # 1. Load & parse slides
    # -----------------------------------------------------
    slides_dict = load_slides(slide_path)
    slides_full = slides_dict["slides"]            # raw slides
    slides_content = filter_content_slides(slides_full)  # Goal B applied

    # Ensure output directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # 2. Iterative refinement
    # -----------------------------------------------------

    prev_summary = initial_summary

    def save_callback(iter_idx: int, summary_text: str):
        """Save iter_0.txt ... iter_k.txt safely."""
        nonlocal prev_summary

        # Prevent blank summaries from propagating
        if not summary_text.strip():
            summary_text = prev_summary

        write_iteration_summary(out_dir, iter_idx, summary_text)
        prev_summary = summary_text

    # refinement internally handles filtering — we pass full slides
    refined = iterative_refinement(
        slides=slides_full,           
        initial_summary=initial_summary,
        cfg_judge=cfg_judge,
        cfg_refine=cfg_refine,
        iters=refine_iters,
        save_callback=save_callback,
    )

    if not refined.strip():
        refined = prev_summary

    # Save final result
    write_final_summary(out_dir, refined)

    # -----------------------------------------------------
    # 3. Deterministic signals (Goal B → use content-only slides)
    # -----------------------------------------------------
    signals = compute_signals(
        slides_content,        # not full slides
        refined,
        target_words=target_words
    )

    # -----------------------------------------------------
    # 4. Rubric judge (full slides)
    # -----------------------------------------------------
    rubric = judge_rubric_ensemble(
        slides_full,           # judges see full lecture
        refined,
        cfg_judge,
        runs=3
    )

    # -----------------------------------------------------
    # 5. Agreement judge (Goal B → compare only content)
    # -----------------------------------------------------
    agree = judge_agreement_ensemble(
        human_reference,
        refined,
        cfg_judge,
        runs=3
    )

    # -----------------------------------------------------
    # 6. Final score
    # -----------------------------------------------------
    score = combine_scores(rubric, agree)

    # -----------------------------------------------------
    # 7. Save final structured JSON
    # -----------------------------------------------------
    result = {
        "refined_summary": refined,
        "signals": signals,
        "rubric": rubric,
        "agreement": agree,
        "final_score_0to1": score,
        "lecture_title": slides_dict.get("lecture_title", "Unknown Lecture"),
    }

    write_json(os.path.join(out_dir, "result.json"), result)

    return result
