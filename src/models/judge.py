# src/models/judge.py

"""
LLM Judges:
-----------
1) Rubric Judge (reference-free)
2) Agreement Judge (reference-aware)
3) Pairwise Judge (A vs B)
4) Ensemble wrappers for stability

Notes:
- Uses gpt-5-chat-latest by default
- Always calls LLM in JSON mode
- Compatible with new OpenAI parameters
"""

from typing import Dict, Any, List
import json
import random

from src.models.llm_client import call_llm, parse_json_or_throw, LLMConfig
from src.utils.chunking import slides_to_text


# ================================================================
# PROMPTS
# ================================================================

RUBRIC_PROMPT = """You are a strict teaching assistant. Evaluate a student-facing summary of a lecture.

IMPORTANT:
- Prioritize FAITHFULNESS over style.
- Cite 1–2 phrases from the slides as evidence.

Score the summary on the following (1–5):
1) coverage
2) faithfulness
3) organization
4) clarity
5) style

Also provide:
- overall_1to10 (1–10 integer)
- two_strengths: [ "...", "..." ]
- two_issues: ["...", "..."]
- faithfulness_evidence: ["...", "..."]

Return ONLY valid JSON:
{
  "coverage": int,
  "faithfulness": int,
  "organization": int,
  "clarity": int,
  "style": int,
  "overall_1to10": int,
  "two_strengths": ["...", "..."],
  "two_issues": ["...", "..."],
  "faithfulness_evidence": ["...", "..."]
}
"""

AGREEMENT_PROMPT = """You are grading agreement between a reference summary and a model summary.

Focus ONLY on:
- essential fact overlap
- missing key points
- inaccuracies added

Return ONLY valid JSON:
{
  "agreement_1to5": int,
  "missing_key_points": ["...", "..."],
  "added_inaccuracies": ["...", "..."]
}
"""

PAIRWISE_PROMPT = """You are comparing two summaries written for the SAME lecture.

Pick which is better overall for students.

Return ONLY JSON:
{
  "winner": "A" or "B",
  "reason": "..."
}
"""


# ================================================================
# Reference-Free Rubric Judge
# ================================================================

def judge_rubric(slides: List[Dict], summary: str, cfg: LLMConfig) -> Dict[str, Any]:
    """
    Reference-free judge comparing summary to slide content.
    """
    slide_text = slides_to_text(slides)
    user_msg = f"[Slides]\n{slide_text}\n\n[Summary]\n{summary}\n\nReturn ONLY JSON."

    raw = call_llm(
        system_prompt=RUBRIC_PROMPT,
        user_prompt=user_msg,
        cfg=cfg,
        json_mode=True
    )
    return parse_json_or_throw(raw)


# ================================================================
# Agreement Judge (reference-based)
# ================================================================

def judge_agreement(reference: str, summary: str, cfg: LLMConfig) -> Dict[str, Any]:
    """
    Compare summary to a human-written reference.
    """
    user_msg = f"[Reference]\n{reference}\n\n[Model Summary]\n{summary}\nReturn ONLY JSON."

    raw = call_llm(
        system_prompt=AGREEMENT_PROMPT,
        user_prompt=user_msg,
        cfg=cfg,
        json_mode=True
    )
    return parse_json_or_throw(raw)


# ================================================================
# Pairwise Judge
# ================================================================

def judge_pairwise(slides: List[Dict], A: str, B: str, cfg: LLMConfig) -> Dict[str, Any]:
    """
    Choose better summary (A or B) for the SAME lecture.
    """
    slide_text = slides_to_text(slides)
    user_msg = (
        f"[Slides]\n{slide_text}\n\n"
        f"[Summary A]\n{A}\n\n"
        f"[Summary B]\n{B}\n\n"
        f"Return ONLY JSON."
    )

    raw = call_llm(
        system_prompt=PAIRWISE_PROMPT,
        user_prompt=user_msg,
        cfg=cfg,
        json_mode=True
    )

    data = parse_json_or_throw(raw)

    # If hallucinated, force correction
    winner = data.get("winner", "").strip()
    if winner not in ("A", "B"):
        winner = random.choice(["A", "B"])
        data["winner"] = winner

    return data


# ================================================================
# Ensemble Wrappers
# ================================================================

def judge_rubric_ensemble(slides, summary, cfg: LLMConfig, runs: int = 3) -> Dict[str, Any]:
    """
    Average scores across several seeds.
    """
    outs = []
    for r in range(runs):
        cfg_r = LLMConfig(
            model=cfg.model,
            max_completion_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            seed=(cfg.seed or 0) + r
        )
        outs.append(judge_rubric(slides, summary, cfg_r))

    keys = ["coverage", "faithfulness", "organization", "clarity", "style", "overall_1to10"]
    avg = {k: int(round(sum(o[k] for o in outs) / len(outs))) for k in keys}

    # Carry narrative fields from first run
    avg["two_strengths"] = outs[0]["two_strengths"]
    avg["two_issues"] = outs[0]["two_issues"]
    avg["faithfulness_evidence"] = outs[0]["faithfulness_evidence"]

    return avg


def judge_agreement_ensemble(reference, summary, cfg: LLMConfig, runs: int = 3) -> Dict[str, Any]:
    outs = []
    for r in range(runs):
        cfg_r = LLMConfig(
            model=cfg.model,
            max_completion_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            seed=(cfg.seed or 0) + r
        )
        outs.append(judge_agreement(reference, summary, cfg_r))

    score = int(round(sum(o["agreement_1to5"] for o in outs) / len(outs)))

    return {
        "agreement_1to5": score,
        "missing_key_points": outs[0]["missing_key_points"],
        "added_inaccuracies": outs[0]["added_inaccuracies"]
    }


def judge_pairwise_ensemble(slides, A, B, cfg: LLMConfig, runs: int = 5) -> Dict[str, Any]:
    wins = {"A": 0, "B": 0}
    reasons = []

    for r in range(runs):
        cfg_r = LLMConfig(
            model=cfg.model,
            max_completion_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            seed=(cfg.seed or 0) + r
        )

        result = judge_pairwise(slides, A, B, cfg_r)
        winner = result["winner"]
        wins[winner] += 1
        reasons.append(result.get("reason", ""))

    final = "A" if wins["A"] >= wins["B"] else "B"

    return {
        "winner": final,
        "wins": wins,
        "reasons_sample": reasons[:2]
    }
