# src/evaluation/pairwise.py

"""
Pairwise comparison utilities.

Implements:
    - Single A/B pairwise judge call
    - Ensemble-based A/B judging
    - Round-robin tournament for N summaries

Used in compare_models.py and other experiments.
"""

from typing import Dict, Any, List
import random

from src.models.judge import pairwise_judge_single, pairwise_judge_ensemble


# ======================================================================
# Round-robin tournament
# ======================================================================

def round_robin_pairwise(
    slides: List[Dict],
    summaries: Dict[str, str],
    cfg_judge,
    runs: int = 5,
) -> Dict[str, Any]:
    """
    Compare all summaries pairwise using the ensemble judge.

    Args:
        slides: parsed slides
        summaries: dict of {name: summary_text}
        cfg_judge: LLMConfig for judging
        runs: ensemble runs per pair

    Returns:
        {
            "wins": {name: count},
            "win_rate": {name: rate},
            "matches": [
                {
                    "A": nameA,
                    "B": nameB,
                    "winner": winning_name,
                    "wins_detail": {"A": x, "B": y},
                    "reasons_sample": ["...", "..."]
                },
                ...
            ]
        }
    """

    names = list(summaries.keys())
    wins = {n: 0 for n in names}
    matches = []
    total_pairs = 0

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            nameA = names[i]
            nameB = names[j]

            total_pairs += 1

            A_text = summaries[nameA]
            B_text = summaries[nameB]

            # Ensemble A/B judge
            result = pairwise_judge_ensemble(
                slides=slides,
                A=A_text,
                B=B_text,
                cfg=cfg_judge,
                runs=runs,
            )

            # Determine winner
            winner_side = result["winner"]  # "A" or "B"
            winner_name = nameA if winner_side == "A" else nameB

            wins[winner_name] += 1

            matches.append({
                "A": nameA,
                "B": nameB,
                "winner": winner_name,
                "wins_detail": result["wins"],
                "reasons_sample": result["reasons_sample"],
            })

    # Normalize win rates
    win_rate = {
        name: wins[name] / max(1, total_pairs)
        for name in names
    }

    return {
        "wins": wins,
        "win_rate": win_rate,
        "matches": matches
    }
