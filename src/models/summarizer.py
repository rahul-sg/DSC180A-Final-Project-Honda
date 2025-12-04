# src/models/summarizer.py

"""
Automatic S0 Summarizer (Content-Only Version — Goal B)

This version:
    ✔ Removes syllabus/logistics slides
    ✔ Summarizes ONLY real teaching content
    ✔ Produces clean 250–350 word summaries
    ✔ Retries if model output is too short
"""

from typing import Dict, List

from src.utils.io import load_slides
from src.utils.chunking import slides_to_text
from src.utils.filter_slides import filter_content_slides
from src.models.llm_client import call_llm, LLMConfig


SUMMARIZER_PROMPT = """
You are a lecture summarization assistant trained to produce clear,
accurate, and academically appropriate summaries.

TASK:
Summarize the following lecture content into a coherent 250–350 word summary
that:
- captures all major topics,
- preserves factual accuracy,
- avoids hallucinations,
- organizes information logically,
- is easy for a student to understand.

IMPORTANT:
- Only use information from the slides provided.
- Ignore syllabus, logistics, or administrative content.

Return ONLY the summary text.
"""


def _call_summarizer_once(slides_text: str, cfg: LLMConfig) -> str:
    """
    Single LLM call for summarization.
    """
    user_prompt = f"{SUMMARIZER_PROMPT}\n\n[Slides]\n{slides_text}"

    summary = call_llm(
        system_prompt="You summarize lecture slides accurately for students.",
        user_prompt=user_prompt,
        cfg=cfg,
        json_mode=False,
    )

    return (summary or "").strip()


def generate_initial_summary(
    pdf_path: str,
    cfg_summarizer: LLMConfig,
    retry_limit: int = 2,
) -> str:
    """
    Produce the initial model-generated summary (S0) for a lecture.

    Steps:
        1. Load slides
        2. Remove syllabus slides (Goal B)
        3. Convert to text
        4. Call LLM (retry if needed)

    Returns:
        ~250–350 word summary string
    """

    # 1. Parse entire slide deck (PDF → structured slides)
    slides_dict = load_slides(pdf_path)
    all_slides: List[Dict] = slides_dict["slides"]

    # 2. Remove administrative/syllabus slides (Goal B)
    content_slides = filter_content_slides(all_slides)

    # 3. Convert to compact text
    slides_text = slides_to_text(content_slides)

    # 4. Call the LLM with simple retry logic
    for attempt in range(retry_limit):
        summary = _call_summarizer_once(slides_text, cfg_summarizer)
        # basic sanity check: require at least ~120 words
        if summary and len(summary.split()) >= 120:
            return summary

    # If summarizer fails
    raise RuntimeError(
        "Summarizer failed to produce a sufficiently long summary "
        f"after {retry_limit} attempts using model {cfg_summarizer.model}."
    )
