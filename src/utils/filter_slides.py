# src/utils/filter_slides.py

"""
Filters out administrative/syllabus slides from lecture PDFs.

Goal B:
    - Keep ONLY real teaching content
    - Remove syllabus, instructor info, grading policies, course logistics

This dramatically improves summarization quality.
"""

from typing import List, Dict
import re


def is_syllabus_slide(slide: Dict) -> bool:
    """
    Decide if a slide is administrative rather than teaching content.
    """
    text = (slide.get("title", "") + " " + slide.get("content", "")).lower()

    # Common syllabus indicators
    syllabus_keywords = [
        "syllabus",
        "instructor",
        "office hours",
        "course info",
        "course information",
        "grading",
        "required materials",
        "policies",
        "academic integrity",
        "late work",
        "makeup",
        "assessment",
        "quiz",
        "final exam",
        "midterm",
        "ta:", "teaching assistant",
        "section",
        "ucsd",
        "university policy",
        "zoom",
        "canvas",
        "connect",
        "hbs case",
        "week 1–10",
        "schedule",
        "course overview",
        "learning objectives",
        "achieving the objective",
    ]

    # If any keyword appears → it's a syllabus slide
    return any(keyword in text for keyword in syllabus_keywords)


def filter_content_slides(slides: List[Dict]) -> List[Dict]:
    """
    Remove all administrative slides.
    Keep *only* those with teaching content.
    """
    clean = [s for s in slides if not is_syllabus_slide(s)]

    # If filtering accidentally removes everything (rare),
    # fallback to original slides.
    return clean if len(clean) > 0 else slides
