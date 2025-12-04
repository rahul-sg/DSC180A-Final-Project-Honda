# src/utils/pdf_parser.py

"""
GENERAL-PURPOSE PDF → Slide Parser

This parser works for ANY slide deck exported to PDF:
- university lectures
- corporate presentations
- conference slides

It extracts:
{
    "lecture_title": "...",
    "slides": [
        {
            "title": "...",
            "content": "cleaned text",
            "raw_text": "...",
            "page_number": X
        }
    ]
}

Design goals:
- Never return empty slides
- Extract meaningful titles
- Handle diagram-heavy pages gracefully
- Avoid assumptions about institution or format
"""

import fitz  # PyMuPDF
from typing import Dict, Any, List
import os


# ------------------------------------------------------------
# Helper: Clean and normalize lines
# ------------------------------------------------------------
def _clean_lines(lines: List[str]) -> List[str]:
    """
    Basic cleaning that works for ANY slide deck:
    - Strip whitespace
    - Remove empty lines
    - Preserve all actual content
    """
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        cleaned.append(line)
    return cleaned


# ------------------------------------------------------------
# Title Extraction — General Purpose Heuristic
# ------------------------------------------------------------
def _extract_title(clean_lines: List[str], page_idx: int) -> str:
    """
    General-purpose title extraction:
      1. Use the first line if it's short (< 12 words)
      2. Otherwise pick the shortest line as title (common for PPT exports)
      3. Fallback: Slide {n}
    """
    if not clean_lines:
        return f"Slide {page_idx + 1}"

    # Case 1: first line is reasonably short → likely the slide title
    first = clean_lines[0]
    if len(first.split()) <= 12:
        return first

    # Case 2: pick the shortest non-empty line (common in PDF exports)
    shortest = min(clean_lines, key=lambda l: len(l))
    if len(shortest.split()) <= 12:
        return shortest

    # Fallback
    return f"Slide {page_idx + 1}"


# ------------------------------------------------------------
# Diagram Detection
# ------------------------------------------------------------
def _detect_diagram(text: str) -> bool:
    """
    Heuristic:
    - Slides with < 15 words likely contain mainly diagrams.
    """
    words = text.split()
    return len(words) < 15


# ------------------------------------------------------------
# Convert PDF → Slide Objects
# ------------------------------------------------------------
def extract_slides_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Convert each PDF page into a structured slide dict.
    Works for ANY type of slide deck.
    """

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    slides = []

    for page_idx, page in enumerate(doc):
        raw_text = page.get_text("text")

        # Split into lines & clean
        raw_lines = raw_text.split("\n")
        clean_lines = _clean_lines(raw_lines)

        # If the page is extremely empty → substitute placeholder
        if not clean_lines:
            clean_lines = ["(No readable text extracted from this slide.)"]

        # Title
        title = _extract_title(clean_lines, page_idx)

        # Content = everything except title
        body_lines = clean_lines[1:] if len(clean_lines) > 1 else []
        content = "\n".join(body_lines).strip()

        # Guarantee non-empty content
        if not content:
            content = "[This slide contains mainly visual content.]"

        # Diagram detection
        if _detect_diagram(content):
            content += "\n[Diagram detected — summarize conceptual meaning rather than visuals.]"

        slides.append({
            "title": title,
            "content": content,
            "raw_text": raw_text,
            "page_number": page_idx + 1,
        })

    lecture_title = os.path.basename(pdf_path).replace(".pdf", "")

    return {
        "lecture_title": lecture_title,
        "slides": slides
    }
