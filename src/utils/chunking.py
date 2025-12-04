# src/utils/chunking.py

"""
Chunking utilities for lecture slides.

Includes:
- Token estimation (4 chars ≈ 1 token heuristic)
- Token-based chunking for long lectures
- Slide → text rendering for LLM judges
"""

from typing import List, Dict


def estimate_tokens(text: str) -> int:
    """
    Estimate tokens using a simple heuristic:
    ~4 characters per token.

    Safer alternative to depending on external tokenizers.
    """
    return max(1, int(len(text) / 4))


def chunk_slides_by_tokens(
    slides: List[Dict],
    max_tokens: int = 1500,
    text_key: str = "content"
) -> List[List[Dict]]:
    """
    Chunk slides so each chunk stays under ~max_tokens (approx).
    Ensures LLM context safety for long lectures.
    """
    chunks: List[List[Dict]] = []
    current, cur_tok = [], 0

    for s in slides:
        t = estimate_tokens(str(s.get(text_key, "")))

        # Start a new chunk if adding this slide would exceed limit
        if current and cur_tok + t > max_tokens:
            chunks.append(current)
            current, cur_tok = [], 0

        current.append(s)
        cur_tok += t

    if current:
        chunks.append(current)

    return chunks


def slides_to_text(
    slides: List[Dict],
    max_chunks: int = 3,
    max_tokens: int = 1500
) -> str:
    """
    Convert a slide list into a compact text block suitable for LLM context.
    The output includes chunk & slide indices for better structure.

    Parameters:
        slides — list of slide dictionaries
        max_chunks — limits # of chunks passed to judge
        max_tokens — max tokens per chunk

    Returns:
        A formatted multi-chunk text representation.
    """
    chunks = chunk_slides_by_tokens(slides, max_tokens=max_tokens)
    chunks = chunks[:max_chunks]

    out = []
    for ci, ch in enumerate(chunks, 1):
        for i, s in enumerate(ch, 1):
            title = s.get("title", "")
            content = s.get("content", "")
            out.append(f"[Chunk {ci} • Slide {i}] {title}\n{content}")

    return "\n\n".join(out)
