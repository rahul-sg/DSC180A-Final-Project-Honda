# src/utils/signals.py

"""
Deterministic signals for lecture summary evaluation.

Includes:
- length error
- section coverage via TF-IDF keyword matching
- glossary recall from bold/code/ALLCAPS tokens
- suspected hallucination rate via cosine similarity

These metrics provide stability for LLM-based judges.
"""

from typing import List, Tuple, Dict
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------
# Helpers
# ---------------------------------------------

def extract_sections(slides: List[Dict], title_key="title", content_key="content") -> List[Tuple[str, str]]:
    """
    Normalize slides into a list of (title, content) tuples.
    Safely handles missing keys.
    """
    out = []
    for s in slides:
        title = str(s.get(title_key, "")).strip()
        content = str(s.get(content_key, "")).strip()
        out.append((title, content))
    return out


def top_keywords_per_section(sections: List[Tuple[str, str]], k: int = 5) -> List[List[str]]:
    """
    Get top-k TF-IDF terms per section to approximate the section's semantic core.
    """
    docs = [t + "\n" + c for (t, c) in sections]
    if len(docs) == 0:
        return [[]]

    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform(docs)
    terms = np.array(vec.get_feature_names_out())

    top_terms = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0:
            top_terms.append([])
            continue
        idx = np.argsort(row.toarray()[0])[-k:]
        top_terms.append([t for t in terms[idx] if t])
    return top_terms


def build_glossary(sections: List[Tuple[str, str]]) -> List[str]:
    """
    Extract glossary items from titles, bold text, code text, ALLCAPS.
    Lowercased for fuzzy matching.
    """
    terms = set()

    for (title, content) in sections:

        # Title tokens
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", title):
            terms.add(tok.lower())

        # Bolded term patterns **term**
        for bold in re.findall(r"\*\*(.+?)\*\*", content):
            for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", bold):
                terms.add(tok.lower())

        # Inline code `term`
        for code in re.findall(r"`(.+?)`", content):
            for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", code):
                terms.add(tok.lower())

        # ALLCAPS
        for cap in re.findall(r"\b[A-Z]{3,}\b", content):
            terms.add(cap.lower())

    return list(terms)


def sentence_split(text: str) -> List[str]:
    """
    Lightweight sentence splitter using punctuation boundaries.
    """
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


# ---------------------------------------------
# Main metric computation
# ---------------------------------------------

def compute_signals(slides: List[Dict], summary: str, target_words: int = 300) -> Dict[str, float]:
    """
    Compute deterministic consistency signals for a model summary.

    Returns:
        {
            "length_error": float,
            "section_coverage_pct": float,
            "glossary_recall": float,
            "suspected_hallucination_rate": float
        }
    """

    sections = extract_sections(slides)

    # ----------------------------
    # 1. Length control
    # ----------------------------
    wc = max(1, len(summary.split()))
    length_error = abs(wc - target_words) / float(target_words)

    # ----------------------------
    # 2. Section coverage
    # ----------------------------
    topk = top_keywords_per_section(sections, k=5)
    summary_lc = summary.lower()
    covered = 0

    for kws in topk:
        if not kws:
            continue
        if any(k in summary_lc for k in kws):
            covered += 1

    section_coverage_pct = 0.0 if len(topk) == 0 else covered / len(topk)

    # ----------------------------
    # 3. Glossary recall
    # ----------------------------
    glossary = build_glossary(sections)
    if glossary:
        hits = sum(1 for g in glossary if g in summary_lc)
        glossary_recall = hits / len(glossary)
    else:
        glossary_recall = 0.0

    # ----------------------------
    # 4. Suspected hallucination rate
    # ----------------------------
    slide_sentences = []
    for (_, content) in sections:
        slide_sentences.extend(sentence_split(content))
    slide_sentences = [s for s in slide_sentences if len(s.split()) >= 4]

    summary_sentences = sentence_split(summary)
    suspected = 0

    if slide_sentences and summary_sentences:
        vec = TfidfVectorizer(stop_words="english", max_features=8000)
        vec.fit(slide_sentences + summary_sentences)
        slide_mat = vec.transform(slide_sentences)

        for s in summary_sentences:
            q = vec.transform([s])
            sims = cosine_similarity(q, slide_mat).ravel()
            if (sims >= 0.25).sum() < 1:
                suspected += 1

        suspected_hallucination_rate = suspected / len(summary_sentences)

    else:
        suspected_hallucination_rate = 0.0

    return {
        "length_error": float(length_error),
        "section_coverage_pct": float(section_coverage_pct),
        "glossary_recall": float(glossary_recall),
        "suspected_hallucination_rate": float(suspected_hallucination_rate),
    }
