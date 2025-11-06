eval_baseline.py
# LLM-as-judge baseline for lecture summarization — robust, reproducible, and easy to extend.

from __future__ import annotations
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import json, re, random
import numpy as np

# Lightweight NLP (no web). Install scikit-learn for TF-IDF and cosine sim.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 0) Chunking utilities
# =========================================================

def chunk_slides(slides: List[Dict], chunk_size: int) -> List[List[Dict]]:
    """
    Split slides (list of dicts) into fixed-size chunks by COUNT.
    """
    return [slides[i:i + chunk_size] for i in range(0, len(slides), chunk_size)]


def estimate_tokens(text: str) -> int:
    """
    Very rough token estimate using ~= 4 chars/token heuristic.
    (Swap with a real tokenizer like tiktoken if you have one.)
    """
    return max(1, int(len(text) / 4))


def chunk_slides_by_tokens(
    slides: List[Dict],
    max_tokens: int = 1500,
    text_key: str = "content"
) -> List[List[Dict]]:
    """
    Chunk slides so each chunk stays under ~max_tokens (approx).
    Safer for LLM context windows than chunking by count.
    """
    chunks: List[List[Dict]] = []
    current, cur_tok = [], 0
    for s in slides:
        t = estimate_tokens(str(s.get(text_key, "")))
        if current and cur_tok + t > max_tokens:
            chunks.append(current)
            current, cur_tok = [], 0
        current.append(s)
        cur_tok += t
    if current:
        chunks.append(current)
    return chunks


def _slides_to_str(slides: List[Dict], max_chunks: int = 3, max_tokens: int = 1500) -> str:
    """
    Render slides into a compact text block for the judge, with token-safe chunking.
    Caps the number of chunks to limit total context size.
    """
    chunks = chunk_slides_by_tokens(slides, max_tokens=max_tokens)
    chunks = chunks[:max_chunks]
    out = []
    for ci, ch in enumerate(chunks, 1):
        for i, s in enumerate(ch, 1):
            out.append(f"[Chunk {ci} • Slide {i}] {s.get('title','')}\n{s.get('content','')}")
    return "\n\n".join(out)


# =========================================================
# 1) Simple deterministic signals (no LLM required)
# =========================================================

def _extract_sections(slides: List[Dict], title_key="title", content_key="content") -> List[Tuple[str, str]]:
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


def _top_keywords_per_section(sections: List[Tuple[str, str]], k: int = 5) -> List[List[str]]:
    """
    Get top-k TF-IDF terms per section to proxy the section’s “essence.”
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


def _build_glossary(sections: List[Tuple[str, str]]) -> List[str]:
    """
    Crude glossary = title tokens + **bold** + `code` + ALLCAPS tokens.
    Lowercased for matching.
    """
    terms = set()
    for (title, content) in sections:
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", title):
            terms.add(tok.lower())
        for bold in re.findall(r"\*\*(.+?)\*\*", content):
            for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", bold):
                terms.add(tok.lower())
        for code in re.findall(r"`(.+?)`", content):
            for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", code):
                terms.add(tok.lower())
        for cap in re.findall(r"\b[A-Z]{3,}\b", content):
            terms.add(cap.lower())
    return list(terms)


def _sentence_split(text: str) -> List[str]:
    """
    Simple sentence splitter using punctuation boundaries.
    """
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def simple_signals(slides: List[Dict], summary: str, target_words: int = 300) -> Dict[str, float]:
    """
    Compute cheap, reproducible metrics to anchor your LLM judge:
      - length_error: deviation from target word count (0 is perfect)
      - section_coverage_pct: % of sections whose top keywords appear in summary
      - glossary_recall: fraction of glossary terms present
      - suspected_hallucination_rate: % of summary sentences that can't be matched to any slide sentence by TF-IDF cosine
    """
    sections = _extract_sections(slides)

    # 1) length control
    wc = max(1, len(summary.split()))
    length_error = abs(wc - target_words) / float(target_words)

    # 2) section coverage via top-k keyword hits
    topk = _top_keywords_per_section(sections, k=5)
    covered = 0
    summary_lc = summary.lower()
    for kws in topk:
        if not kws:
            continue
        hit = any(k in summary_lc for k in kws)
        covered += 1 if hit else 0
    section_coverage_pct = 0.0 if len(topk) == 0 else covered / len(topk)

    # 3) glossary recall
    glossary = _build_glossary(sections)
    if glossary:
        hits = sum(1 for g in glossary if g in summary_lc)
        glossary_recall = hits / len(glossary)
    else:
        glossary_recall = 0.0

    # 4) hallucination proxy via retrieval similarity
    slide_sentences = []
    for (_, content) in sections:
        slide_sentences.extend(_sentence_split(content))
    slide_sentences = [s for s in slide_sentences if len(s.split()) >= 4]
    summary_sentences = _sentence_split(summary)

    suspected = 0
    if slide_sentences and summary_sentences:
        vec = TfidfVectorizer(stop_words="english", max_features=8000)
        vec.fit(slide_sentences + summary_sentences)
        slide_mat = vec.transform(slide_sentences)
        for s in summary_sentences:
            q = vec.transform([s])
            sims = cosine_similarity(q, slide_mat).ravel()
            # conservative: if no reasonably similar slide sentence exists, flag
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


# =========================================================
# 2) Centralized LLM call (wire your provider here)
# =========================================================

@dataclass
class LLMConfig:
    provider: str = "openai"         # e.g., "openai", "anthropic", "local"
    model: str = "gpt-4o-mini"       # change to your model
    temperature: float = 0.2
    max_tokens: int = 512
    seed: int | None = None          # use if your provider supports it


def call_llm(system: str, user: str, cfg: LLMConfig) -> str:
    """
    Single chokepoint for model calls. Replace internals with your SDK.
    Return raw text (ideally JSON due to our prompts).
    """
    # Example (pseudo) OpenAI usage:
    # from openai import OpenAI
    # client = OpenAI()
    # resp = client.chat.completions.create(
    #   model=cfg.model,
    #   temperature=cfg.temperature,
    #   max_tokens=cfg.max_tokens,
    #   seed=cfg.seed,
    #   messages=[{"role":"system","content":system},{"role":"user","content":user}],
    # )
    # return resp.choices[0].message.content
    return '{"note":"Replace call_llm() internals with your provider!"}'


# =========================================================
# 3) Judge prompts & helpers
# =========================================================

_RUBRIC_PROMPT = """You are a strict teaching assistant. Evaluate a student-facing summary of a lecture.

IMPORTANT: Prioritize FAITHFULNESS over style. Do NOT reward eloquence if facts are unsupported by slides.
When judging faithfulness, cite 1–2 concrete phrases from the slides that support or contradict the summary.

SCORE the summary on the following 5 dimensions from 1 (poor) to 5 (excellent):
1) Coverage (major topics included; breadth of core points covered)
2) Faithfulness (all claims supported by the slides; no contradictions or hallucinations)
3) Organization (logical flow, signposting, paragraph cohesion)
4) Pedagogical clarity (definitions/examples help a student learn)
5) Style/Conciseness (clear, concise; within target length; no fluff)

Return ONLY JSON with keys:
{
  "coverage": int,
  "faithfulness": int,
  "organization": int,
  "clarity": int,
  "style": int,
  "overall_1to10": int,
  "two_strengths": ["...", "..."],
  "two_issues": ["...", "..."],
  "faithfulness_evidence": ["slide quote or paraphrase", "slide quote or paraphrase"]
}
"""

_AGREEMENT_PROMPT = """You are grading agreement with a reference answer.
Judge overlap of essential facts and omissions. Ignore stylistic differences.

Return ONLY JSON:
{
  "agreement_1to5": int,
  "missing_key_points": ["...", "..."],
  "added_inaccuracies": ["...", "..."]
}
"""

_PAIRWISE_PROMPT = """You are selecting the better study summary for students.

Choose the better overall summary for students to study for THIS lecture.
Return ONLY JSON: {"winner": "A"|"B", "reason": "..."}
"""


def _force_json(text: str) -> Dict[str, Any]:
    """
    Make a best effort to parse the model output as JSON.
    Extracts the first {...} block if strict parsing fails.
    """
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        raise ValueError(f"LLM did not return valid JSON:\n{text}")


# =========================================================
# 4) Judges (single-call versions)
# =========================================================

def judge_scores(slides: List[Dict], summary: str, cfg: LLMConfig) -> Dict[str, Any]:
    """
    Reference-free judge: compare summary directly against the slides on a rubric.
    Returns integer scores and qualitative strengths/issues.
    """
    slides_str = _slides_to_str(slides, max_chunks=3, max_tokens=1500)
    user_msg = f"[Slides]\n{slides_str}\n\n[Summary]\n{summary}\n\nReturn ONLY JSON."
    raw = call_llm(_RUBRIC_PROMPT, user_msg, cfg)
    data = _force_json(raw)
    data["_meta"] = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "seed": cfg.seed,
        "system_prompt_hash": hash(_RUBRIC_PROMPT),
        "user_len": len(user_msg),
    }
    return data


def judge_agreement(reference: str, summary: str, cfg: LLMConfig) -> Dict[str, Any]:
    """
    Reference-aware judge: compare summary to a human-written reference.
    Useful for calibration, not to overfit style.
    """
    user_msg = f"[Reference]\n{reference}\n\n[Model summary]\n{summary}\n\nReturn ONLY JSON."
    raw = call_llm(_AGREEMENT_PROMPT, user_msg, cfg)
    data = _force_json(raw)
    data["_meta"] = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "seed": cfg.seed,
        "system_prompt_hash": hash(_AGREEMENT_PROMPT),
        "user_len": len(user_msg),
    }
    return data


def pairwise_judge(slides: List[Dict], A: str, B: str, cfg: LLMConfig) -> Dict[str, Any]:
    """
    Head-to-head judge: choose the better of two summaries for the same lecture.
    """
    slides_str = _slides_to_str(slides, max_chunks=3, max_tokens=1500)
    user_msg = f"[Slides]\n{slides_str}\n\n[Summary A]\n{A}\n\n[Summary B]\n{B}\n\nReturn ONLY JSON."
    raw = call_llm(_PAIRWISE_PROMPT, user_msg, cfg)
    data = _force_json(raw)

    # Force the winner to be "A" or "B" if the model free-forms
    if data.get("winner") not in ("A", "B"):
        wtxt = json.dumps(data).lower()
        if "summary a" in wtxt or '"a"' in wtxt:
            data["winner"] = "A"
        elif "summary b" in wtxt or '"b"' in wtxt:
            data["winner"] = "B"
        else:
            data["winner"] = random.choice(["A", "B"])
            data["reason"] = (data.get("reason") or "") + " (tie-broken randomly)"

    data["_meta"] = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "seed": cfg.seed,
        "system_prompt_hash": hash(_PAIRWISE_PROMPT),
        "user_len": len(user_msg),
    }
    return data


# =========================================================
# 5) Ensemble wrappers (reduce variance)
# =========================================================

def judge_scores_ensemble(slides, summary, cfg, runs: int = 3) -> Dict[str, Any]:
    """
    Call the rubric judge multiple times (varying seed), average scalar fields,
    and carry over example text fields from the first call.
    """
    outs = []
    for r in range(runs):
        cfg_r = LLMConfig(**{**cfg.__dict__, "seed": (cfg.seed or 0) + r})
        outs.append(judge_scores(slides, summary, cfg_r))

    meanable = ["coverage","faithfulness","organization","clarity","style","overall_1to10"]
    avg = {k: int(round(np.mean([o.get(k,0) for o in outs]))) for k in meanable}
    # carry qualitative fields from run 0 for readability
    avg["two_strengths"] = outs[0].get("two_strengths", [])
    avg["two_issues"] = outs[0].get("two_issues", [])
    avg["faithfulness_evidence"] = outs[0].get("faithfulness_evidence", [])
    avg["_stdev_overall"] = float(np.std([o.get("overall_1to10",0) for o in outs], ddof=1))
    return avg


def judge_agreement_ensemble(reference, summary, cfg, runs: int = 3) -> Dict[str, Any]:
    """
    Call the agreement judge multiple times and average the agreement score.
    """
    outs = []
    for r in range(runs):
        cfg_r = LLMConfig(**{**cfg.__dict__, "seed": (cfg.seed or 0) + r})
        outs.append(judge_agreement(reference, summary, cfg_r))

    avg = {"agreement_1to5": int(round(np.mean([o.get("agreement_1to5",0) for o in outs])))}
    avg["missing_key_points"] = outs[0].get("missing_key_points", [])
    avg["added_inaccuracies"] = outs[0].get("added_inaccuracies", [])
    avg["_stdev_agreement"] = float(np.std([o.get("agreement_1to5",0) for o in outs], ddof=1))
    return avg


def pairwise_judge_ensemble(slides, A, B, cfg, runs: int = 5) -> Dict[str, Any]:
    """
    Multiple pairwise votes with A/B order shuffling to reduce position bias.
    Returns winner ("A" or "B"), per-side win counts, and a couple sample reasons.
    """
    wins = {"A":0, "B":0}
    reasons = []
    for r in range(runs):
        cfg_r = LLMConfig(**{**cfg.__dict__, "seed": (cfg.seed or 0) + r})
        if r % 2 == 0:
            res = pairwise_judge(slides, A, B, cfg_r)
            w = res.get("winner","A")
            wins[w] += 1
            reasons.append(res.get("reason",""))
        else:
            res = pairwise_judge(slides, B, A, cfg_r)
            w = res.get("winner","A")           # winner relative to swapped order
            w = "A" if w == "B" else "B"        # map back to original A/B names
            wins[w] += 1
            reasons.append(res.get("reason",""))
    final_winner = "A" if wins["A"] >= wins["B"] else "B"
    return {"winner": final_winner, "wins": wins, "reasons_sample": reasons[:2]}


# =========================================================
# 6) Scoring combiner & evaluation entry points
# =========================================================

def llm_score(
    rubric: Dict[str, Any],
    agree: Dict[str, Any],
    weights: Dict[str, float] | None = None
) -> float:
    """
    Combine rubric (5 dims) and agreement into a scalar in [0,1].
    Default reweights faithfulness higher (paper-backed).
    """
    w = weights or {"coverage":1, "faithfulness":2, "organization":1, "clarity":1, "style":1}
    denom = sum(w.values()) * 5.0
    r = (
        w["coverage"] * int(rubric.get("coverage", 0)) +
        w["faithfulness"] * int(rubric.get("faithfulness", 0)) +
        w["organization"] * int(rubric.get("organization", 0)) +
        w["clarity"] * int(rubric.get("clarity", 0)) +
        w["style"] * int(rubric.get("style", 0))
    ) / denom  # 0..1

    a = max(0, min(5, int(agree.get("agreement_1to5", 0)))) / 5.0
    return float(0.5 * r + 0.5 * a)


def evaluate_one_summary(
    slides: List[Dict],
    model_summary: str,
    human_reference: str,
    cfg: LLMConfig,
    target_words: int = 300,
) -> Dict[str, Any]:
    """
    End-to-end evaluation for a single summary:
      - deterministic signals
      - ensemble rubric judge
      - ensemble reference agreement
      - final scalar score
    """
    sig = simple_signals(slides, model_summary, target_words=target_words)
    rubric = judge_scores_ensemble(slides, model_summary, cfg, runs=3)
    agree  = judge_agreement_ensemble(human_reference, model_summary, cfg, runs=3)
    score = llm_score(rubric, agree)
    return {
        "signals": sig,
        "rubric": rubric,
        "agreement": agree,
        "final_score_0to1": score,
    }


def round_robin_pairwise(
    slides: List[Dict],
    summaries: Dict[str, str],
    cfg: LLMConfig
) -> Dict[str, Any]:
    """
    Compare all summaries pairwise using the ensemble judge.
    Returns win counts, win rates, and a match log.
    """
    names = list(summaries.keys())
    wins = {n: 0 for n in names}
    matches = []
    total_pairs = 0

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            total_pairs += 1
            Aname, Bname = names[i], names[j]
            res = pairwise_judge_ensemble(slides, summaries[Aname], summaries[Bname], cfg, runs=5)
            win_name = Aname if res["winner"] == "A" else Bname
            wins[win_name] += 1
            matches.append({
                "A": Aname, "B": Bname,
                "winner": win_name,
                "wins_detail": res["wins"],
                "reasons_sample": res["reasons_sample"],
            })

    win_rate = {k: (v / max(1, total_pairs)) for k, v in wins.items()}
    return {"wins": wins, "win_rate": win_rate, "matches": matches}


# =========================================================
# 7) Sanity test (direction of information)
# =========================================================

def sanity_direction_of_info(slides, summary, cfg) -> Dict[str, Any]:
    """
    Faithfulness should drop when the judge cannot see the slides.
    Use in quick CI checks to catch prompt regressions.
    """
    r_full = judge_scores_ensemble(slides, summary, cfg, runs=2)
    r_blind = judge_scores_ensemble([], summary, cfg, runs=2)
    return {
        "faithfulness_full": r_full["faithfulness"],
        "faithfulness_blind": r_blind["faithfulness"],
        "passed": r_full["faithfulness"] > r_blind["faithfulness"]
    }


# =========================================================
# 8) Minimal smoke test (run file directly)
# =========================================================

if __name__ == "__main__":
    # Example slides and summaries
    slides = [
        {"title":"Gradient Descent","content":"Update parameters opposite the gradient to minimize loss."},
        {"title":"Learning Rate","content":"Too high diverges; too low slows convergence; schedules can help."},
        {"title":"Stopping Criteria","content":"Use validation loss, gradient norm, or max steps."},
    ]
    human_ref = "Covers update rule, impact of learning rate, and stopping criteria."
    model_sum = "Update parameters using the negative gradient. High LR diverges. Stop by validation or gradient."

    # Configure your model (wire call_llm() first to actually run judges)
    cfg = LLMConfig(provider="openai", model="gpt-4o-mini", temperature=0.2, max_tokens=512, seed=7)

    # Deterministic signals run without an LLM:
    print("Signals (no LLM needed):", simple_signals(slides, model_sum, target_words=80))

    # The lines below require a real call_llm implementation:
    # res = evaluate_one_summary(slides, model_sum, human_ref, cfg)
    # print("Eval:", res)
    # rr = round_robin_pairwise(slides, {"A": model_sum, "B": "Another summary..."}, cfg)
    # print("Pairwise:", rr)
