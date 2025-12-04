# src/utils/io.py

"""
IO utilities for:
- Loading PDF or JSON lecture content
- Writing iterative summaries (iter_0.txt, iter_1.txt, ...)
- Managing experiment output folders
"""

import os
import json
from typing import Dict, Any
from pathlib import Path

from src.utils.pdf_parser import extract_slides_from_pdf


# ---------------------------------------------
# Loading lecture slides (PDF or JSON)
# ---------------------------------------------

def load_slides(path: str) -> Dict[str, Any]:
    """
    Load lecture slides from:
        - PDF (auto-parsed)
        - JSON (must contain {lecture_title, slides: [...]})
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Slide file not found: {path}")

    if path.suffix.lower() == ".pdf":
        return extract_slides_from_pdf(str(path))

    elif path.suffix.lower() == ".json":
        with open(path, "r") as f:
            return json.load(f)

    else:
        raise ValueError(f"Unsupported slide format: {path.suffix}")


# ---------------------------------------------
# Writing iterative summaries
# ---------------------------------------------

def write_iteration_summary(output_dir: str, iteration: int, text: str):
    """
    Save the summary for a refinement iteration:
        iter_0.txt, iter_1.txt, iter_2.txt, ...
    """

    if text is None:
        text = ""

    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"iter_{iteration}.txt"

    with open(out_path, "w") as f:
        f.write(text.strip() + "\n")


def write_final_summary(output_dir: str, text: str):
    """
    Save the final refined summary as final.txt.
    """

    if text is None:
        text = ""

    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / "final.txt"

    with open(out_path, "w") as f:
        f.write(text.strip() + "\n")


# ---------------------------------------------
# Saving full evaluation result
# ---------------------------------------------

def write_json(path: str, data: Dict[str, Any]):
    """
    Serialize evaluation results to JSON.
    Handles numpy types by coercing via default=str.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
