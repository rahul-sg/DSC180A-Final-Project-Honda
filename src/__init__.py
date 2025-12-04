# src/__init__.py

"""
Top-level package for the DSC180A Final Project (Honda Group).

This project implements:
- PDF → Slide parsing
- Deterministic evaluation signals
- LLM judges (rubric, agreement, pairwise)
- Iterative summary refinement
- Complete evaluation pipeline

Subpackages:
    utils/       → Chunking, signals, PDF parsing, IO helpers
    models/      → LLM client, judge, refinement
    evaluation/  → Scoring, pipeline, pairwise comparisons
    experiments/ → Runner scripts and demos
"""
