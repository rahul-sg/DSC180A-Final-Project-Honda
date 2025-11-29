# DSC180A-Final-Project-Honda

Automated Lecture Summarization & LLM-Based Evaluation Pipeline

This project builds an end-to-end pipeline that takes lecture slides or documents as input, uses an LLM to generate a summary, and then uses another LLM as an evaluator to grade the summary based on rubric-style criteria such as accuracy, coverage, and hallucination detection. The system also runs iterative refinement loops, where the summary is repeatedly evaluated and improved based on LLM feedback.

This project was developed for DSC 180A, UCSD.

Features

- PDF / DOCX ingestion: Extracts text from lecture slides and documents.

- LLM summarization: Uses a smaller LLM (gpt-5-mini) to generate a concise and accurate summary.

- LLM evaluation / grading: Uses an evaluator model (gpt-5-nano) to judge accuracy, coverage, and hallucinations.

- Iterative feedback loop: Each evaluation triggers a revised summary, improving quality over 3 iterations.

- Modular design: Pipeline functions are cleanly separated (extract → summarize → judge → revise).

- Easily extensible: Swap in different models, rubrics, or iteration counts.
