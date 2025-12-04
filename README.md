# 📘 **DSC180A – Lecture Summarization Evaluation System**  
### *LLM-as-Judge baseline with iterative refinement, deterministic signals, and ensemble scoring*

This repository contains a reproducible evaluation pipeline for lecture summarization using a hybrid **LLM-as-judge** approach. It implements:

- Reference-free rubric scoring  
- Reference-aware agreement scoring  
- Deterministic faithfulness and hallucination detection signals  
- Iterative summary refinement  
- Pairwise comparison between summaries  
- Ensemble scoring to reduce variance  
- A fully pluggable OpenAI API backend  

This project follows the methodology outlined in our capstone work and is designed for extensibility and research transparency.

---

## 🚀 **Features**

### ✔ Reference-free evaluation  
Compares a model summary to lecture slides using an LLM judge.

### ✔ Reference-aware evaluation  
Measures agreement with a human-written summary.

### ✔ Iterative refinement loop (*S₀ → S₁ → S₂ → S₃*)  
Summaries are improved using judge feedback.

### ✔ Deterministic non-LLM metrics  
- Length deviation  
- Keyword coverage  
- Glossary recall  
- Hallucination rate using TF-IDF sentence retrieval  

### ✔ Ensemble (multi-seed) judging  
Reduces variance in LLM outputs.

### ✔ Pluggable architecture  
All LLM calls are centralized in `call_llm()`.

---

## 📂 **Repository Structure**

```
.
├── eval_baseline.py        # Main evaluation and refinement logic
├── environment.yml         # Conda environment (recommended for reproducibility)
├── .env.example            # Template for API keys (safe to commit)
├── .gitignore              # Excludes virtual environments and secrets
└── README.md               # This file
```

---

## ⚙️ **Installation**

### 1. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate dsc180a-eval
```

Verify Python:

```bash
python --version   # Should show Python 3.10+
```

---

## 🔑 **Environment Variables**

Copy the example:

```bash
cp .env.example .env
```

Edit `.env` and insert your key:

```
OPENAI_API_KEY=sk-xxxx...
```

The project uses `python-dotenv` to load this file automatically.

---

## ▶️ **Running the Evaluation Pipeline**

In the project root:

```bash
python eval_baseline.py
```

This will:

1. Load example slides + summary  
2. Compute deterministic signals  
3. Run LLM rubric judge  
4. Run agreement judge  
5. Perform iterative refinement  
6. Output the final refined summary  
7. Compute a final numeric score  

---

## 🧪 **Using Your Own Slides or Summaries**

Replace the example block in `eval_baseline.py`:

```python
slides = [...]
human_ref = "..."
model_sum = "..."
```

Or load them from files:

```python
import json
slides = json.load(open("slides.json"))
model_sum = open("summary.txt").read()
```

---

## 🔄 **Enabling / Disabling Iterative Refinement**

In the final evaluation call:

```python
res = evaluate_one_summary(
    slides,
    model_sum,
    human_ref,
    cfg,
    refine_iters=3   # number of refinement steps
)
```

Set:

```python
refine_iters=0
```

to disable refinement.

---

## 📊 **What the Output Looks Like**

The evaluation returns a structured dictionary:

```json
{
  "refined_summary": "...",
  "signals": {...},
  "rubric": {...},
  "agreement": {...},
  "final_score_0to1": 0.82
}
```

- **signals** = deterministic metrics  
- **rubric** = reference-free LLM scoring  
- **agreement** = alignment with reference summary  
- **final_score_0to1** = combined scalar score  

---

## 🧩 **Customizing the LLM Backend**

All API calls live in **one function**:

```python
def call_llm(system, user, cfg):
    ...
```

You can replace this logic with:

- Anthropic Claude  
- Azure OpenAI  
- Local models (llama.cpp / vLLM)  
- Fine-tuned LLMs  

---

## 🔒 **Security Notes**

- `.env` **must not be committed**  
- Use `.env.example` for others to know what variables are needed  
- Never hardcode API keys in your Python files  

---

## 📜 **License**

This project is for academic use as part of UCSD’s DSC180A capstone course.  
Forking, modifying, and extending is encouraged for research purposes.

---

## 🙋‍♂️ **Need Additional Files?**

I can generate:

- A polished `environment.yml`  
- A matching `requirements.txt`  
- A preconfigured project structure  
- A Jupyter notebook version of the evaluation  
- A visual architecture diagram of the pipeline  

Just ask: **“Generate the extras.”**
