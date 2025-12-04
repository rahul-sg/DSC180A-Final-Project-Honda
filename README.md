# 📘 DSC180A – Evaluation Strategies for Next-Generation AI Systems

This repository implements a complete pipeline for generating, refining, and evaluating large‑language‑model (LLM) summarizations of university lecture materials. The system uses as an example: **lecture PDFs**. 

This system processes the slides and produces an initial summary (S₀), iteratively refines the summary using rubric‑based improvement steps, and evaluates the final output using both deterministic metrics and LLM‑based judges. An interactive Streamlit dashboard is included for exploration of results.

## 📁 Project Structure

```
DSC180A-Final-Project/
├── data/
│   ├── references/
│   ├── slides/
│   └── summaries/
│       ├── model_s0/
│           └── lectureX.txt
│       └── refined_iterations/
│           └── lectureX/
│               ├── iter_0.txt … iter_n.txt
│               ├── final.txt
│               └── result.json
│
├── src/
│   ├── evaluation/
│   ├── experiments/
│   ├── models/
│   ├── utils/
│   └── visualization/
│
├── environment.yml
├── requirements.txt
├── eval_baseline.py
└── README.md
```

## 🛠 Environment Setup

### Start-Up Scripts

#### If you have Mac/Linux (bash)

```bash
chmod +x startup.sh
source startup.sh
```

#### If you have Windows (PowerShell)
```powershell
powershell -ExecutionPolicy Bypass -File startup.ps1
```

### Running the Start-Up Script does the following (you don't have to do this if you ran the startup script)

#### Create environment
```bash
conda env create -f environment.yml
conda activate dsc180a-eval
```

#### Environment variables
Create `.env`:
```
OPENAI_API_KEY=your_key_here
```

### To deactivate conda environment at anytime, run
```bash
conda deactivate
```

## ▶️ Running Evaluation
Evaluate a lecture:
```bash
python -m src.experiments.run_eval lecture1 yes
```

Parameters:
- 0: `lecture[Number]` - replace number with which lecture you want to test
- 1: `yes` - takes `yes` or `no` by default if omitted and forces regeneration of S0

This generates:
```
data/summaries/refined_iterations/lecture1/
    iter_0.txt
    iter_1.txt
    iter_2.txt
    iter_3.txt
    final.txt
    result.json
```
And: 

```
data/slides/lecture1/
    lecture1.txt
```

## 📊 Dashboards

#### NOTE: RUN THESE AFTER YOU RUN THE INITIAL MODEL SO THE FILES ARE GENERATED

### Static Dashboard
Run:
```bash
python -m src.visualization.dashboard lecture1
```

### Interactive Streamlit Dashboard
Run:
```bash
streamlit run src/visualization/interactive_dashboard.py
```

Includes:
- Summary length trends  
- Deterministic signal bar charts  
- Rubric radar chart  
- Agreement score  
- Semantic drift visualization  
- Heatmap of rubric vs. agreement  
- Full text viewer for iteration outputs  

## 🚀 Workflow Overview

### 1. Preprocessing
Lecture PDFs are converted into text and segmented using utilities in `src/utils/`.

### 2. Initial Summarization (S₀)
`models/summarizer.py` generates a baseline summary stored under  
`data/summaries/model_s0/`.

### 3. Iterative Refinement
`models/refinement.py` improves S₀ over multiple iterations. Outputs are saved as  
`iter_0.txt`, `iter_1.txt`, … and `final.txt`.

### 4. Evaluation
Evaluation combines:

#### Deterministic Metrics (`signals.py`)
- **Length Error**
- **Section Coverage %**
- **Glossary Recall**
- **Suspected Hallucination Rate**

#### Rubric‑Based LLM Evaluation (`judge.py`)
Scores (1–5):
- Coverage  
- Faithfulness  
- Organization  
- Clarity  
- Style  

Also includes:
- Two strengths  
- Two issues  
- Faithfulness evidence  

#### Agreement Judge
Compares final summary to reference:
- agreement_1to5  
- missing_key_points  
- added_inaccuracies  

### Example `result.json`
Contains:
```
refined_summary
signals → { length_error, section_coverage_pct, glossary_recall, ... }
rubric → { coverage, faithfulness, organization, clarity, style, ... }
agreement → { agreement_1to5, missing_key_points, added_inaccuracies }
final_score_0to1
lecture_title
```

## 🧪 Example Test Data

We have provided some basic test results within the following domains:
- `data/summaries/refined_iterations/lecture1/` - UCSD MGT 45 (Financial & Managerial Accounting) [Dr. Andreya Pérez Silva] - Week 1 Slides
- `data/summaries/refined_iterations/lecture2/` - UCSD MGT 45 (Financial & Managerial Accounting) [Dr. Andreya Pérez Silva] - Week 2 Slides
- `data/summaries/refined_iterations/lecture2/` - UCSD LATI 10 (Reading North by South: Latin American Studies and the US Liberation Movements) [Dr. Amy Kennemore] - Week 3 Slides
- `data/summaries/refined_iterations/lecture4/` - UCSD ANTH 2 (Human Origins) [Maria Carolina Marchetto, PhD] - Week 2 Slides
- `data/summaries/refined_iterations/lecture5/` - UCSD EDS/SOCI 117 (Language, Culture, and Education) [Gabrielle Jones, Ph.D.] - Week 2 Wednesday Slides
- `data/summaries/refined_iterations/lecture6/` - UCSD 
- `data/summaries/refined_iterations/lecture7/` - UCSD

The LLM-Generated initial summaries for each respective test-set are here:
```
data/slides/lecture1.txt
data/slides/lecture2.txt
data/slides/lecture3.txt
data/slides/lecture4.txt
data/slides/lecture5.txt
data/slides/lecture6.txt
data/slides/lecture7.txt
```
And the Human-Written summaries for each respective test-set are here:
```
data/references/lecture1_reference.txt
data/references/lecture2_reference.txt
data/references/lecture3_reference.txt
data/references/lecture4_reference.txt
data/references/lecture5_reference.txt
data/references/lecture6_reference.txt
data/references/lecture7_reference.txt
```

## 💽 To Add Your Own Data

1. Add a PDF file of your lecture slide under the `data/slides/` folder with the following naming scheme: `lectureN.pdf` where N is the next available number in the folder (you would start with `lecture8.pdf`)
2. Write a 250-300 word human summarization for your lecture slides and save it under the `data/references/` folder with the following naming scheme: `lectureN_reference.txt` where N is the next available number in the folder (you would start with `lecture8_reference.txt`)
3. Navigate back to the root folder `DSC180A-Final-Project-Honda` and run the program


## 📜 License

This project was developed for the UC San Diego DSC180A Capstone for Fall 2025.

Evaluation Strategies for Next-Generation AI Systems

Industry Partner - **Honda Research Labs**

## 👥 Authors
Rahul Sengupta | Akshay Medidi | Zeyu (Edward) Qi | Zachary Thomason 
## 👥 Mentors
Rajeev Chhajer | Ryan Lingo