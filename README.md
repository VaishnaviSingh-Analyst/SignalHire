# SignalHire — AI-Powered Candidate Ranking

> Given 100,000 candidate profiles, rank the **top 100** most suitable candidates for a **Senior AI Engineer** role — built for the Redrob AI Challenge.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-green?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![sentence-transformers](https://img.shields.io/badge/sentence--transformers-all--MiniLM--L6--v2-orange)](https://www.sbert.net)

---

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Quick Start](#quick-start)
- [How to Run](#how-to-run)
  - [Precompute (GPU)](#1-precompute-gpu)
  - [Rank (CPU)](#2-rank-cpu)
  - [Streamlit Dashboard](#3-streamlit-dashboard)
  - [Docker](#docker)
- [Scoring Methodology](#scoring-methodology)
- [Disqualification & Penalties](#disqualification--penalties)
- [Project Structure](#project-structure)
- [Output Format](#output-format)
- [Sandbox Constraints](#sandbox-constraints)
- [Performance](#performance)

---

## Pipeline Overview

```
                   candidates.jsonl (100K)
                           │
                           ▼
          ┌─────────────────────────────────┐
          │  PHASE A: Precompute (GPU)      │  ~4 min
          │  ───────────────────────        │
          │  · Stream & parse JSONL         │
          │  · Disqualify bad actors        │  honeypot / ghost / pure research
          │  · Embed profiles               │  all-MiniLM-L6-v2 → 384-dim
          │  · Compute 4 sub-scores         │  technical, career, availability, seniority
          │  · Serialize artifacts          │  .npy + .pkl → ~155 MB
          └──────────────┬──────────────────┘
                           │
                           ▼
          ┌─────────────────────────────────┐
          │  PHASE B: Ranking (CPU)         │  <2 min
          │  ──────────────────────         │
          │  · Load cached artifacts        │
          │  · Cosine similarity (vectorized)│
          │  · Composite score              │  S = penalty × Σ(wᵢ · scoreᵢ)
          │  · argpartition → top 100       │
          │  · Generate reasoning           │
          │  · Validate & write CSV         │
          └──────────────┬──────────────────┘
                           │
                           ▼
                    submission.csv
                    (100 ranked candidates)
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA (optional, for faster precompute)
- ~500 MB disk space for data + ~160 MB for artifacts

### Setup

```bash
# 1. Clone & enter
cd SignalHire

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### 1. Precompute (GPU)

Processes all 100K candidates: disqualifies bad actors, generates embeddings, computes sub-scores.

> **GPU recommended.** The config defaults to `device="cuda"` in `config.py`.  
> No GPU? Set `EMBEDDING_DEVICE = "cpu"` in `config.py:29` (will be slower, ~30–60 min).

```bash
python precompute.py
```

**Artifacts produced** (saved to `artifacts/`):

| File | Size | Description |
|---|---|---|
| `embeddings.npy` | ~147 MB | 384-dim embeddings for 99,965 candidates |
| `candidate_ids.npy` | ~1.5 MB | Parallel array of candidate IDs |
| `jd_embedding.npy` | ~1.7 KB | JD embedding for cosine similarity |
| `subscores.pkl` | ~7.1 MB | Dict of sub-scores + penalty multipliers |
| `disqualified.json` | ~4 KB | Log of disqualified candidates |

### 2. Rank (CPU)

Loads artifacts, computes composite scores, selects top 100, generates reasoning, validates output.

> Runs entirely on **CPU** using NumPy. No GPU needed.

```bash
python rank.py
```

**Output:** `output/submission.csv`

### 3. Streamlit Dashboard

A live, explainable ranking workbench — every control re-ranks all 100K candidates
in milliseconds because scoring is a single matrix multiply over precomputed artifacts.

```bash
streamlit run app.py
```

| Feature | What it does |
|---|---|
| **Live weight sliders** | Drag any signal weight and watch the shortlist reshuffle instantly |
| **Custom JD search** | Paste any job description or plain-English query ("RAG engineer, short notice") — it's embedded and ranked on the fly |
| **Evidence-cited cards** | Each candidate shows *which* JD requirement matched *which* exact skill or profile sentence, plus flagged gaps |
| **Stability badges** | Each candidate's rank is stress-tested under 200 random ±20% weight perturbations — "stable 98%" means the pick isn't an artifact of one weight choice |
| **Diversity (MMR) slider** | Penalizes near-duplicate profiles so the shortlist covers distinct archetypes |
| **Blind screening mode** | Hides names, companies and institutions to reduce reviewer bias |
| **Compare** | Radar-chart side-by-side of up to 4 candidates |
| **Insights** | Score landscape + fairness audit (shortlist vs pool by education tier, country, YoE) |
| **Integrity** | Honeypot/ghost showcase with concrete caught examples |
| **Export** | Submission CSV, personalized outreach pack, reproducible ranking config |

### Docker

```bash
docker build -t recruiteriq .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output recruiteriq python rank.py
```

> **Note:** The Docker image pre-downloads the embedding model. For GPU access, add `--gpus all`.

---

## Scoring Methodology

### Composite Score Formula

```
S = penalty_multiplier × (
    0.35 × technical_fit
  + 0.25 × career_quality
  + 0.20 × availability_signal
  + 0.12 × seniority_fit
  + 0.08 × semantic_similarity
)
```

All sub-scores normalized to [0, 1].

### Signal Breakdown

| Signal | Weight | What It Measures |
|---|---|---|
| **Technical Fit** | 0.35 | JD skill match — embeddings, vector DBs, Python, eval frameworks |
| **Career Quality** | 0.25 | Product company history, tenure stability, title progression |
| **Availability** | 0.20 | Open to work, response rate, notice period, interview completion |
| **Seniority Fit** | 0.12 | YoE range (6–9 ideal), education tier bonus |
| **Semantic Similarity** | 0.08 | Embedding cosine vs JD — catches strong engineers with non-obvious keywords |

### Sub-Score Components

**Technical Fit:**
- Must-have: embeddings/retrieval (0.25), vector databases (0.20), Python (0.15), eval frameworks (0.15)
- Nice-to-have: LLM fine-tuning (0.10), learning-to-rank (0.10), HR-tech (0.05)
- Bonuses: production signals, retrieval signals

**Career Quality:**
- Non-consulting role: +0.30
- ML/AI title at 50+ company: +0.15
- Median tenure ≥36mo: +0.25 (≥24mo: +0.20, ≥18mo: +0.05)
- Upward title progression: +0.20 (career sorted chronologically before comparing)

**Availability:**
- Open to work: +0.25
- Recent activity (≤30d): +0.20
- Response rate: up to +0.15
- Interview completion rate: up to +0.15
- Short notice period (≤30d): +0.05

**Seniority Fit:**
- YoE 6–9: 1.0 | 4–5 or 10–12: 0.7 | 3 or 13–15: 0.4 | else: 0.1
- Tier-1 education: +0.05 | Tier-2: +0.02

---

## Disqualification & Penalties

### Hard Disqualify (Removed from Pool)

| Rule | Trigger |
|---|---|
| **Honeypot** | Years of experience exceeds career timeline + 5yr buffer |
| **Ghost** | Profile completeness < 5% + no verified email/phone |
| **Pure Research** | All roles are "researcher" with no production evidence |

### Soft Penalties (Score Multipliers)

| Condition | Multiplier | Effect |
|---|---|---|
| All roles at consulting firms (TCS, Infosys, Wipro, etc.) | × **0.15** | Severely penalizes consulting-only careers |
| No coding activity > 18 months | × **0.80** | Red flags stale skills |
| CV / Speech / Robotics only with no retrieval signals | × **0.85** | Niche focus, poor JD alignment |

---

## Project Structure

```
SignalHire/
├── config.py                # Weights, paths, keyword lists, penalties
├── disqualify.py            # Honeypot/ghost/research detection
├── signals.py               # 4 sub-score functions
├── evidence.py              # Evidence extraction + honest reasoning generation
├── engine.py                # Vectorized re-ranking, MMR diversity, stability analysis
├── precompute.py            # Phase A: ingest → embed → score → serialize
├── rank.py                  # Phase B: load → score → top-100 → CSV
├── app.py                   # Interactive Streamlit dashboard
├── .streamlit/config.toml   # Dark theme
├── requirements.txt         # Python dependencies
├── Dockerfile               # Python 3.10-slim container
├── data/
│   ├── candidates.jsonl     # 100K candidate profiles (~487 MB)
│   ├── job_description.docx # Target job description
│   ├── validate_submission.py  # Challenge submission validator
│   ├── candidate_schema.json   # Data schema definition
│   └── sample_*             # Sample candidates & submissions
├── artifacts/               # Generated by precompute.py
│   ├── embeddings.npy
│   ├── candidate_ids.npy
│   ├── jd_embedding.npy
│   ├── subscores.pkl
│   └── disqualified.json
├── output/
│   └── submission.csv       # Final ranked output
└── Documentation/
    ├── PRD.md
    ├── TRD.md
    ├── APP_FLOW.md
    └── ...
```

---

## Output Format

`output/submission.csv` — validated per challenge spec:

```csv
candidate_id,rank,score,reasoning
CAND_0081846,1,0.870,"6.7yr Lead AI Engineer at Razorpay; strong match on embeddings, vector search, python, information retrieval; production evidence (serving, ndcg); actively looking, 73% response rate, 30d notice."
CAND_0055905,2,0.869,"8.1yr Senior Machine Learning Engineer at Flipkart; strong match on embeddings, vector search, python, information retrieval; production evidence (deployed, serving); actively looking, 87% response rate."
...
```

Reasoning strings are **evidence-based**: skills and production signals are only
claimed when they actually appear in the profile, and missing must-haves are
called out as gaps.

- **100 rows** (ranks 1–100)
- **Scores non-increasing** by rank
- **Tie-breaking** by candidate_id ascending
- Each candidate includes a **reasoning snippet** explaining the match

---

## Sandbox Constraints

| Constraint | How It's Met |
|---|---|
| **CPU-only ranking** | Phase B uses NumPy only — no GPU dependency |
| **16 GB RAM limit** | Streaming JSONL parser, batched embedding, vectorized NumPy operations |
| **No network** | Model pre-downloaded in Docker build (`SentenceTransformer` cached) |
| **<5 min ranking** | `np.argpartition` — O(N) selection, ~155 MB artifact load |

---

## Performance

Measured on **NVIDIA RTX 3050 (6 GB VRAM) + 12-core CPU**:

| Phase | Device | Time | Throughput |
|---|---|---|---|
| Precompute (100K → 99,965) | GPU (CUDA) | **~4.2 min** | ~400 candidates/sec |
| Ranking & Validation | CPU | **~1.5 min** | ~65K candidates/sec |
| **Total pipeline** | Hybrid | **~5.7 min** | 100K → top 100 |

---

*Built for the Redrob AI Challenge — India Runs Hackathon*
