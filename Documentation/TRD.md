# Technical Requirements Document (TRD)
**RecruiterIQ — AI Candidate Ranking System**

| Field | Value |
|---|---|
| Version | 1.0 |
| System | RecruiterIQ — Offline Hybrid Scoring Pipeline |
| Constraint | CPU-only · 16GB RAM · 5 min ranking · No network at inference |

---

## 1. System Overview

Two-phase system:
- **Phase A — Precomputation** (offline, before submission): embed 100k candidates, compute sub-scores, serialize artifacts
- **Phase B — Ranking** (sandbox execution, ≤5 min): load artifacts, score, sort, emit CSV

---

## 2. Functional Requirements

### FR-01: Data Ingestion
- Parse `candidates.jsonl` (100k lines, streaming — do not load all into RAM at once)
- Validate each record against `candidate_schema.json`
- Handle missing fields gracefully (default values, not crashes)
- Stream batch size: 1,000 records per chunk

### FR-02: Disqualification
- Hard-drop honeypots before any scoring (saves compute)
- Disqualify conditions (see Disqualify Rules below)
- Flag output: `{candidate_id, disqualify_reason}` for audit log

### FR-03: Profile Embedding
- Build profile blob per candidate (headline + summary + career descriptions + skill names)
- Embed using `all-MiniLM-L6-v2` (384-dim, CPU-optimized)
- Batch size: 512 for memory efficiency
- Output: `(N, 384)` float32 numpy array, saved as `artifacts/embeddings.npy`
- Also save `artifacts/candidate_ids.npy` (aligned index → candidate_id map)

### FR-04: JD Embedding
- Embed JD text once: `artifacts/jd_embedding.npy`
- JD text = full `job_description.docx` content (must be extracted to `.txt` first)

### FR-05: Sub-Score Computation
- Compute `technical_fit`, `career_quality`, `availability_signal`, `seniority_fit` per candidate
- Serialize to `artifacts/subscores.pkl` (dict: candidate_id → score dict)
- All scores: float32, range [0.0, 1.0]

### FR-06: Composite Scoring (Ranking Phase)
- Load `embeddings.npy`, `jd_embedding.npy`, `subscores.pkl`
- Compute `semantic_similarity` = cosine(candidate_embedding, jd_embedding)
- Compute composite S per candidate (see formula in PRD)
- Sort descending, take top 100
- Enforce monotonic scores: if `score[i] < score[i+1]`, clamp

### FR-07: Reasoning Generation
- Per top-100 candidate: generate 1–2 sentence string
- Must reference: years_of_experience, top skill match, strongest company, availability status
- Must NOT be generic filler (e.g., "Strong candidate with relevant skills")
- Length: 120–300 characters

### FR-08: CSV Output
- Columns: `candidate_id, rank, score, reasoning`
- Exactly 100 rows + header
- Ranks 1–100, each exactly once
- Score: float, 3 decimal places, non-increasing
- Encoding: UTF-8
- Filename: `<participant_id>.csv`

### FR-09: Validation
- Run `validate_submission.py` on output
- Must pass with zero errors before submission

### FR-10: Demo Sandbox
- Streamlit app accepting: JD text + candidate JSON → returns ranked top-10 with score breakdown
- Deployable to HuggingFace Spaces (CPU, public)
- Loads pre-computed artifacts from repo

---

## 3. Non-Functional Requirements

| Requirement | Target |
|---|---|
| Ranking runtime (Phase B) | < 300 seconds on CPU, 16GB |
| Precomputation runtime | < 60 minutes on CPU (offline, no constraint) |
| Peak RAM (Phase B) | < 14GB (leave 2GB buffer) |
| Embedding model size | < 200MB |
| Output correctness | 0 format errors on validator |
| Honeypot rate in top 100 | ≤ 10 (hard disqualification threshold) |
| Reproducibility | Same seed → same output, every run |

---

## 4. Disqualification Rules (Hard Filters)

```python
CONSULTING_FIRMS = [
    "TCS", "Infosys", "Wipro", "Capgemini", "Cognizant",
    "Accenture", "HCL", "Mphasis", "Tech Mahindra", "Hexaware",
    "IBM GBS", "Deloitte", "EY", "KPMG", "Mindtree"
]

def is_honeypot(candidate) -> bool:
    # Rule 1: Experience > career timeline
    yoe = candidate['profile']['years_of_experience']
    earliest_start = min(parse_year(r['dates']['start']) for r in candidate['career_history'])
    if yoe > (2025 - earliest_start + 2):  # +2 tolerance
        return True
    
    # Rule 2: Ghost profile
    s = candidate['redrob_signals']
    if (s['profile_completeness_score'] < 5 and
        not s['verified_email'] and not s['verified_phone']):
        return True
    
    return False

def should_hard_penalize(candidate) -> float:
    # Returns penalty multiplier [0.0 – 1.0]; 1.0 = no penalty
    roles = candidate['career_history']
    companies = [r['company'] for r in roles]
    
    # All consulting, no product company
    consulting_count = sum(1 for c in companies if any(f in c for f in CONSULTING_FIRMS))
    if consulting_count == len(companies):
        return 0.15  # Severe penalty, not hard drop (might have 1 good role missed)
    
    return 1.0
```

---

## 5. Embedding Strategy

### Model: `sentence-transformers/all-MiniLM-L6-v2`
- Size: 80MB
- Dimensions: 384
- CPU inference: ~0.5ms/sample
- 100k candidates: ~50,000ms = ~50 seconds (batched at 512)

### Profile Blob Construction
```python
def build_profile_blob(candidate) -> str:
    parts = []
    p = candidate['profile']
    parts.append(p.get('headline', ''))
    parts.append(p.get('summary', ''))
    parts.append(p.get('current_title', ''))
    
    for role in candidate.get('career_history', []):
        parts.append(role.get('title', ''))
        parts.append(role.get('description', ''))
    
    # Skills: name only, proficiency as qualifier
    for skill in candidate.get('skills', []):
        prof = skill.get('proficiency', '')
        name = skill.get('name', '')
        if prof in ['expert', 'advanced']:
            parts.append(f"expert {name}")
        else:
            parts.append(name)
    
    return ' '.join(filter(None, parts))[:512]  # Truncate at 512 tokens
```

### RAM Estimate for Embeddings
- 100k × 384 dims × 4 bytes (float32) = **153MB** — trivially fits in 16GB

---

## 6. Scoring Engine Requirements

### Technical Fit Computation
```
Input:  candidate skills[], career_history[].description, skill_assessment_scores{}
Output: float [0.0, 1.0]

Logic:
  1. Build candidate skill set with proficiency weights
  2. Check skill_assessment_scores for platform-verified overrides
  3. Score against JD_MUST_HAVES and JD_NICE_TO_HAVES
  4. Check career descriptions for production evidence keywords
  5. Normalize to [0, 1]
```

### Production Evidence Keywords (for career description scan)
```python
PRODUCTION_SIGNALS = [
    "production", "deployed", "shipped", "serving", "inference",
    "million users", "at scale", "latency", "throughput",
    "A/B test", "evaluated", "benchmark", "NDCG", "MRR"
]
RETRIEVAL_SIGNALS = [
    "embeddings", "vector", "semantic search", "RAG", "retrieval",
    "sentence-transformers", "FAISS", "Pinecone", "Milvus", "Qdrant",
    "Weaviate", "OpenSearch", "Elasticsearch", "BM25", "dense retrieval"
]
```

### Availability Signal Computation
```
Input:  redrob_signals{}
Output: float [0.0, 1.0]

Weighted combination:
  open_to_work_flag       × 0.25
  last_active_recency()   × 0.20
  recruiter_response_rate × 0.15
  response_speed_score()  × 0.10
  interview_completion    × 0.15
  applications_30d_score()× 0.10
  notice_period_score()   × 0.05
```

---

## 7. Artifact Manifest

| Artifact | Size (est.) | Description |
|---|---|---|
| `artifacts/embeddings.npy` | ~153MB | float32 (100k, 384) |
| `artifacts/candidate_ids.npy` | ~6MB | str array aligned to embeddings |
| `artifacts/jd_embedding.npy` | ~1.5KB | float32 (384,) |
| `artifacts/subscores.pkl` | ~20MB | dict: cand_id → {tech, career, avail, senior} |
| `artifacts/disqualified.json` | ~1MB | list of {cand_id, reason} |

---

## 8. Seed & Reproducibility

```python
import numpy as np, random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
```

Deterministic output required for Stage 3 sandbox reproduction.

---

## 9. Constraints Summary

| Constraint | Handling |
|---|---|
| No network at ranking | All models/artifacts bundled or pre-downloaded |
| CPU only | MiniLM (no GPU dep), pure numpy cosine |
| 16GB RAM | Streaming ingestion, float32 arrays, no full JSONL in RAM |
| 5 min runtime | Phase B only loads + scores (no embedding at rank time) |
| Max 3 submissions | `validate_submission.py` mandatory before each |

---

## 10. Dependencies

```
sentence-transformers==2.7.0
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2   # cosine_similarity utility
tqdm==4.66.4          # progress bars
pyyaml==6.0.1         # metadata template
streamlit==1.35.0     # demo sandbox
python-docx==1.1.2    # JD extraction
```

---

*RecruiterIQ TRD v1.0 · 2025*
