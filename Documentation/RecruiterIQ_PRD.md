# RecruiterIQ — Product Requirements Document
**AI Candidate Ranking System | India Runs Hackathon (Redrob AI)**

| Field | Value |
|---|---|
| Version | 2.0 |
| Challenge | Intelligent Candidate Discovery & Ranking |
| Target Role | Senior AI Engineer — Founding Team @ Redrob AI |
| Dataset | 100,000 candidates (`candidates.jsonl`) |
| Output | Top 100 ranked candidates (`<participant_id>.csv`) |
| Sandbox Constraint | CPU-only · 16GB RAM · 5 min runtime · No network calls |

---

## 1. Problem Restatement

Rank top 100 out of 100,000 candidates for a **Senior AI Engineer** role. Dataset contains:
- Keyword stuffers (AI skills on non-technical profiles)
- Honeypots (~80 impossible profiles — disqualify if >10 in top 100)
- Behavioral twins (same skills, wildly different availability)
- Plain-language strong engineers who don't buzzword-match

**Core trap to avoid:** Ranking by skills keyword density. A candidate whose skills list contains 15 AI terms but has "Marketing Manager" in career history is noise, not signal.

---

## 2. Winning Approach

### Architecture Decision: Offline Hybrid Scoring (No API calls at inference time)

```
Pre-computation (offline, before submission):
  candidates.jsonl
      → Normalize + Disqualify
      → Embed candidate profiles (sentence-transformers, CPU)
      → Compute per-signal sub-scores
      → Cache scoring artifacts

Ranking step (must run in < 5 min, CPU, 16GB):
  Load cached artifacts
      → Weighted composite score per candidate
      → Sort → Take top 100
      → Generate reasoning strings
      → Write CSV
```

**Why this wins:**
- No LLM API needed at ranking time (satisfies no-network constraint)
- Pre-computed embeddings = fast inference pass
- Multi-signal scoring defeats keyword stuffers AND honeypots
- Behavioral signals surface available candidates over unavailable ones
- Fully reproducible in Docker sandbox

---

## 3. Disqualification Layer (Pre-filter before scoring)

Run before any scoring. Hard removes:

| Rule | Signal | Threshold |
|---|---|---|
| Consulting-only | All career roles at TCS/Infosys/Wipro/Capgemini/Accenture | Drop if zero product company roles |
| Pure research | All roles are "Researcher" with no deployment evidence | Drop |
| No code in 18mo | `last_active_date` + `career_history` latest role end | Flag/down-rank |
| Title chaser | Median tenure < 1.5 yrs across career | Down-rank |
| Honeypot | `years_of_experience` > (current_year - company_founded_year) on earliest role | Hard drop |
| CV/Speech/Robotics only | Skills + titles contain zero NLP/IR/retrieval signals | Down-rank |
| LangChain-only AI | Only AI exp = recent LangChain + OpenAI wrappers, no embeddings/vector DBs | Down-rank |

> Honeypot check: parse `career_history[0].dates` start year vs company tenure. If a candidate claims 8 years of experience at a company that has existed for 3 years, ground truth relevance = 0. Drop these.

---

## 4. Scoring Methodology

### 4.1 Composite Score Formula

```
S = w1·Technical_Fit
  + w2·Career_Quality
  + w3·Availability_Signal
  + w4·Seniority_Fit
  + w5·Semantic_Similarity
```

All sub-scores normalized to [0, 1] before weighting.

### 4.2 Weight Allocation

| Signal | Weight | Rationale |
|---|---|---|
| `Technical_Fit` | **0.35** | Core JD requirements — embeddings, vector DBs, Python, eval frameworks |
| `Career_Quality` | **0.25** | Product company history, tenure, progression, shipped systems |
| `Availability_Signal` | **0.20** | Behavioral signals — availability, responsiveness, recency |
| `Seniority_Fit` | **0.12** | Years exp, title progression, company stage fit |
| `Semantic_Similarity` | **0.08** | Embedding cosine similarity — catch plain-language strong candidates |

### 4.3 Sub-Score Definitions

#### `Technical_Fit` (0–1)

Check against JD must-haves using skill field + career description text:

| Criterion | Points |
|---|---|
| Embeddings/retrieval in production (sentence-transformers, BGE, E5) | +0.25 |
| Vector DB experience (Pinecone, Milvus, Qdrant, Weaviate, FAISS) | +0.20 |
| Strong Python (proficiency=expert OR skill_assessment score ≥70) | +0.15 |
| Evaluation framework design (NDCG, MRR, MAP, A/B) | +0.15 |
| LLM fine-tuning (LoRA, QLoRA, PEFT) | +0.10 (nice-to-have) |
| Learning-to-rank (XGBoost, neural rankers) | +0.10 (nice-to-have) |
| HR-tech / marketplace domain | +0.05 (bonus) |

Cap at 1.0. Normalize raw points.

**Skill proficiency weight:**
- `expert` = 1.0 · signal strength
- `advanced` = 0.75
- `intermediate` = 0.4
- `beginner` = 0.1

Use `skill_assessment_scores` dict if present — platform-verified trumps self-reported proficiency.

#### `Career_Quality` (0–1)

| Criterion | Points |
|---|---|
| ≥1 product company role (non-consulting) | +0.30 |
| ≥1 role with title: Engineer/ML/Data/Research at company_size ≥ 50 | +0.15 |
| No solo consulting-only history | +0.15 |
| Median tenure ≥ 2.0 yrs | +0.20 |
| Career progression (title seniority increases over time) | +0.20 |

Consulting company list to penalize: `["TCS", "Infosys", "Wipro", "Capgemini", "Cognizant", "Accenture", "HCL", "Mphasis", "Tech Mahindra", "Hexaware"]`

#### `Availability_Signal` (0–1)

Derived entirely from `redrob_signals`:

| Signal | Weight | Direction |
|---|---|---|
| `open_to_work_flag` | 0.25 | True = +1 |
| `last_active_date` recency | 0.20 | < 30d = 1.0, < 90d = 0.6, < 180d = 0.3, older = 0.0 |
| `recruiter_response_rate` | 0.15 | Direct use (0–1) |
| `avg_response_time_hours` | 0.10 | ≤4h = 1.0, ≤24h = 0.7, ≤72h = 0.4, >72h = 0.1 |
| `interview_completion_rate` | 0.15 | Direct use (0–1) |
| `applications_submitted_30d` | 0.10 | ≥3 = 1.0, 1–2 = 0.5, 0 = 0.0 |
| `notice_period_days` | 0.05 | ≤30 = 1.0, ≤60 = 0.7, ≤90 = 0.4, >90 = 0.1 |

> Key insight from dataset docs: "A perfect-on-paper candidate with low response rate + old last_activity is effectively unavailable." Availability weight = 0.20 of total ensures this doesn't bury good candidates, but a zero availability score meaningfully drops rank.

#### `Seniority_Fit` (0–1)

JD target: 6–8 years total, 4–5 in applied ML at product companies.

| Criterion | Score |
|---|---|
| `years_of_experience` 6–9 | 1.0 |
| 4–5 or 10–12 | 0.7 |
| 3 or 13–15 | 0.4 |
| <3 or >15 | 0.1 |

Apply education tier bonus:
- `tier` 1 (IIT/IISc/NIT/BITS) = +0.05
- `tier` 2 = +0.02
- `tier` 3–5 = 0

Cap at 1.0.

#### `Semantic_Similarity` (0–1)

**Purpose:** Catch strong candidates who describe skills in plain language rather than buzzwords.

**Method:**
1. Pre-compute offline: embed all 100k candidates using `all-MiniLM-L6-v2` (CPU-friendly, 80MB model)
2. Embed JD once
3. At ranking time: cosine similarity between candidate embedding and JD embedding
4. Normalize to [0, 1]

**Profile blob for embedding** (concatenate):
```
{headline} {summary} {current_title} {career_history[*].description} {skills[*].name}
```

Do NOT include skills list alone as proxy for capability — include role descriptions.

---

## 5. Honeypot Detection

Hard drop candidates matching any:

```python
def is_honeypot(candidate):
    for role in candidate['career_history']:
        start_year = parse_year(role['dates']['start'])
        current_year = 2025
        if candidate['profile']['years_of_experience'] > (current_year - start_year + 10):
            return True  # Claimed more exp than career allows
    
    # Profile completeness 0 + no verified contact = ghost profile
    signals = candidate['redrob_signals']
    if (signals['profile_completeness_score'] < 5 and 
        not signals['verified_email'] and 
        not signals['verified_phone']):
        return True
    
    return False
```

Target: keep honeypots out of top 100. ≤10 allowed before disqualification.

---

## 6. Implementation Plan

### 6.1 File Structure

```
recruiteriq/
├── precompute.py          # Run once offline — embeds + caches scores
├── rank.py                # Main ranking script (must run < 5 min)
├── signals.py             # All sub-score functions
├── disqualify.py          # Honeypot + hard disqualifier rules
├── config.py              # Weights, thresholds, consulting firm list
├── requirements.txt
├── Dockerfile
├── artifacts/
│   ├── embeddings.npy     # Pre-computed candidate embeddings (cached)
│   ├── jd_embedding.npy   # JD embedding
│   └── scores_cache.pkl   # Pre-computed sub-scores (optional)
└── output/
    └── <participant_id>.csv
```

### 6.2 `precompute.py` (run before submission)

```python
# 1. Load all 100k candidates from candidates.jsonl
# 2. Run disqualifier — mark honeypots
# 3. Build profile blob per candidate
# 4. Batch embed using sentence-transformers (all-MiniLM-L6-v2)
#    - Batch size 512 for CPU efficiency
#    - Estimated time: ~20-40 min on CPU for 100k
# 5. Compute Technical_Fit, Career_Quality, Availability, Seniority per candidate
# 6. Save embeddings.npy + scores_cache.pkl
```

### 6.3 `rank.py` (sandbox execution, < 5 min)

```python
# 1. Load cached embeddings + sub-scores (precomputed)
# 2. Load JD embedding
# 3. Compute semantic similarity (fast cosine on cached vectors)
# 4. Compute composite score S for all candidates
# 5. Sort descending, take top 100
# 6. Generate reasoning string per candidate
# 7. Validate: no score inversion, exactly 100 rows, ranks 1–100
# 8. Write CSV
```

### 6.4 Reasoning String Template

```python
def generate_reasoning(candidate, scores):
    top_skills = get_top_matching_skills(candidate, JD_SKILLS)
    career_highlight = get_strongest_role(candidate)
    availability = "actively looking" if candidate['redrob_signals']['open_to_work_flag'] else "passive"
    
    return (
        f"{candidate['profile']['years_of_experience']}yr ML engineer with production "
        f"{top_skills[0]} + {top_skills[1]} experience at {career_highlight['company']} "
        f"({career_highlight['company_size']} co); {availability}, "
        f"response rate {candidate['redrob_signals']['recruiter_response_rate']:.0%}."
    )
```

---

## 7. Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | CPU-friendly, 80MB, good IR quality |
| Vector similarity | `numpy` cosine (no FAISS needed — single JD query) | No overhead, pure CPU |
| Data parsing | `json` + `pandas` | JSONL native |
| Scoring | Pure Python + numpy | No GPU deps |
| Validation | `validate_submission.py` (provided) | Run before upload |
| Sandbox | Docker (CPU, 16GB, Python 3.10) | Matches Stage 3 constraint |
| Demo | HuggingFace Spaces or Streamlit Cloud | Required by submission rules |

---

## 8. Evaluation Metric Alignment

```
Final = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10
```

| Metric | What it penalizes | How we address it |
|---|---|---|
| NDCG@10 | Wrong candidates in top 10 | Strict disqualifier + high Technical_Fit weight |
| NDCG@50 | Score inversions in top 50 | Continuous scoring (no hard buckets) + monotonic output |
| MAP | Missing relevant candidates | Semantic similarity catches plain-language strong engineers |
| P@10 | Any irrelevant in top 10 | Honeypot detection + consulting filter |

NDCG@10 is 50% of score — **top 10 precision is the win condition.** Technical_Fit (0.35) + Career_Quality (0.25) = 0.60 of composite drives top-10 quality.

---

## 9. Traps & Counter-measures

| Trap | Counter |
|---|---|
| Keyword stuffers | Check role titles + career descriptions, not just skills field |
| Honeypots | Year-of-experience vs earliest company founding date check |
| Behavioral twins | Availability_Signal (0.20) creates meaningful tie-breaking |
| Plain-language engineers | Semantic similarity via embedding catches them |
| Consulting careers | Hard penalize in Career_Quality sub-score |
| LangChain-only "AI" | Check for embeddings/vector DB evidence in career descriptions |
| Over-seniored | Seniority_Fit penalizes 15+ yr candidates (beyond target range) |

---

## 10. Clarifying Questions (Confirm Before Building)

1. Does `candidate_schema.json` define all possible company names, or is consulting firm list custom-built?
2. Are `skill_assessment_scores` available for most candidates, or sparse?
3. Is `career_history[*].description` free text (for embedding) or structured fields?
4. What's the `education.tier` encoding — numeric 1–5 or string labels?
5. Does `dates` in career_history contain structured `{start, end}` or raw string like "Jan 2019 – Mar 2022"?
6. Is the sandbox pre-computation allowed (embeddings cached in Docker image), or must everything run in 5 min from cold start?

---

## 11. Submission Checklist

- [ ] `precompute.py` runs clean on full `candidates.jsonl`
- [ ] `rank.py` completes in < 5 min on CPU, 16GB
- [ ] `validate_submission.py` passes with zero errors
- [ ] Exactly 100 rows, ranks 1–100 each exactly once
- [ ] Scores non-increasing with rank
- [ ] Reasoning strings 1–2 sentences, specific (not generic)
- [ ] No score inversions
- [ ] ≤3 submissions total — do not waste on untested builds
- [ ] Sandbox link live (HuggingFace Spaces / Streamlit / Colab)
- [ ] `submission_metadata_template.yaml` filled
- [ ] AI tool usage declared

---

## 12. Why This Wins

| Judging Criterion | RecruiterIQ Answer |
|---|---|
| **Ranking quality** | Multi-signal composite defeats keyword systems; behavioral signals surface available candidates; honeypot detection prevents disqualification |
| **Methodology clarity** | Explicit weights, documented sub-scores, reproducible in Docker |
| **Explainability** | Per-candidate reasoning string, score breakdown by signal category, interpretable weights |
| **Code reproduction** | Pure Python + numpy + sentence-transformers, no GPU, runs in sandbox |
| **Interview defense** | Every weight choice is JD-grounded and documented here |

---

*RecruiterIQ — Built for India Runs Hackathon · Redrob AI Challenge · 2025*
