# Implementation Plan
**RecruiterIQ — Build Execution Roadmap**

| Field | Value |
|---|---|
| Total estimated time | 12–16 hours |
| Language | Python 3.10 |
| Submission limit | 3 max — plan for 2 real submissions |
| Critical path | precompute.py → rank.py → validate → submit |

---

## 0. Pre-Build Checklist (Before writing code)

- [ ] Extract `job_description.docx` → `job_description.txt` (python-docx or LibreOffice)
- [ ] Extract `redrob_signals_doc.docx` → read signal definitions
- [ ] Extract `submission_spec.docx` → re-read all constraints
- [ ] Run `validate_submission.py --help` to understand its checks
- [ ] Open `sample_candidates.json` — inspect actual field shapes
- [ ] Open `candidate_schema.json` — check nullability, enum values, date formats
- [ ] Check `sample_submission.csv` — verify exact CSV column names + format

---

## Phase 1 — Environment Setup (1–2 hours)

### Step 1.1: Project Init

```bash
mkdir recruiteriq && cd recruiteriq
python -m venv venv && source venv/bin/activate

pip install sentence-transformers numpy pandas scikit-learn tqdm \
            pyyaml python-docx streamlit

mkdir -p artifacts output docs
```

### Step 1.2: Scaffold Files

```
recruiteriq/
├── config.py
├── disqualify.py
├── signals.py
├── precompute.py
├── rank.py
├── app.py              # Streamlit demo
├── requirements.txt
├── Dockerfile
├── validate_submission.py   # copy from workspace
└── data/
    ├── candidates.jsonl
    ├── candidate_schema.json
    ├── sample_candidates.json
    ├── job_description.txt  # extracted
    └── sample_submission.csv
```

### Step 1.3: config.py

Write all constants first. Nothing else imports correctly without config.

**Verify immediately:** Print weights — must sum to 1.0.

```python
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"
assert abs(sum(AVAILABILITY_WEIGHTS.values()) - 1.0) < 1e-6
```

---

## Phase 2 — Data Layer (2–3 hours)

### Step 2.1: Schema Validator

```python
# Check actual field presence in sample_candidates.json first
import json
with open('data/sample_candidates.json') as f:
    samples = json.load(f)
    
# Print all keys to verify schema matches assumption
for s in samples:
    print(s.keys())
    print(s['redrob_signals'].keys())
    break
```

Fix any field name discrepancies in `config.py` BEFORE writing scoring code.

### Step 2.2: JSONL Streaming Parser

```python
# precompute.py — streaming (never load all 100k at once)
def stream_candidates(jsonl_path: str, chunk_size: int = 1000):
    buffer = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                candidate = json.loads(line.strip())
                buffer.append(candidate)
                if len(buffer) >= chunk_size:
                    yield buffer
                    buffer = []
            except json.JSONDecodeError as e:
                log.warning(f"Parse error: {e}")
    if buffer:
        yield buffer
```

### Step 2.3: Date Parser (critical for honeypot detection)

Date strings in career_history are NOT guaranteed ISO format. Write defensive parser:

```python
def parse_year(date_str: str) -> Optional[int]:
    """Extract year from various date formats:
    'Jan 2019', '2019', '2019-01', 'January 2019', 'Present'
    """
    if not date_str or date_str == "Present":
        return None
    # Try regex: find 4-digit year
    import re
    match = re.search(r'\b(19|20)\d{2}\b', str(date_str))
    return int(match.group()) if match else None
```

**Test on all 10 sample candidates before proceeding.**

---

## Phase 3 — Disqualification Engine (1–2 hours)

### Step 3.1: Honeypot Detector

```python
# disqualify.py
def is_honeypot(candidate: dict) -> tuple[bool, str]:
    yoe = candidate['profile'].get('years_of_experience', 0)
    career = candidate.get('career_history', [])
    
    if career:
        start_years = [parse_year(r['dates']['start']) for r in career]
        start_years = [y for y in start_years if y is not None]
        if start_years:
            earliest = min(start_years)
            if yoe > (2025 - earliest + HONEYPOT_YOE_TOLERANCE):
                return True, f"Claims {yoe}yr YoE but career starts {earliest} ({2025-earliest}yr ago)"
    
    # Ghost profile check
    s = candidate.get('redrob_signals', {})
    if (s.get('profile_completeness_score', 100) < HONEYPOT_COMPLETENESS_MAX and
        not s.get('verified_email', True) and
        not s.get('verified_phone', True)):
        return True, "Ghost profile: <5% complete, no verified contacts"
    
    return False, ""
```

### Step 3.2: Consulting Penalty

```python
def should_hard_penalize(candidate: dict) -> tuple[float, str]:
    companies = [r.get('company', '') for r in candidate.get('career_history', [])]
    consulting_count = sum(
        1 for c in companies
        if any(firm.lower() in c.lower() for firm in CONSULTING_FIRMS)
    )
    if consulting_count == len(companies) and len(companies) > 0:
        return CONSULTING_PENALTY, "All-consulting career, no product company"
    return 1.0, ""
```

**Unit test:** Create fake candidates matching each rule. Verify detection rate.

---

## Phase 4 — Scoring Engine (3–4 hours)

### Step 4.1: Technical Fit

This is the highest-weight signal (0.35). Get it right.

```python
# signals.py
def compute_technical_fit(candidate: dict) -> float:
    score = 0.0
    
    # Build skill lookup: {normalized_name: proficiency_weight}
    skill_weights = {"beginner": 0.1, "intermediate": 0.4, "advanced": 0.75, "expert": 1.0}
    skills = {}
    for skill in candidate.get('skills', []):
        name = skill.get('name', '').lower()
        prof = skill_weights.get(skill.get('proficiency', 'beginner'), 0.1)
        skills[name] = prof
    
    # Override with platform assessment scores if available
    assessments = candidate.get('redrob_signals', {}).get('skill_assessment_scores', {})
    for skill_name, assessed_score in assessments.items():
        normalized = skill_name.lower()
        if normalized in skills or any(kw in normalized for kw in ['python', 'ml', 'nlp']):
            skills[normalized] = assessed_score / 100.0
    
    # Score against JD must-haves
    for criterion, cfg in JD_MUST_HAVES.items():
        best_match = 0.0
        for kw in cfg['keywords']:
            for skill_name, skill_strength in skills.items():
                if kw.lower() in skill_name:
                    best_match = max(best_match, skill_strength)
        score += best_match * cfg['weight']
    
    # Score against career descriptions (production evidence)
    career_text = ' '.join(
        r.get('description', '') for r in candidate.get('career_history', [])
    ).lower()
    
    production_bonus = sum(1 for kw in PRODUCTION_SIGNALS if kw.lower() in career_text)
    retrieval_bonus = sum(1 for kw in RETRIEVAL_SIGNALS if kw.lower() in career_text)
    
    score += min(production_bonus * 0.02, 0.10)   # max +0.10 bonus
    score += min(retrieval_bonus * 0.02, 0.10)    # max +0.10 bonus
    
    # Nice-to-haves (smaller contribution)
    for criterion, cfg in JD_NICE_TO_HAVES.items():
        for kw in cfg['keywords']:
            if any(kw.lower() in s for s in skills.keys()) or kw.lower() in career_text:
                score += cfg['weight'] * 0.5
                break
    
    return min(score, 1.0)
```

### Step 4.2: Career Quality

```python
def compute_career_quality(candidate: dict) -> float:
    score = 0.0
    career = candidate.get('career_history', [])
    
    if not career:
        return 0.0
    
    companies = [r.get('company', '') for r in career]
    
    # Product company presence
    non_consulting = [c for c in companies if not any(f.lower() in c.lower() for f in CONSULTING_FIRMS)]
    if non_consulting:
        score += 0.30
    
    # Meaningful seniority in history
    senior_titles = ['senior', 'lead', 'principal', 'staff', 'head', 'director']
    ml_titles = ['ml', 'machine learning', 'ai', 'data scientist', 'nlp', 'research']
    for role in career:
        title_lower = role.get('title', '').lower()
        if any(t in title_lower for t in ml_titles):
            score += 0.15
            break
    
    # Tenure quality (median tenure across roles)
    durations = []
    for role in career:
        start_yr = parse_year(role['dates']['start'])
        end_str = role['dates'].get('end', 'Present')
        end_yr = 2025 if end_str == 'Present' else parse_year(end_str)
        if start_yr and end_yr:
            durations.append(end_yr - start_yr)
    
    if durations:
        median_tenure = sorted(durations)[len(durations)//2]
        if median_tenure >= 3:
            score += 0.20
        elif median_tenure >= 2:
            score += 0.12
        elif median_tenure >= 1.5:
            score += 0.05
    
    # Career progression (last title > first title seniority)
    score += 0.20 if _has_upward_progression(career) else 0.0
    
    # No consulting penalty already handled via penalty_multiplier
    if not non_consulting:
        score = 0.0  # redundant safety (penalty_multiplier handles it)
    
    return min(score, 1.0)
```

### Step 4.3: Availability Signal

```python
def compute_availability_signal(candidate: dict) -> float:
    s = candidate.get('redrob_signals', {})
    score = 0.0
    
    # open_to_work (0.25)
    score += 0.25 if s.get('open_to_work_flag', False) else 0.0
    
    # last_active_recency (0.20)
    from datetime import date
    last_active = s.get('last_active_date')
    if last_active:
        try:
            delta = (date.today() - date.fromisoformat(last_active[:10])).days
            if delta <= 30:   score += 0.20
            elif delta <= 90: score += 0.12
            elif delta <= 180:score += 0.06
        except: pass
    
    # recruiter_response_rate (0.15)
    score += s.get('recruiter_response_rate', 0.0) * 0.15
    
    # response_speed (0.10)
    hours = s.get('avg_response_time_hours', 999)
    if hours <= 4:    score += 0.10
    elif hours <= 24: score += 0.07
    elif hours <= 72: score += 0.04
    else:             score += 0.01
    
    # interview_completion_rate (0.15)
    score += s.get('interview_completion_rate', 0.0) * 0.15
    
    # applications_submitted_30d (0.10)
    apps = s.get('applications_submitted_30d', 0)
    if apps >= 3:   score += 0.10
    elif apps >= 1: score += 0.05
    
    # notice_period (0.05)
    notice = s.get('notice_period_days', 90)
    if notice <= 30:  score += 0.05
    elif notice <= 60:score += 0.035
    elif notice <= 90:score += 0.02
    else:             score += 0.005
    
    return min(score, 1.0)
```

### Step 4.4: Seniority Fit

```python
def compute_seniority_fit(candidate: dict) -> float:
    yoe = candidate['profile'].get('years_of_experience', 0)
    
    if IDEAL_YOE_MIN <= yoe <= IDEAL_YOE_MAX:    base = 1.0
    elif 4 <= yoe < 6 or 9 < yoe <= 12:          base = 0.7
    elif yoe == 3 or 12 < yoe <= 15:             base = 0.4
    else:                                         base = 0.1
    
    # Education tier bonus
    edu_tier = min((e.get('tier', 5) for e in candidate.get('education', [])), default=5)
    tier_bonus = {1: 0.05, 2: 0.02}.get(edu_tier, 0.0)
    
    return min(base + tier_bonus, 1.0)
```

**Unit tests for all 4 scorers before running precompute on full dataset.**

---

## Phase 5 — Precomputation (1–2 hours)

### Step 5.1: Embedding Setup

```python
# precompute.py
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(EMBEDDING_MODEL)
# Warm up
_ = model.encode(["warm up"], batch_size=1)
```

### Step 5.2: Full Run on Sample First

```bash
# Test on 10 samples before 100k
python precompute.py --input data/sample_candidates.json --output artifacts_test/
```

Check:
- `artifacts_test/embeddings.npy` shape = (10, 384)
- `artifacts_test/subscores.pkl` has 10 entries
- No crashes, no NaN scores

### Step 5.3: Full Run on 100k

```bash
time python precompute.py --input data/candidates.jsonl --output artifacts/
```

Monitor:
- RAM usage: `watch -n5 free -h`
- Expected runtime: 30–60 min
- Expected peak RAM: ~2GB

---

## Phase 6 — Ranking Script (1–2 hours)

### Step 6.1: rank.py Core

```python
import numpy as np, pickle, json, csv
from config import WEIGHTS, TOP_K, SCORE_DECIMAL_PLACES

def run_ranking(artifacts_dir, jd_embedding_path, output_path, jsonl_path):
    # Load
    E = np.load(f"{artifacts_dir}/embeddings.npy")          # (N, 384)
    IDs = np.load(f"{artifacts_dir}/candidate_ids.npy")     # (N,)
    with open(f"{artifacts_dir}/subscores.pkl", "rb") as f:
        subscores = pickle.load(f)
    jd_vec = np.load(jd_embedding_path)                     # (384,)
    
    # Semantic similarity (vectorized, fast)
    sim = E @ jd_vec   # already unit-normalized, so this = cosine sim
    
    # Composite score
    scores = np.zeros(len(IDs))
    for i, cid in enumerate(IDs):
        s = subscores.get(cid, {})
        penalty = s.get('penalty_multiplier', 1.0)
        scores[i] = penalty * (
            WEIGHTS['technical_fit']       * s.get('technical_fit', 0) +
            WEIGHTS['career_quality']      * s.get('career_quality', 0) +
            WEIGHTS['availability_signal'] * s.get('availability_signal', 0) +
            WEIGHTS['seniority_fit']       * s.get('seniority_fit', 0) +
            WEIGHTS['semantic_similarity'] * float(sim[i])
        )
    
    # Top 100
    top_idx = np.argsort(scores)[::-1][:TOP_K]
    
    # Monotonic enforcement
    for i in range(1, TOP_K):
        if scores[top_idx[i]] > scores[top_idx[i-1]]:
            scores[top_idx[i]] = scores[top_idx[i-1]]
    
    # Build candidate lookup for reasoning
    candidates = load_candidates_by_ids(jsonl_path, set(IDs[top_idx]))
    
    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for rank, idx in enumerate(top_idx, 1):
            cid = IDs[idx]
            score = round(float(scores[idx]), SCORE_DECIMAL_PLACES)
            candidate = candidates.get(cid, {})
            reasoning = generate_reasoning(candidate, subscores.get(cid, {}))
            writer.writerow([cid, rank, score, reasoning])
    
    print(f"✓ Top {TOP_K} written to {output_path}")
```

### Step 6.2: Reasoning Generator

```python
def generate_reasoning(candidate: dict, scores: dict) -> str:
    p = candidate.get('profile', {})
    s = candidate.get('redrob_signals', {})
    
    yoe = p.get('years_of_experience', '?')
    top_skills = scores.get('top_skills', ['ML', 'Python'])
    company = scores.get('strongest_company', p.get('current_company', 'a product company'))
    open_work = "actively looking" if s.get('open_to_work_flag') else "passive"
    rr = s.get('recruiter_response_rate', 0)
    notice = s.get('notice_period_days', 90)
    
    skills_str = ' + '.join(top_skills[:2]) if top_skills else 'ML systems'
    
    reasoning = (
        f"{yoe}yr ML engineer with production {skills_str} experience at {company}; "
        f"{open_work}, {rr:.0%} response rate, {notice}d notice."
    )
    
    # Trim to 300 chars max
    return reasoning[:300]
```

### Step 6.3: Timing Test

```bash
time python rank.py
# Must be < 300 seconds
```

If slow: check that semantic similarity uses vectorized `@` not a loop.

---

## Phase 7 — Validation & First Submission (1 hour)

### Step 7.1: Run Validator

```bash
python validate_submission.py output/<participant_id>.csv
```

Common failures to check:
- Score not non-increasing (monotonic bug)
- Reasoning contains newlines (strip them)
- Score has > 3 decimal places
- Duplicate ranks
- candidate_id format wrong

### Step 7.2: Honeypot Audit

```bash
python -c "
import json
disq = json.load(open('artifacts/disqualified.json'))
honeypots = [d for d in disq if d['disqualify_type'] == 'HONEYPOT']
print(f'Total honeypots detected: {len(honeypots)}')
"
```

Check: top 100 CSV should contain ≤10 honeypot IDs. Cross-reference if needed.

### Step 7.3: First Submission

- [ ] Validator passes with 0 errors
- [ ] Sandbox URL is live and accessible
- [ ] `submission_metadata_template.yaml` filled
- [ ] File named `<participant_id>.csv`
- [ ] Submission #1 of 3

---

## Phase 8 — Demo Sandbox (2 hours)

### Step 8.1: Streamlit App Skeleton

```python
# app.py
import streamlit as st
import json, numpy as np, pandas as pd

st.set_page_config(page_title="RecruiterIQ", layout="wide")

# Sidebar
with st.sidebar:
    st.title("RecruiterIQ")
    mode = st.radio("Mode", ["Demo (10 candidates)", "Full Pipeline"])
    jd_text = st.text_area("Job Description", height=200)
    run_btn = st.button("▶ Run Ranking", type="primary")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏆 Shortlist", "📊 Score Explorer", "📖 Methodology", "🚫 Disqualified"])

if run_btn:
    if not jd_text:
        st.warning("Load a job description to begin ranking")
    else:
        with st.spinner("Scoring candidates..."):
            results = run_demo_ranking(jd_text, mode)
        
        with tab1:
            render_shortlist(results)
        
        with tab2:
            render_score_explorer(results)
```

### Step 8.2: Deploy to HuggingFace Spaces

```
# Space requirements.txt — must be CPU-only
sentence-transformers==2.7.0
numpy==1.26.4
streamlit==1.35.0
```

Push code + `artifacts/` (embeddings + subscores) to Space.
Verify it loads in < 30 seconds.

---

## Phase 9 — Second Submission (if needed)

After Phase 8: review any obvious mistakes in top-10 reasoning strings or scores.

Possible improvements before submission 2:
- Increase CONSULTING_PENALTY from 0.15 → 0.05 (more nuanced)
- Add `github_activity_score` bonus (+0.02 if score > 70) to technical_fit
- Tune `IDEAL_YOE_MIN/MAX` based on what top candidates show
- Improve reasoning strings to be more specific

**Only submit 2 if submission 1 had clear errors. Don't burn submission 3 on guesses.**

---

## Risk Register

| Risk | Probability | Mitigation |
|---|---|---|
| JSONL parsing crashes on malformed lines | Medium | Try/except per line, log + skip |
| Embedding RAM exceeds 14GB | Low | Streaming + float32 already planned |
| rank.py > 5 min | Low | Vectorized cosine, no Python loops |
| >10 honeypots in top 100 | Low | Honeypot filter runs pre-scoring |
| Validator fails on CSV | Medium | Run validator after every test run |
| Date parsing fails silently | High | Unit test parse_year on all samples first |
| Wrong consulting firm names in dataset | Medium | Log all company names, spot-check |

---

## Time Budget Summary

| Phase | Task | Hours |
|---|---|---|
| 0 | Pre-build checklist | 0.5 |
| 1 | Environment + scaffold | 1.5 |
| 2 | Data layer + parsers | 2.0 |
| 3 | Disqualification engine | 1.5 |
| 4 | Scoring engine (all 4 signals) | 3.5 |
| 5 | Precomputation (full 100k) | 1.5 |
| 6 | rank.py + reasoning | 1.5 |
| 7 | Validate + submit #1 | 1.0 |
| 8 | Streamlit sandbox | 2.0 |
| 9 | Review + submit #2 | 1.0 |
| **Total** | | **~16 hrs** |

**Critical path:** Phase 4 (scoring) is where quality is won or lost. Don't rush it.

---

*RecruiterIQ Implementation Plan v1.0 · 2025*
