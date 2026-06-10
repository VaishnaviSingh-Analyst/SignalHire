# Backend Schema Document
**RecruiterIQ — Data Models, Artifacts & Interfaces**

---

## 1. Input Schema

### 1.1 Candidate Record (from `candidates.jsonl`)

One JSON object per line. Full schema per `candidate_schema.json`:

```python
class CandidateRecord:
    candidate_id: str          # "CAND_0042871"  (7-digit, unique)

    profile: Profile
    career_history: List[Role]   # 1–10 items
    education: List[Education]   # 0–5 items
    skills: List[Skill]          # variable
    certifications: List[Cert]   # variable
    languages: List[Language]    # variable
    redrob_signals: RedrobSignals
```

```python
class Profile:
    name: str                    # anonymized
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: int     # self-reported
    current_title: str
    current_company: str
    company_size: str            # "1-10" | "11-50" | "51-200" | "201-500" | "501-1000" | "1000+"
    industry: str
```

```python
class Role:
    company: str
    title: str
    dates: DateRange             # {start: str, end: str | "Present"}
    duration: str                # human string e.g. "2 years 3 months"
    is_current: bool
    industry: str
    company_size: str
    description: str             # free text — KEY for production evidence
```

```python
class DateRange:
    start: str                   # "Jan 2019" or "2019" or ISO
    end: str                     # "Mar 2022" | "Present"
```

```python
class Education:
    institution: str
    degree: str
    field: str
    years: DateRange
    grade: str                   # "9.1 CGPA" | "First Class" | etc.
    tier: int                    # 1–5 (1=IIT/IISc, 5=unknown)
```

```python
class Skill:
    name: str
    proficiency: str             # "beginner" | "intermediate" | "advanced" | "expert"
    endorsements: int
    duration_months: int
```

```python
class RedrobSignals:
    profile_completeness_score: int          # 0–100
    signup_date: str                         # ISO date
    last_active_date: str                    # ISO date
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float           # 0.0–1.0
    avg_response_time_hours: float
    skill_assessment_scores: Dict[str, int]  # {"Python": 88, "ML": 74}
    connection_count: int
    endorsements_received: int
    notice_period_days: int                  # 0–180
    expected_salary_range_inr_lpa: SalaryRange
    preferred_work_mode: str                 # "remote"|"hybrid"|"onsite"|"flexible"
    willing_to_relocate: bool
    github_activity_score: int               # -1 (not linked) or 0–100
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float         # 0.0–1.0
    offer_acceptance_rate: float             # -1.0 (no history) or 0.0–1.0
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool
```

```python
class SalaryRange:
    min: float    # INR LPA
    max: float    # INR LPA
```

---

## 2. Internal Data Models

### 2.1 CandidateScores (computed, stored in `subscores.pkl`)

```python
class CandidateScores:
    candidate_id: str

    # Sub-scores (all float, [0.0, 1.0])
    technical_fit: float
    career_quality: float
    availability_signal: float
    seniority_fit: float

    # Derived metadata (for reasoning generation)
    top_skills: List[str]           # top 3 matching JD skills
    strongest_company: str          # highest-signal company in career
    strongest_role_title: str
    is_open_to_work: bool
    response_rate: float
    years_of_experience: int
    penalty_multiplier: float       # 1.0 = no penalty, 0.15 = consulting-only

    # Flags
    disqualified: bool
    disqualify_reason: Optional[str]
    is_honeypot: bool
```

### 2.2 RankedCandidate (runtime, top 100 output)

```python
class RankedCandidate:
    rank: int                       # 1–100
    candidate_id: str
    composite_score: float          # [0.0, 1.0], 3 decimal places
    semantic_similarity: float      # cosine vs JD embedding
    scores: CandidateScores
    reasoning: str                  # 1–2 sentences, 120–300 chars
```

### 2.3 DisqualifiedRecord (audit log)

```python
class DisqualifiedRecord:
    candidate_id: str
    disqualify_type: str    # "HONEYPOT" | "HARD_PENALTY" | "GHOST_PROFILE"
    reason: str
    years_of_experience: int
    earliest_role_year: Optional[int]
    profile_completeness: int
```

---

## 3. Artifact Schemas

### 3.1 `artifacts/embeddings.npy`
```
dtype:  float32
shape:  (N, 384)          # N = candidates passing disqualification
notes:  Row i corresponds to candidate_ids[i]
        Normalized to unit vectors (for fast cosine via dot product)
```

### 3.2 `artifacts/candidate_ids.npy`
```
dtype:  str (Unicode)
shape:  (N,)
notes:  Aligned index with embeddings.npy
        candidate_ids[i] → embeddings[i]
```

### 3.3 `artifacts/jd_embedding.npy`
```
dtype:  float32
shape:  (384,)
notes:  Unit-normalized
        Embedded from: full JD text (title + requirements + description)
```

### 3.4 `artifacts/subscores.pkl`
```
type:   Python dict, pickled
schema: {
    "CAND_0042871": {
        "technical_fit": 0.91,
        "career_quality": 0.88,
        "availability_signal": 0.96,
        "seniority_fit": 0.82,
        "penalty_multiplier": 1.0,
        "top_skills": ["FAISS", "Sentence-Transformers", "Python"],
        "strongest_company": "Swiggy",
        "strongest_role_title": "Senior ML Engineer",
        "is_open_to_work": true,
        "response_rate": 0.94,
        "years_of_experience": 7,
        "disqualified": false,
        "disqualify_reason": null,
        "is_honeypot": false
    },
    ...
}
```

### 3.5 `artifacts/disqualified.json`
```json
[
    {
        "candidate_id": "CAND_0000041",
        "disqualify_type": "HONEYPOT",
        "reason": "Claims 8yr YoE but earliest role starts 2022 (3yr ago)",
        "years_of_experience": 8,
        "earliest_role_year": 2022,
        "profile_completeness": 34
    },
    ...
]
```

---

## 4. Output Schema

### 4.1 Submission CSV

```
Columns:    candidate_id, rank, score, reasoning
Types:      str,          int,  float, str
Constraints:
  - Exactly 100 data rows
  - rank: integers 1–100, each exactly once
  - score: float, 3 decimal places, non-increasing with rank
  - reasoning: 1–2 sentences, no newlines, UTF-8
  - candidate_id: format "CAND_XXXXXXX" (7 digits)
```

Example rows:
```csv
candidate_id,rank,score,reasoning
CAND_0042871,1,0.899,"7yr ML engineer with production FAISS + Qdrant at Swiggy; actively looking, 94% response rate, 30-day notice."
CAND_0019234,2,0.887,"6yr applied ML at Meesho and Zomato, strong embedding retrieval + eval frameworks; open to work, last active 2 days ago."
```

### 4.2 Submission Metadata YAML

```yaml
participant_id: "PART_XXXX"
submission_number: 1          # 1, 2, or 3 (max 3)
model_type: "hybrid"          # keyword | embedding | hybrid | llm
ai_tools_used:
  - "sentence-transformers (all-MiniLM-L6-v2)"
  - "numpy"
  - "Claude (architecture design + code review)"
precomputed_artifacts: true
artifact_files:
  - "artifacts/embeddings.npy"
  - "artifacts/subscores.pkl"
  - "artifacts/jd_embedding.npy"
sandbox_url: "https://huggingface.co/spaces/<username>/recruiteriq"
notes: "Hybrid offline scoring. Precomputed embeddings + sub-scores. Ranking step runs in <5 min CPU."
```

---

## 5. Config Schema (`config.py`)

```python
# Score weights (must sum to 1.0)
WEIGHTS = {
    "technical_fit":      0.35,
    "career_quality":     0.25,
    "availability_signal":0.20,
    "seniority_fit":      0.12,
    "semantic_similarity":0.08,
}

# Technical fit: JD skill mapping
JD_MUST_HAVES = {
    "embeddings":   {"keywords": ["sentence-transformers", "BGE", "E5", "embeddings", "dense retrieval"], "weight": 0.25},
    "vector_db":    {"keywords": ["FAISS", "Pinecone", "Milvus", "Qdrant", "Weaviate", "OpenSearch"], "weight": 0.20},
    "python":       {"keywords": ["Python"], "weight": 0.15},
    "eval_frameworks": {"keywords": ["NDCG", "MRR", "MAP", "A/B test", "evaluation", "ranking"], "weight": 0.15},
}
JD_NICE_TO_HAVES = {
    "llm_finetuning": {"keywords": ["LoRA", "QLoRA", "PEFT", "fine-tuning", "fine tuning"], "weight": 0.10},
    "ltr":            {"keywords": ["learning-to-rank", "LTR", "XGBoost", "LambdaMART"], "weight": 0.10},
    "hrtech":         {"keywords": ["HR tech", "talent", "ATS", "recruitment", "marketplace"], "weight": 0.05},
}

# Consulting firm hard-penalty list
CONSULTING_FIRMS = [
    "TCS", "Infosys", "Wipro", "Capgemini", "Cognizant",
    "Accenture", "HCL", "Mphasis", "Tech Mahindra", "Hexaware",
    "IBM GBS", "Deloitte", "EY", "KPMG", "Mindtree", "Persistent"
]
CONSULTING_PENALTY = 0.15    # multiplier if all-consulting career

# Seniority fit: ideal range
IDEAL_YOE_MIN = 6
IDEAL_YOE_MAX = 9

# Availability signal weights
AVAILABILITY_WEIGHTS = {
    "open_to_work":           0.25,
    "last_active_recency":    0.20,
    "recruiter_response_rate":0.15,
    "response_speed":         0.10,
    "interview_completion":   0.15,
    "applications_30d":       0.10,
    "notice_period":          0.05,
}

# Embedding
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 512
EMBEDDING_DIM = 384
MAX_PROFILE_BLOB_CHARS = 1024

# Honeypot detection
HONEYPOT_YOE_TOLERANCE = 2      # years
HONEYPOT_COMPLETENESS_MAX = 5   # % completeness threshold

# Output
TOP_K = 100
SEED = 42
SCORE_DECIMAL_PLACES = 3
```

---

## 6. Function Signatures

```python
# disqualify.py
def is_honeypot(candidate: dict) -> bool
def should_hard_penalize(candidate: dict) -> float   # returns penalty multiplier

# signals.py
def compute_technical_fit(candidate: dict) -> float
def compute_career_quality(candidate: dict) -> float
def compute_availability_signal(candidate: dict) -> float
def compute_seniority_fit(candidate: dict) -> float
def compute_subscores(candidate: dict) -> CandidateScores

# precompute.py
def build_profile_blob(candidate: dict) -> str
def embed_batch(blobs: List[str], model) -> np.ndarray    # (N, 384)
def run_precomputation(jsonl_path: str, jd_text: str) -> None

# rank.py
def compute_composite(scores: CandidateScores, sim: float) -> float
def generate_reasoning(candidate: dict, scores: CandidateScores) -> str
def enforce_monotonic(scores: np.ndarray) -> np.ndarray
def run_ranking(artifacts_dir: str, output_path: str) -> None

# validate.py (wrapper around provided script)
def validate_csv(csv_path: str) -> bool
```

---

*RecruiterIQ Backend Schema v1.0 · 2025*
