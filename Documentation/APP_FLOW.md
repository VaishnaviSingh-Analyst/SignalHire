# App Flow Document
**RecruiterIQ — System & User Flows**

---

## 1. High-Level System Flow

```
                        ┌─────────────────────────────────────┐
                        │         PHASE A: PRECOMPUTATION     │
                        │         (offline, one-time run)     │
                        └─────────────────────────────────────┘

candidates.jsonl ──────► [1. Stream & Parse]
                              │
                              ▼
                         [2. Validate Schema]
                              │
                         ┌────┴────────┐
                         │  honeypot?  │──YES──► disqualified.json (audit)
                         └────┬────────┘
                              │ NO
                              ▼
                         [3. Build Profile Blob]
                              │
                              ▼
                         [4. Batch Embed]  ◄── all-MiniLM-L6-v2 (loaded once)
                         (batch=512, CPU)
                              │
                              ▼
                         [5. Compute Sub-Scores]
                         technical_fit
                         career_quality
                         availability_signal
                         seniority_fit
                              │
                              ▼
                    ┌─────────────────────┐
                    │   ARTIFACTS SAVED   │
                    │  embeddings.npy     │
                    │  candidate_ids.npy  │
                    │  subscores.pkl      │
                    │  jd_embedding.npy   │
                    │  disqualified.json  │
                    └─────────────────────┘


                        ┌─────────────────────────────────────┐
                        │         PHASE B: RANKING            │
                        │  (sandbox: CPU, 16GB, 5min max)     │
                        └─────────────────────────────────────┘

                    ┌─────────────────────┐
                    │   LOAD ARTIFACTS    │
                    └────────┬────────────┘
                             │
                             ▼
                    [6. Cosine Similarity]
                    embeddings × jd_embedding
                    → semantic_sim[100k]
                             │
                             ▼
                    [7. Composite Score]
                    S = weighted_sum(
                      tech, career, avail,
                      seniority, semantic_sim
                    ) for all 100k candidates
                             │
                             ▼
                    [8. Sort Descending]
                    Take top 100
                             │
                             ▼
                    [9. Generate Reasoning]
                    Per candidate: 1–2 sentence string
                             │
                             ▼
                    [10. Validate & Write CSV]
                    validate_submission.py
                             │
                        ┌────┴────┐
                        │ PASS?   │
                        └────┬────┘
                        YES  │  NO
                             │   └──► Fix + re-run (max 3 submissions)
                             ▼
                    <participant_id>.csv  ──► SUBMIT
```

---

## 2. Precomputation Flow (precompute.py)

```
START
  │
  ├─► Load JD from job_description.txt
  │       │
  │       └─► embed_jd() → save jd_embedding.npy
  │
  ├─► Open candidates.jsonl (streaming, 1k chunks)
  │       │
  │       └─► FOR EACH CHUNK:
  │               │
  │               ├─► validate_schema(record)
  │               │       └─► on failure: log + skip (don't crash)
  │               │
  │               ├─► is_honeypot(record)?
  │               │       └─► YES: append to disqualified[], skip
  │               │
  │               ├─► build_profile_blob(record) → blob_str
  │               │
  │               ├─► BATCH ACCUMULATE blobs[]
  │               │       └─► when len(blobs) == 512:
  │               │               embed_batch(blobs) → vectors
  │               │               append to embeddings_list[]
  │               │               flush blobs[]
  │               │
  │               └─► compute_subscores(record)
  │                       → append to subscores_dict{}
  │
  ├─► Flush remaining blobs (last partial batch)
  │
  ├─► Stack embeddings_list → np.array (N, 384)
  │
  ├─► Save:
  │     artifacts/embeddings.npy
  │     artifacts/candidate_ids.npy
  │     artifacts/subscores.pkl
  │     artifacts/disqualified.json
  │
  └─► DONE. Print summary stats.
```

---

## 3. Ranking Flow (rank.py)

```
START
  │
  ├─► Load artifacts (all ≤ 200MB total)
  │     embeddings.npy  → E  (N, 384)
  │     candidate_ids.npy → IDs  (N,)
  │     subscores.pkl   → S_dict  {id: {tech, career, avail, senior}}
  │     jd_embedding.npy → jd_vec  (384,)
  │
  ├─► Compute semantic_sim
  │     sim = E @ jd_vec / (||E|| × ||jd_vec||)
  │     → (N,) float32
  │
  ├─► Compute composite score for all N candidates
  │     FOR i in range(N):
  │       s = S_dict[IDs[i]]
  │       score[i] = (
  │         0.35 * s['tech'] +
  │         0.25 * s['career'] +
  │         0.20 * s['avail'] +
  │         0.12 * s['seniority'] +
  │         0.08 * sim[i]
  │       )
  │
  ├─► Sort indices by score descending
  │     top100_idx = argsort(score)[::-1][:100]
  │
  ├─► For each of top 100:
  │     load candidate record (from jsonl by id)
  │     generate_reasoning(candidate, scores)
  │
  ├─► Enforce monotonic scores
  │     for i in 1..99:
  │       if score[i] > score[i-1]: score[i] = score[i-1]
  │
  ├─► Write CSV
  │     candidate_id, rank, score (3dp), reasoning
  │
  ├─► Run validate_submission.py
  │       └─► FAIL: print errors, exit(1)
  │
  └─► DONE. Print: "Top 100 written. Validation passed."
```

---

## 4. Demo App Flow (Streamlit)

```
USER OPENS APP
      │
      ▼
  Sidebar loads
      │
      ├─► Check artifacts exist?
      │       NO → Show warning banner
      │       YES → Show "✓ Artifacts ready"
      │
      ├─► User selects Mode:
      │       ● Full Pipeline (100k) → requires full artifacts
      │       ● Demo Mode (sample_candidates.json) → always available
      │
      └─► User loads JD (upload or paste)
                │
                ▼
         [▶ Run Ranking] clicked
                │
                ▼
         Show progress bar
         "Scoring candidates…"
                │
                ▼
         rank.py logic runs (in-process)
                │
                ▼
         ┌──────────────────────────┐
         │  Results ready           │
         │  Switch to Tab 1         │
         └──────────────────────────┘
                │
      ┌─────────┼──────────────────────────────┐
      │         │                              │
      ▼         ▼                              ▼
  Tab 1:     Tab 2:                        Tab 3:
  Ranked     Score Explorer               Methodology
  Shortlist  (charts, weight sliders)     (static doc)
      │
      ▼
  User clicks candidate card
      │
      └─► Expand: full career, skills, behavioral signals, score breakdown

  User clicks [⬇ Export CSV]
      │
      └─► Download <participant_id>.csv (pre-validated)
```

---

## 5. State Machine: Candidate Lifecycle

```
          ┌──────────┐
          │  INGESTED │
          └─────┬─────┘
                │
        ┌───────┴────────┐
        │  HONEYPOT?     │
        └───────┬────────┘
         YES    │    NO
          ▼     │     ▼
    DISQUALIFIED  ELIGIBLE
                       │
               ┌───────┴───────┐
               │  SUB-SCORES   │
               │  COMPUTED     │
               └───────┬───────┘
                       │
               ┌───────┴──────────────┐
               │  COMPOSITE SCORED    │
               └───────┬──────────────┘
                       │
              ┌────────┴──────────┐
              │  IN TOP 100?      │
              └────────┬──────────┘
               YES     │     NO
                ▼      │      ▼
           SHORTLISTED       SCORED_OUT
                │
                ▼
         REASONING GENERATED
                │
                ▼
           IN OUTPUT CSV
```

---

## 6. Error Flow

```
ERROR TYPE                   HANDLING
─────────────────────────────────────────────────────
JSONL parse error            Log candidate_id, skip, continue
Missing field                Use default value (not crash)
Honeypot detected            Add to disqualified.json, skip scoring
Embedding batch failure      Retry once, then skip batch + log
Score NaN / Inf              Set to 0.0, log warning
Score inversion in top 100   Clamp to previous score (monotonic enforce)
Validator FAIL               Print exact error lines, exit(1), do not submit
>10 honeypots in top 100     Print DISQUALIFICATION WARNING, do not submit
RAM > 14GB                   Reduce batch size to 256, clear intermediate lists
```

---

*RecruiterIQ App Flow v1.0 · 2025*
