# UI/UX Design Document
**RecruiterIQ — Demo Sandbox Interface**

| Field | Value |
|---|---|
| Interface | Streamlit app (HuggingFace Spaces deployment) |
| Audience | Hackathon judges + technical reviewers |
| Primary job | Show the ranking system working, explain every score |
| Platform | Web browser (desktop primary), CPU backend |

---

## 1. Design Principles

**Signal over decoration.** Judges are engineers. Every visual element must carry information — no decorative gradients, no hero banners. The interface is a scoring dashboard, not a marketing page.

**Explainability is the product.** The ranking is only as trustworthy as its explanation. Score breakdown bars per candidate must be more prominent than the composite score.

**Speed of comprehension.** Judge scans top-10 cards in under 30 seconds. Rank + name + headline + top reason = one glance. Deeper breakdown = one click/expand.

---

## 2. Color & Typography

### Palette
```
Background:    #0F1117   (Streamlit dark default)
Surface:       #1A1D27   (cards, panels)
Surface-alt:   #232636   (alternate rows, inputs)
Accent-blue:   #4F8EF7   (primary CTA, rank badges, links)
Accent-green:  #3ECF8E   (availability signals, positive indicators)
Accent-amber:  #F59E0B   (warnings, medium confidence)
Accent-red:    #EF4444   (disqualifiers, low scores, honeypot flags)
Text-primary:  #F1F5F9
Text-muted:    #94A3B8
Border:        #2D3148
```

### Type
- **Rank badge + score**: `font-weight: 800, font-size: 2rem, font-family: monospace`
- **Candidate name/title**: `font-weight: 600, font-size: 1.1rem`
- **Reasoning text**: `font-size: 0.9rem, color: text-muted`
- **Score labels**: `font-size: 0.75rem, uppercase, letter-spacing: 0.08em`

---

## 3. App Structure (Page Map)

```
RecruiterIQ
├── Sidebar
│   ├── Mode toggle: [Full Pipeline | Demo Mode]
│   ├── JD input: text area or file upload
│   ├── Candidate input: JSON paste or file upload (demo mode)
│   ├── Run button
│   └── System status (artifacts loaded, model ready)
│
├── Main Area
│   ├── [Tab 1] Ranked Shortlist     ← default view
│   ├── [Tab 2] Score Explorer        ← signal breakdown chart
│   ├── [Tab 3] Methodology           ← weights, formula, decisions
│   └── [Tab 4] Disqualified          ← audit log
```

---

## 4. Screen Designs

### 4.1 Ranked Shortlist Tab

```
┌─────────────────────────────────────────────────────────────┐
│  🏆  Top 100 Candidates  ·  Ranked for: Senior AI Engineer  │
│  Showing 1–10 of 100  [▼ Filter by confidence]  [⬇ Export] │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  #1     ████████████████████████████████  92.4      │   │
│  │         Candidate: CAND_0042871                     │   │
│  │         Senior ML Engineer · Bangalore              │   │
│  │         7yr · Swiggy → Flipkart · open to work ✓   │   │
│  │                                                     │   │
│  │  Technical  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░  91%                 │   │
│  │  Career     ▓▓▓▓▓▓▓▓▓▓▓▓▓░░░  88%                 │   │
│  │  Available  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  96%                 │   │
│  │  Seniority  ▓▓▓▓▓▓▓▓▓▓▓▓░░░░  82%                 │   │
│  │  Semantic   ▓▓▓▓▓▓▓▓▓▓▓▓▓░░░  87%                 │   │
│  │                                                     │   │
│  │  "7yr ML engineer with production FAISS + Qdrant   │   │
│  │  experience at Swiggy; actively looking,           │   │
│  │  response rate 94%, notice 30d."                   │   │
│  │                                      [▼ Expand]    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  #2     ████████████████████████████░░  89.1        │   │
│  │  ...                                                │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Expanded card state** (on click):
```
┌─────────────────────────────────────────────────────────────┐
│  #1  CAND_0042871  ·  Composite: 92.4                      │
├─────────────────────────────────────────────────────────────┤
│  CAREER HISTORY                                             │
│  2023–now   Senior ML Engineer · Swiggy (Series D)         │
│  2019–2023  ML Engineer · Flipkart                         │
│  2017–2019  Data Scientist · Mu Sigma                      │
│                                                             │
│  TOP SKILLS (verified)                                      │
│  ● Python [expert] [platform score: 88]                    │
│  ● Sentence-Transformers [advanced]                        │
│  ● Qdrant [advanced]                                       │
│  ● FAISS [intermediate]                                    │
│                                                             │
│  BEHAVIORAL SIGNALS                                         │
│  Last active: 3 days ago      Response rate: 94%           │
│  Open to work: Yes            Notice: 30 days              │
│  GitHub score: 72             Interview completion: 100%   │
│                                                             │
│  RAW SCORE BREAKDOWN                                        │
│  Technical Fit    × 0.35 =  0.91 × 0.35 = 0.319          │
│  Career Quality   × 0.25 =  0.88 × 0.25 = 0.220          │
│  Availability     × 0.20 =  0.96 × 0.20 = 0.192          │
│  Seniority Fit    × 0.12 =  0.82 × 0.12 = 0.098          │
│  Semantic Sim     × 0.08 =  0.87 × 0.08 = 0.070          │
│  ─────────────────────────────────────────────            │
│  COMPOSITE SCORE             =        0.899 → 89.9        │
└─────────────────────────────────────────────────────────────┘
```

---

### 4.2 Score Explorer Tab

```
┌─────────────────────────────────────────────────────────────┐
│  Score Distribution                                         │
│                                                             │
│  ┌──────────────────────────────────────────┐              │
│  │  Composite Score Histogram (top 1000)    │              │
│  │  Bar chart: x=score bucket, y=count      │              │
│  │  Highlight: top 100 in accent-blue       │              │
│  └──────────────────────────────────────────┘              │
│                                                             │
│  ┌──────────────────────────────────────────┐              │
│  │  Signal Correlation Scatter              │              │
│  │  x = Technical Fit, y = Career Quality  │              │
│  │  color = Availability (green→red)        │              │
│  │  size = Composite score                  │              │
│  └──────────────────────────────────────────┘              │
│                                                             │
│  Weight Sensitivity Slider                                  │
│  Technical  ──●──────────  0.35                           │
│  Career     ────●────────  0.25                           │
│  Available  ──────●──────  0.20                           │
│  [Recalculate ranking]  ← shows how top-10 shifts         │
└─────────────────────────────────────────────────────────────┘
```

---

### 4.3 Methodology Tab

```
┌─────────────────────────────────────────────────────────────┐
│  How RecruiterIQ Ranks                                      │
├─────────────────────────────────────────────────────────────┤
│  FORMULA                                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  S = 0.35·Tech + 0.25·Career + 0.20·Avail           │  │
│  │      + 0.12·Seniority + 0.08·Semantic                │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  PIPELINE                                                   │
│  [Ingest 100k] → [Disqualify] → [Load Artifacts]          │
│       → [Score] → [Rank] → [Generate Reasoning] → [CSV]   │
│                                                             │
│  DISQUALIFIERS (5 hard rules)                              │
│  ✗ All-consulting career (no product company)              │
│  ✗ Honeypot (impossible experience timeline)               │
│  ✗ Ghost profile (completeness < 5, no verified contact)   │
│  ✗ Pure research (no deployment evidence)                  │
│  ✗ No-code recent (title + activity signals)               │
│                                                             │
│  [View full methodology doc ↗]                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 4.4 Disqualified Tab

```
┌─────────────────────────────────────────────────────────────┐
│  Disqualified Candidates Audit Log                          │
│  Total disqualified: 1,842  ·  Honeypots detected: 78      │
├─────────────────────────────────────────────────────────────┤
│  candidate_id    reason                  disqualify_type    │
│  CAND_0000041    YoE > career timeline   HONEYPOT           │
│  CAND_0002197    All consulting career   HARD_PENALTY       │
│  CAND_0003841    Ghost profile           HONEYPOT           │
│  ...                                                        │
│                                                             │
│  [⬇ Export audit log]                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Sidebar Design

```
┌──────────────────────┐
│  RecruiterIQ         │
│  ───────────────     │
│  Mode:               │
│  ● Full Pipeline     │
│  ○ Demo (sample 10)  │
│                      │
│  Job Description:    │
│  [Upload .docx/.txt] │
│  or paste text ▼     │
│  ┌──────────────┐    │
│  │              │    │
│  └──────────────┘    │
│                      │
│  [▶ Run Ranking]     │
│                      │
│  ──── System ────    │
│  ✓ Model loaded      │
│  ✓ Artifacts ready   │
│  ✓ 100k candidates   │
│  ✗ JD not loaded     │
└──────────────────────┘
```

---

## 6. States & Error Handling

| State | Display |
|---|---|
| No JD loaded | Amber banner: "Load a job description to begin ranking" |
| Artifacts missing | Red banner: "Run precompute.py first — embeddings not found" |
| Running | Spinner + progress bar: "Scoring candidates… 47,231 / 100,000" |
| Done | Green banner: "Top 100 ranked in 4.2 seconds" |
| Validator error | Red banner with exact error string from validate_submission.py |
| Honeypot limit exceeded | Red alert: "WARNING: >10 honeypots detected in top 100 — do not submit" |

---

## 7. Demo Mode (for judges without full dataset)

- Load `sample_candidates.json` (10 candidates provided)
- Accept JD paste in sidebar
- Show score breakdown for all 10
- Label clearly: "Demo Mode — 10 candidates"
- Ranking logic identical to full pipeline

---

*RecruiterIQ UI/UX Design v1.0 · 2025*
