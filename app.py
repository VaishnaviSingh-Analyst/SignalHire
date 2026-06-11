import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    ARTIFACTS_DIR,
    CANDIDATES_PATH,
    SAMPLE_PATH,
    OUTPUT_DIR,
    TOP_K,
    WEIGHTS,
)
from evidence import collect_evidence, generate_reasoning
from rank import load_candidates_by_ids


@st.cache_data(show_spinner=False)
def cached_candidates(ids: tuple) -> dict:
    return load_candidates_by_ids(ids)

st.set_page_config(page_title="RecruiterIQ", layout="wide")

SEED = 42
np.random.seed(SEED)


@st.cache_resource
def load_artifacts():
    artifacts = {}
    artifacts["embeddings"] = np.load(str(ARTIFACTS_DIR / "embeddings.npy")).astype(np.float32)
    artifacts["candidate_ids"] = np.load(str(ARTIFACTS_DIR / "candidate_ids.npy"), allow_pickle=True)
    artifacts["jd_embedding"] = np.load(str(ARTIFACTS_DIR / "jd_embedding.npy")).astype(np.float32)
    with open(ARTIFACTS_DIR / "subscores.pkl", "rb") as f:
        artifacts["subscores"] = pickle.load(f)
    with open(ARTIFACTS_DIR / "disqualified.json", "r") as f:
        artifacts["disqualified"] = json.load(f)
    return artifacts


def compute_ranking(embeddings, candidate_ids, jd_embedding, subscores_dict, weights):
    n = len(candidate_ids)
    semantic_sim = embeddings @ jd_embedding
    weight_vector = np.array(
        [weights["technical_fit"], weights["career_quality"], weights["availability_signal"], weights["seniority_fit"]],
        dtype=np.float32,
    )
    subscore_matrix = np.zeros((n, 4), dtype=np.float32)
    penalty_multipliers = np.ones(n, dtype=np.float32)
    for i, cid in enumerate(candidate_ids):
        ss = subscores_dict.get(cid, {})
        subscore_matrix[i, 0] = ss.get("technical_fit", 0.0)
        subscore_matrix[i, 1] = ss.get("career_quality", 0.0)
        subscore_matrix[i, 2] = ss.get("availability_signal", 0.0)
        subscore_matrix[i, 3] = ss.get("seniority_fit", 0.0)
        penalty_multipliers[i] = ss.get("penalty_multiplier", 1.0)
    base_scores = subscore_matrix @ weight_vector
    scores = penalty_multipliers * (base_scores + weights["semantic_similarity"] * semantic_sim)
    top_indices = np.argpartition(-scores, TOP_K)[:TOP_K]
    top_k_pairs = [(float(scores[i]), str(candidate_ids[i])) for i in top_indices]
    top_k_pairs.sort(key=lambda x: (-round(x[0], 3), x[1]))
    return top_k_pairs, subscore_matrix, scores


def render_score_bars(scores_dict):
    labels = list(scores_dict.keys())
    values = list(scores_dict.values())
    colors = ["#4F8EF7", "#3ECF8E", "#F59E0B", "#EF4444", "#94A3B8"]
    fig, ax = plt.subplots(figsize=(6, 1.5))
    for i, (label, val) in enumerate(zip(labels, values)):
        ax.barh(label, val, color=colors[i % len(colors)], height=0.6)
        ax.text(val + 0.01, i, f"{val:.0%}", va="center", fontsize=9, color="#94A3B8")
    ax.set_xlim(0, 1)
    ax.set_facecolor("#1A1D27")
    fig.patch.set_facecolor("#1A1D27")
    ax.tick_params(colors="#94A3B8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.title("RecruiterIQ")
    st.caption("AI Candidate Ranking System — Redrob AI Challenge")

    artifacts_ready = False
    try:
        artifacts = load_artifacts()
        artifacts_ready = True
    except Exception as e:
        st.error(f"Artifacts not found: {e}. Run precompute.py first.")

    with st.sidebar:
        st.header("Configuration")
        mode = st.radio("Mode", ["Full Pipeline (100k)", "Demo (sample 10)"], index=0)

        jd_text = st.text_area(
            "Job Description",
            height=150,
            placeholder="Paste job description or leave blank for default JD...",
        )

        run_btn = st.button("Run Ranking", type="primary", width="stretch")

        st.divider()
        st.caption("System Status")
        if artifacts_ready:
            n = len(artifacts["candidate_ids"])
            st.success(f"Candidates: {n:,}")
            disq_count = len(artifacts["disqualified"])
            st.info(f"Disqualified: {disq_count}")
        else:
            st.warning("Artifacts not loaded")

    tab1, tab2, tab3, tab4 = st.tabs(["Ranked Shortlist", "Score Explorer", "Methodology", "Disqualified"])

    if run_btn and artifacts_ready:
        with st.spinner("Scoring candidates..."):
            emb = artifacts["embeddings"]
            ids = artifacts["candidate_ids"]
            jd_emb = artifacts["jd_embedding"]
            subs = artifacts["subscores"]

            rank_pairs, subscore_matrix, all_scores = compute_ranking(emb, ids, jd_emb, subs, WEIGHTS)

        candidates_by_id = cached_candidates(tuple(cid for _, cid in rank_pairs))

        with tab1:
            st.subheader(f"Top {TOP_K} Candidates")
            for rank_idx, (score_val, cid) in enumerate(rank_pairs):
                rank = rank_idx + 1
                cand = candidates_by_id.get(cid)
                ss = subs.get(cid, {})
                if cand is None:
                    continue

                profile = cand.get("profile", {})
                signals = cand.get("redrob_signals", {})

                with st.expander(
                    f"#{rank}  {profile.get('current_title', '?')} @ {profile.get('current_company', '?')}  —  Score: {score_val:.3f}",
                    expanded=(rank <= 3),
                ):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f"**Years Exp:** {profile.get('years_of_experience', '?')}")
                        st.markdown(f"**Headline:** {profile.get('headline', '')[:100]}")
                        st.markdown(f"**Location:** {profile.get('location', '?')}")
                        st.markdown(f"**Open to work:** {'Yes' if signals.get('open_to_work_flag') else 'No'}")
                        st.markdown(f"**Response rate:** {signals.get('recruiter_response_rate', 0):.0%}")
                        st.markdown(f"**Notice:** {signals.get('notice_period_days', 90)}d")

                    with col2:
                        score_dict = {
                            "Technical": ss.get("technical_fit", 0),
                            "Career": ss.get("career_quality", 0),
                            "Availability": ss.get("availability_signal", 0),
                            "Seniority": ss.get("seniority_fit", 0),
                            "Semantic": float(emb[ids == cid][0] @ jd_emb) if len(ids[ids == cid]) > 0 else 0,
                        }
                        render_score_bars(score_dict)

                    reasoning = generate_reasoning(cand)
                    st.caption(f"*{reasoning}*")

        with tab2:
            st.subheader("Score Distribution")
            top_1000_scores = np.sort(all_scores)[-1000:]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(top_1000_scores, bins=30, color="#4F8EF7", edgecolor="#1A1D27")
            ax.axvline(x=all_scores[np.argpartition(-all_scores, TOP_K)[:TOP_K]].min(), color="#EF4444", linestyle="--", label="Top 100 cutoff")
            ax.set_xlabel("Composite Score")
            ax.set_ylabel("Count (top 1000)")
            ax.set_facecolor("#1A1D27")
            fig.patch.set_facecolor("#1A1D27")
            ax.tick_params(colors="#94A3B8")
            ax.spines["bottom"].set_color("#2D3148")
            ax.spines["left"].set_color("#2D3148")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Signal Correlation")
            tech_scores = subscore_matrix[:, 0]
            career_scores = subscore_matrix[:, 1]
            avail_scores = subscore_matrix[:, 2]
            senior_scores = subscore_matrix[:, 3]

            fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].scatter(tech_scores, career_scores, c=avail_scores, cmap="viridis", alpha=0.3, s=5)
            axes[0].set_xlabel("Technical Fit")
            axes[0].set_ylabel("Career Quality")
            axes[0].set_facecolor("#1A1D27")
            axes[0].tick_params(colors="#94A3B8")
            axes[0].spines["bottom"].set_color("#2D3148")
            axes[0].spines["left"].set_color("#2D3148")

            sc = axes[1].scatter(tech_scores, senior_scores, c=all_scores, cmap="plasma", alpha=0.3, s=5)
            axes[1].set_xlabel("Technical Fit")
            axes[1].set_ylabel("Seniority Fit")
            axes[1].set_facecolor("#1A1D27")
            axes[1].tick_params(colors="#94A3B8")
            axes[1].spines["bottom"].set_color("#2D3148")
            axes[1].spines["left"].set_color("#2D3148")

            fig2.patch.set_facecolor("#1A1D27")
            st.pyplot(fig2)
            plt.close(fig2)

        with tab3:
            st.header("Methodology")
            st.markdown("""
            ### Composite Score Formula
            ```
            S = 0.35 × Technical_Fit
              + 0.25 × Career_Quality
              + 0.20 × Availability_Signal
              + 0.12 × Seniority_Fit
              + 0.08 × Semantic_Similarity
            ```
            All sub-scores normalized to [0, 1].
            """)

            st.markdown("### Scoring Signals")
            st.markdown("""
            | Signal | Weight | What it measures |
            |---|---|---|
            | Technical Fit | 0.35 | JD skill match (embeddings, vector DBs, Python, eval) |
            | Career Quality | 0.25 | Product company history, tenure, progression |
            | Availability | 0.20 | Open to work, response rate, notice period |
            | Seniority Fit | 0.12 | YoE range 6-9 ideal, education tier bonus |
            | Semantic Similarity | 0.08 | Embedding cosine vs JD — catches plain-language engineers |
            """)

            st.markdown("### Disqualification Rules")
            st.markdown("""
            - **Honeypot**: YoE > career timeline + buffer (2yr)
            - **Ghost**: completeness < 5% + no verified email/phone
            - **Consulting Penalty**: All-consulting career → 0.15× multiplier on composite
            """)

            st.markdown("### Pipeline")
            st.markdown("""
            1. **Precompute** (offline): Stream 100K JSONL → disqualify → embed (MiniLM-L6-v2) → sub-scores → serialize artifacts
            2. **Ranking** (sandbox, <5min): Load artifacts → vectorized cosine → weighted composite → top 100 → reasoning → validated CSV
            """)

        with tab4:
            st.subheader("Disqualified Candidates")
            disqualified = artifacts.get("disqualified", [])
            st.metric("Total Disqualified", len(disqualified))
            if disqualified:
                df = pd.DataFrame(disqualified)
                st.dataframe(df, width="stretch", hide_index=True)
            else:
                st.info("No disqualified candidates found.")

    elif run_btn and not artifacts_ready:
        st.error("Cannot run ranking: artifacts not loaded. Run precompute.py first.")

    if not run_btn:
        with tab1:
            st.info("Configure settings in the sidebar and click 'Run Ranking' to begin.")
        with tab2:
            st.info("Run ranking to see score distribution charts.")
        with tab3:
            st.info("See methodology below.")
            st.markdown("""
            ### Composite Score Formula
            ```
            S = 0.35 × Technical_Fit + 0.25 × Career_Quality + 0.20 × Availability_Signal + 0.12 × Seniority_Fit + 0.08 × Semantic_Similarity
            ```
            """)
        with tab4:
            if artifacts_ready:
                disqualified = artifacts.get("disqualified", [])
                st.metric("Total Disqualified", len(disqualified))
                if disqualified:
                    df = pd.DataFrame(disqualified)
                    st.dataframe(df, width="stretch", hide_index=True)
            else:
                st.info("No data loaded.")


if __name__ == "__main__":
    main()
