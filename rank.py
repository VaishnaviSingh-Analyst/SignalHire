import csv
import json
import pickle
import subprocess
import sys
import time

import numpy as np

from config import (
    ARTIFACTS_DIR,
    CANDIDATES_PATH,
    OUTPUT_DIR,
    TOP_K,
    VALIDATE_SCRIPT,
    WEIGHTS,
)
from evidence import generate_reasoning


def load_candidates_by_ids(target_ids) -> dict:
    """Single pass over the JSONL collecting all requested candidates,
    instead of rescanning the 100K-line file once per id."""
    remaining = set(target_ids)
    found = {}
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not remaining:
                break
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            cid = cand.get("candidate_id")
            if cid in remaining:
                found[cid] = cand
                remaining.discard(cid)
    return found


def main():
    t0 = time.time()

    print("Loading artifacts ...")
    embeddings = np.load(str(ARTIFACTS_DIR / "embeddings.npy")).astype(np.float32)
    candidate_ids = np.load(str(ARTIFACTS_DIR / "candidate_ids.npy"), allow_pickle=True)
    jd_embedding = np.load(str(ARTIFACTS_DIR / "jd_embedding.npy")).astype(np.float32)

    with open(ARTIFACTS_DIR / "subscores.pkl", "rb") as f:
        subscores_dict = pickle.load(f)

    n = len(candidate_ids)
    print(f"Loaded {n} candidates, embeddings shape: {embeddings.shape}")

    semantic_sim = embeddings @ jd_embedding

    weight_vector = np.array(
        [WEIGHTS["technical_fit"], WEIGHTS["career_quality"], WEIGHTS["availability_signal"], WEIGHTS["seniority_fit"]],
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

    print("Computing composite scores ...")
    base_scores = subscore_matrix @ weight_vector
    scores = penalty_multipliers * (base_scores + WEIGHTS["semantic_similarity"] * semantic_sim)

    top_indices = np.argpartition(-scores, TOP_K)[:TOP_K]
    top_k_pairs = [(float(scores[i]), str(candidate_ids[i])) for i in top_indices]

    top_k_pairs.sort(key=lambda x: (-round(x[0], 3), x[1]))

    top_ids = np.array([p[1] for p in top_k_pairs], dtype=object)

    print(f"Top score: {top_k_pairs[0][0]:.4f}, bottom: {top_k_pairs[-1][0]:.4f}")

    print("Loading top candidates for reasoning ...")
    candidates_by_id = load_candidates_by_ids(cid for _, cid in top_k_pairs)
    top_candidates = [candidates_by_id.get(cid) for _, cid in top_k_pairs]

    output_path = OUTPUT_DIR / "submission.csv"
    print(f"Writing {output_path} ...")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank_idx, (score_val, cid) in enumerate(top_k_pairs):
            rank = rank_idx + 1
            cand = top_candidates[rank_idx]
            reasoning = generate_reasoning(cand or {})
            writer.writerow([cid, rank, f"{round(score_val, 3):.3f}", reasoning])

    elapsed = time.time() - t0
    print(f"Ranking complete in {elapsed:.1f}s")

    print("Validating submission ...")
    result = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT), str(output_path)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        print("VALIDATION FAILED")
        sys.exit(1)
    print("Submission valid!")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
