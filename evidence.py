"""Evidence extraction: which JD requirement matched which exact skill or
profile sentence. Powers honest, citable reasoning in the CSV and the UI
instead of templated claims."""

import re
from typing import Dict, List

from config import (
    JD_MUST_HAVES,
    JD_MUST_KEYWORDS,
    JD_NICE_TO_HAVES,
    JD_NICE_KEYWORDS,
    PRODUCTION_SIGNALS,
    RETRIEVAL_SIGNALS,
)
from signals import _build_skill_dict, skill_matches_keyword
from textmatch import keyword_pattern, matched_keywords, matches_keyword

SNIPPET_RADIUS = 60


def _snippet_around(text: str, keyword: str) -> str:
    m = keyword_pattern(keyword).search(text.lower())
    if not m:
        return ""
    idx = m.start()
    start = max(0, idx - SNIPPET_RADIUS)
    end = min(len(text), m.end() + SNIPPET_RADIUS)
    snippet = text[start:end].strip()
    snippet = re.sub(r"\s+", " ", snippet)
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


def _career_text(candidate: dict) -> str:
    return " ".join(
        (r.get("description", "") or "") for r in candidate.get("career_history", [])
    )


def _profile_blob(candidate: dict) -> str:
    profile = candidate.get("profile", {})
    return " ".join(
        [
            profile.get("headline", "") or "",
            profile.get("summary", "") or "",
            profile.get("current_title", "") or "",
            _career_text(candidate),
        ]
    )


def _match_criterion(keywords: List[str], skill_dict: Dict[str, float], blob: str) -> dict:
    """Best match for one JD criterion: prefers declared skills (with
    proficiency weight) over plain text mentions."""
    best = None
    for kw in keywords:
        kw_lower = kw.lower().strip()
        for skill_name, skill_weight in skill_dict.items():
            if skill_matches_keyword(kw_lower, skill_name):
                if best is None or skill_weight > best["strength"]:
                    best = {
                        "source": "skill",
                        "keyword": kw,
                        "detail": skill_name,
                        "strength": skill_weight,
                    }
        if (best is None or best["strength"] < 0.3) and matches_keyword(blob.lower(), kw_lower):
            best = {
                "source": "text",
                "keyword": kw,
                "detail": _snippet_around(blob, kw_lower),
                "strength": 0.3,
            }
    return best


def collect_evidence(candidate: dict) -> dict:
    """Returns matched/missing JD criteria plus production and retrieval
    signal hits, each with the skill or text snippet that triggered it."""
    skill_dict = _build_skill_dict(candidate)
    blob = _profile_blob(candidate)
    career_text = _career_text(candidate)

    matched, missing = [], []
    for group, criteria, kw_map in (
        ("must-have", JD_MUST_HAVES, JD_MUST_KEYWORDS),
        ("nice-to-have", JD_NICE_TO_HAVES, JD_NICE_KEYWORDS),
    ):
        for criterion, weight in criteria.items():
            hit = _match_criterion(kw_map.get(criterion, [criterion]), skill_dict, blob)
            entry = {"criterion": criterion, "group": group, "weight": weight}
            if hit:
                matched.append({**entry, **hit})
            elif group == "must-have":
                missing.append(entry)

    production_hits = [
        {"keyword": s, "snippet": _snippet_around(career_text, s)}
        for s in matched_keywords(career_text, PRODUCTION_SIGNALS)
    ]
    retrieval_hits = [
        {"keyword": s, "snippet": _snippet_around(career_text, s)}
        for s in matched_keywords(career_text, RETRIEVAL_SIGNALS)
    ]

    return {
        "matched": matched,
        "missing_must_haves": missing,
        "production": production_hits,
        "retrieval": retrieval_hits,
    }


def generate_reasoning(candidate: dict, evidence: dict = None, max_len: int = 300) -> str:
    """Honest one-liner citing only what the profile actually shows."""
    if not candidate:
        return ""
    if evidence is None:
        evidence = collect_evidence(candidate)

    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    title = (profile.get("current_title") or "engineer").strip()
    yoe = profile.get("years_of_experience", 0)
    company = profile.get("current_company") or ""
    at_company = f" at {company}" if company else ""

    skill_hits = [
        e["detail"] for e in evidence["matched"]
        if e["source"] == "skill" and e["group"] == "must-have"
    ]
    strengths = ", ".join(dict.fromkeys(skill_hits))[:80] if skill_hits else ""
    strengths_part = f"; strong match on {strengths}" if strengths else ""

    prod_part = ""
    if evidence["production"]:
        kws = ", ".join(dict.fromkeys(h["keyword"] for h in evidence["production"][:2]))
        prod_part = f"; production evidence ({kws})"

    open_work = signals.get("open_to_work_flag", False)
    avail_bits = ["actively looking" if open_work else "passive"]
    rr = signals.get("recruiter_response_rate", 0) or 0
    if rr > 0:
        avail_bits.append(f"{rr:.0%} response rate")
    notice = signals.get("notice_period_days", 90) or 90
    if notice < 90:
        avail_bits.append(f"{notice}d notice")

    missing_part = ""
    if evidence["missing_must_haves"]:
        gaps = ", ".join(m["criterion"] for m in evidence["missing_must_haves"][:2])
        missing_part = f" Gap: {gaps}."

    reasoning = (
        f"{yoe}yr {title}{at_company}{strengths_part}{prod_part}; "
        f"{', '.join(avail_bits)}.{missing_part}"
    )

    for ch in ["\n", "\r", "\"", "\t"]:
        reasoning = reasoning.replace(ch, " ")
    reasoning = re.sub(r"\s+", " ", reasoning).strip()
    if len(reasoning) > max_len:
        reasoning = reasoning[: max_len - 3] + "..."
    return reasoning
