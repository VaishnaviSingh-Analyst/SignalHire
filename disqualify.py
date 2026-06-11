import re
from datetime import datetime
from typing import Optional, Tuple

from config import (
    CONSULTING_FIRMS,
    CONSULTING_PENALTY,
    CURRENT_YEAR,
    CV_SPEECH_ROBOTICS_KW,
    CV_SPEECH_ROBOTICS_PENALTY,
    DATE_PATTERN,
    GHOST_COMPLETENESS_THRESHOLD,
    HONEYPOT_YEAR_BUFFER,
    ML_AI_TITLE_KEYWORDS,
    NO_CODE_DAYS,
    NO_CODE_PENALTY,
    PRODUCTION_SIGNALS,
    REFERENCE_DATE,
    RESEARCHER_TITLE_KW,
    RETRIEVAL_SIGNALS,
)
from textmatch import matches_any


def parse_year(date_str: Optional[str]) -> Optional[int]:
    if date_str is None:
        return None
    date_str = str(date_str).strip().lower()
    if date_str in ("present", "now", ""):
        return CURRENT_YEAR
    match = DATE_PATTERN.search(date_str)
    if match:
        return int(match.group(0))
    return None


def sort_career_chronologically(career: list) -> list:
    """Oldest role first. The dataset lists roles newest-first, so any
    positional logic must sort explicitly instead of trusting order."""
    def key(role):
        start = str(role.get("start_date") or "")
        year = parse_year(start)
        # ISO date strings sort lexicographically; fall back to year only.
        return (year if year is not None else 0, start)
    return sorted(career, key=key)


def _latest_role(career: list) -> Optional[dict]:
    if not career:
        return None
    for role in career:
        if role.get("is_current", False):
            return role
    return sort_career_chronologically(career)[-1]


def _earliest_role_start_year(candidate: dict) -> Optional[int]:
    career = candidate.get("career_history", [])
    if not career:
        return None
    years = []
    for role in career:
        y = parse_year(role.get("start_date"))
        if y is not None:
            years.append(y)
    return min(years) if years else None


def _all_roles_at_consulting(candidate: dict) -> bool:
    career = candidate.get("career_history", [])
    if not career:
        return False
    saw_consulting = False
    for role in career:
        company = role.get("company", "")
        if not company:
            continue
        company_lower = company.strip().lower()
        if any(firm in company_lower for firm in CONSULTING_FIRMS):
            saw_consulting = True
        else:
            return False
    # All-empty company names must not count as all-consulting.
    return saw_consulting


def _is_pure_research(candidate: dict) -> bool:
    career = candidate.get("career_history", [])
    if not career:
        return False
    researcher_count = 0
    for role in career:
        title = (role.get("title", "") or "").lower()
        if any(kw in title for kw in RESEARCHER_TITLE_KW):
            researcher_count += 1
    if researcher_count < len(career):
        return False
    desc_text = " ".join((r.get("description", "") or "") for r in career)
    if matches_any(desc_text, PRODUCTION_SIGNALS):
        return False
    return True


def _is_no_code_18mo(candidate: dict) -> bool:
    signals = candidate.get("redrob_signals", {})
    last_active = signals.get("last_active_date", "")
    if not last_active:
        return False
    try:
        last_dt = datetime.strptime(str(last_active)[:10], "%Y-%m-%d").date()
        ref_dt = datetime.strptime(REFERENCE_DATE, "%Y-%m-%d").date()
        if (ref_dt - last_dt).days <= NO_CODE_DAYS:
            return False
    except (ValueError, TypeError):
        return False
    career = candidate.get("career_history", [])
    if not career:
        return True
    latest = _latest_role(career)
    if latest.get("is_current", False):
        return False
    end_date = latest.get("end_date", "")
    end_year = parse_year(end_date)
    if end_year is None:
        duration = latest.get("duration_months", 0) or 0
        if duration > 0:
            return False
        return True
    if CURRENT_YEAR - end_year >= 2:
        return True
    return False


def is_honeypot(candidate: dict) -> Tuple[bool, str]:
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience")
    earliest_year = _earliest_role_start_year(candidate)

    if yoe is not None and earliest_year is not None:
        max_reasonable = CURRENT_YEAR - earliest_year + HONEYPOT_YEAR_BUFFER
        if yoe > max_reasonable:
            return (
                True,
                f"HONEYPOT: yoe={yoe} > {CURRENT_YEAR}-{earliest_year}+{HONEYPOT_YEAR_BUFFER}={max_reasonable}",
            )

    signals = candidate.get("redrob_signals", {})
    completeness = signals.get("profile_completeness_score", 100)
    verified_email = signals.get("verified_email", False)
    verified_phone = signals.get("verified_phone", False)

    if completeness < GHOST_COMPLETENESS_THRESHOLD and not verified_email and not verified_phone:
        return (
            True,
            f"GHOST: completeness={completeness}, no verified email or phone",
        )

    if _is_pure_research(candidate):
        return (True, "PURE_RESEARCH: all roles researcher, no deployment evidence")

    return False, ""


_CV_KW_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in CV_SPEECH_ROBOTICS_KW) + r")\b"
)


def _matches_any_kw(text: str) -> bool:
    return bool(_CV_KW_PATTERN.search(text.lower()))


def _is_cv_speech_robotics_only(candidate: dict) -> bool:
    has_cv_robotics = False
    for sk in candidate.get("skills", []):
        name = sk.get("name", "").lower().strip()
        if _matches_any_kw(name):
            has_cv_robotics = True
            break
    if not has_cv_robotics:
        title_text = " ".join(
            (r.get("title", "") or "") for r in candidate.get("career_history", [])
        )
        if not _matches_any_kw(title_text):
            return False
    desc_text = " ".join(
        (r.get("description", "") or "") for r in candidate.get("career_history", [])
    )
    combined = desc_text + " " + " ".join(
        sk.get("name", "") for sk in candidate.get("skills", [])
    )
    if matches_any(combined, RETRIEVAL_SIGNALS):
        return False
    title_text_all = " ".join(
        (r.get("title", "") or "") for r in candidate.get("career_history", [])
    ).lower()
    if any(kw in title_text_all for kw in ML_AI_TITLE_KEYWORDS):
        return True
    return False


def should_hard_penalize(candidate: dict) -> Tuple[float, str]:
    reasons = []
    penalty = 1.0

    if _all_roles_at_consulting(candidate):
        penalty *= CONSULTING_PENALTY
        reasons.append("ALL_ROLES_CONSULTING")

    if _is_no_code_18mo(candidate):
        penalty *= NO_CODE_PENALTY
        reasons.append("NO_CODE_18MO")

    if _is_cv_speech_robotics_only(candidate):
        penalty *= CV_SPEECH_ROBOTICS_PENALTY
        reasons.append("CV_SPEECH_ROBOTICS_ONLY")

    if not reasons:
        return (1.0, "")
    return (penalty, " | ".join(reasons))
