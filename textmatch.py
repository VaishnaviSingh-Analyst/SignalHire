"""Word-boundary keyword matching for signal scans.

Plain substring checks produced false positives: "rag" matched inside
"storage", "map" inside "roadmap". Keywords now match at a word boundary;
stems of 4+ chars may extend with word characters so "eval" still matches
"evaluation" and "benchmark" matches "benchmarking", while short keywords
("rag", "mrr", "map", "e5") must match as whole words.
"""

import re
from functools import lru_cache
from typing import List


@lru_cache(maxsize=1024)
def keyword_pattern(kw: str) -> re.Pattern:
    k = kw.lower().strip()
    esc = re.escape(k)
    if len(k) >= 4 and k[-1].isalnum():
        return re.compile(r"\b" + esc + r"\w*", re.IGNORECASE)
    return re.compile(r"\b" + esc + r"\b", re.IGNORECASE)


def matches_keyword(text_lower: str, kw: str) -> bool:
    return bool(keyword_pattern(kw).search(text_lower))


def matches_any(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(keyword_pattern(kw).search(t) for kw in keywords)


def matched_keywords(text: str, keywords: List[str]) -> List[str]:
    t = text.lower()
    return [kw for kw in keywords if keyword_pattern(kw).search(t)]
