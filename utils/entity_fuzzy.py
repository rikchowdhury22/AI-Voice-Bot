# utils/entity_fuzzy.py
from typing import Optional, Dict, List, Set
from rapidfuzz import process, fuzz
from .spelling_helper import CANON_PROJECTS, CANON_CATEGORIES

PROJECT_LIST: List[str] = sorted({p for p in CANON_PROJECTS})
CATEGORY_LIST: List[str] = sorted({c for c in CANON_CATEGORIES})

# Build variant -> canonical maps
VAR_TO_CANON_PROJECT: Dict[str, str] = {}
for canon, variants in CANON_PROJECTS.items():
    for v in variants | {canon}:
        VAR_TO_CANON_PROJECT[v.lower()] = canon

VAR_TO_CANON_CAT: Dict[str, str] = {}
for canon, variants in CANON_CATEGORIES.items():
    for v in variants | {canon}:
        VAR_TO_CANON_CAT[v.lower()] = canon


def _token_candidates(text: str) -> List[str]:
    """Return canonical project candidates found by token inclusion."""
    toks = text.lower().split()
    found: List[str] = []
    for tok in toks:
        if tok in VAR_TO_CANON_PROJECT:
            found.append(VAR_TO_CANON_PROJECT[tok])
    # If both brand "Ashar" and a specific project are present, drop "Ashar"
    if "Ashar" in found and len(set(found)) > 1:
        found = [c for c in found if c != "Ashar"]
    # De‑dup preserving order
    seen: Set[str] = set()
    uniq = []
    for c in found:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def detect_project(text: str, threshold_high: int = 90, threshold_low: int = 78) -> Optional[str]:
    """
    Prefer whole-string fuzzy match; fall back to token hits.
    This avoids picking 'Ashar' when the user says 'ashar pulse'.
    """
    # 1) Whole-string fuzzy match against canonical list
    match = process.extractOne(text, PROJECT_LIST, scorer=fuzz.WRatio)
    if match:
        name, score, _ = match
        if score >= threshold_high:
            return name

    # 2) Token candidates (drop 'Ashar' if other candidates exist)
    cand = _token_candidates(text)
    if len(cand) == 1:
        return cand[0]
    elif len(cand) > 1:
        # pick the one with best fuzzy score against the full text
        best = process.extractOne(text, cand, scorer=fuzz.WRatio)
        if best:
            name, score, _ = best
            return name

    # 3) If whole-string fuzzy was good enough but not perfect, accept it
    if match:
        name, score, _ = match
        if score >= threshold_low:
            return name

    return None


def detect_category(text: str, threshold_high: int = 90, threshold_low: int = 78) -> Optional[str]:
    # Quick substring pass
    for key in VAR_TO_CANON_CAT:
        if key in text.lower():
            return VAR_TO_CANON_CAT[key]
    # Fuzzy fallback
    match = process.extractOne(text, CATEGORY_LIST, scorer=fuzz.WRatio)
    if not match:
        return None
    name, score, _ = match
    if score >= threshold_high:
        return name
    if score >= threshold_low:
        return name
    return None
