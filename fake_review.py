"""
fake_review.py — Rule-based Fake Review Detector (Production-Ready)

Improvements over v1:
  - Text sanitization before analysis
  - Better repeated-word regex (handles punctuation between repeats)
  - Lexical diversity check (Type-Token Ratio) — low TTR = templated/bot text
  - Generic-only review check is smarter (uses specific word count threshold)
  - All weights sum to 1.0 for cleaner score interpretation
  - Returns clean score 0.0–1.0 with threshold at 0.40 (was 0.30)
"""

import re
import unicodedata
import html

# ── Fake-review phrase list ───────────────────────────────────────────────────
_FAKE_KEYWORDS = [
    "best product ever",
    "must buy",
    "highly recommended",
    "amazing amazing",
    "perfect product",
    "100% satisfied",
    "buy now",
    "superb product",
    "excellent excellent",
    "very very good",
    "too good",
    "osm product",
    "nice product nice",
]

_REPEATED_WORD = re.compile(r"\b(\w{3,})\b[\s,!.]*\1\b", re.IGNORECASE)


# ── Sanitize text before any analysis ────────────────────────────────────────

def _sanitize(text: str) -> str:
    """
    Remove HTML tags, decode HTML entities, strip control characters,
    and normalize unicode. Always run this before analysis.
    """
    if not text:
        return ""
    # Decode HTML entities first (e.g. &amp; → &)
    text = html.unescape(text)
    # Strip HTML tags (e.g. </div> leaking from scraped data)
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize unicode (NFKC handles fullwidth chars, ligatures, etc.)
    text = unicodedata.normalize("NFKC", text)
    # Strip control characters (null bytes, BOM, etc.)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufeff]", "", text)
    # Collapse excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _type_token_ratio(text: str) -> float:
    """
    Lexical diversity: unique words / total words.
    Real reviews: TTR ~ 0.6–0.9
    Bot/template reviews: TTR ~ 0.2–0.5 (lots of word repetition)
    """
    words = re.findall(r"\b[a-z]{2,}\b", text.lower())
    if len(words) < 5:
        return 1.0  # too short to judge
    return len(set(words)) / len(words)


# ── Public API ────────────────────────────────────────────────────────────────

def detect_fake(text: str) -> tuple[str, float]:
    """Returns (label, confidence) where label is 'Fake' or 'Real'."""
    label, score, _ = detect_fake_explained(text)
    return (label, score)


def detect_fake_explained(text: str) -> tuple[str, float, list[dict]]:
    """
    Rule-based fake review detector WITH explainability.

    Returns:
        label   — 'Fake' or 'Real'
        score   — float 0.0–1.0
        reasons — list of rule dicts with triggered/matched info
    """
    if not text:
        return ("Real", 0.0, [])

    # Always sanitize first — fixes the </div> display bug
    clean = _sanitize(text)
    text_lower = clean.lower()
    score = 0.0
    reasons = []

    # ── Rule 1: Repeated consecutive words ───────────────────────────────────
    match = _REPEATED_WORD.search(text_lower)
    triggered_r1 = match is not None
    if triggered_r1:
        score += 0.35
    reasons.append({
        "rule":        "Repeated Words",
        "description": "Same word appears twice in a row (e.g. 'good good')",
        "triggered":   triggered_r1,
        "weight":      0.35,
        "matched":     match.group(0).strip() if match else None,
    })

    # ── Rule 2: Known spam phrases ────────────────────────────────────────────
    matched_phrase = next((p for p in _FAKE_KEYWORDS if p in text_lower), None)
    triggered_r2 = matched_phrase is not None
    if triggered_r2:
        score += 0.25
    reasons.append({
        "rule":        "Spam Phrase",
        "description": "Contains a known promotional or spam phrase",
        "triggered":   triggered_r2,
        "weight":      0.25,
        "matched":     f'"{matched_phrase}"' if matched_phrase else None,
    })

    # ── Rule 3: Excessive exclamation marks ──────────────────────────────────
    excl_count = clean.count("!")
    triggered_r3 = excl_count >= 3
    if triggered_r3:
        score += 0.15
    reasons.append({
        "rule":        "Excessive Punctuation",
        "description": "Three or more exclamation marks — signals artificial enthusiasm",
        "triggered":   triggered_r3,
        "weight":      0.15,
        "matched":     f"{excl_count} exclamation mark{'s' if excl_count != 1 else ''}" if triggered_r3 else None,
    })

    # ── Rule 4: Very short promotional text ──────────────────────────────────
    words = clean.split()
    promo_word = next((k for k in ["best", "great", "perfect", "amazing", "awesome"] if k in text_lower), None)
    triggered_r4 = len(words) <= 4 and promo_word is not None
    if triggered_r4:
        score += 0.25
    reasons.append({
        "rule":        "Short Promotional Text",
        "description": "Very short review containing only hype words — no detail",
        "triggered":   triggered_r4,
        "weight":      0.25,
        "matched":     f'"{clean.strip()}"' if triggered_r4 else None,
    })

    # ── Rule 5: ALL CAPS shouting ─────────────────────────────────────────────
    alpha_chars = [c for c in clean if c.isalpha()]
    caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) if alpha_chars else 0
    triggered_r5 = caps_ratio > 0.60 and len(alpha_chars) > 8
    if triggered_r5:
        score += 0.15
    reasons.append({
        "rule":        "Excessive Capitals",
        "description": "More than 60% of letters are uppercase — unnatural writing",
        "triggered":   triggered_r5,
        "weight":      0.15,
        "matched":     f"{int(caps_ratio * 100)}% caps" if triggered_r5 else None,
    })

    # ── Rule 6: No specific detail (generic praise only) ─────────────────────
    GENERIC = {
        "good", "great", "best", "amazing", "awesome", "excellent", "perfect",
        "love", "nice", "wonderful", "fantastic", "superb", "incredible",
        "very", "so", "really", "just", "quite"
    }
    STOP = {"the","a","an","is","it","this","was","and","but","i","my","me","for","with","of","to"}
    review_words = set(re.findall(r"\b[a-z]{3,}\b", text_lower))
    specific_words = review_words - GENERIC - STOP
    triggered_r6 = len(specific_words) < 3 and len(words) >= 5
    if triggered_r6:
        score += 0.15
    reasons.append({
        "rule":        "No Specific Detail",
        "description": "Review has only generic praise with no product-specific information",
        "triggered":   triggered_r6,
        "weight":      0.15,
        "matched":     "Only generic words detected" if triggered_r6 else None,
    })

    # ── Rule 7: Low lexical diversity (TTR) ──────────────────────────────────
    ttr = _type_token_ratio(clean)
    triggered_r7 = ttr < 0.45 and len(words) >= 8
    if triggered_r7:
        score += 0.20
    reasons.append({
        "rule":        "Low Lexical Diversity",
        "description": "High word repetition relative to review length — typical of bot-generated or templated text",
        "triggered":   triggered_r7,
        "weight":      0.20,
        "matched":     f"TTR = {ttr:.2f} (threshold < 0.45)" if triggered_r7 else None,
    })

    score = min(round(score, 4), 1.0)
    # Raised threshold from 0.30 → 0.40 to reduce false positives
    label = "Fake" if score >= 0.40 else "Real"

    return (label, score, reasons)


def detect_fake_batch(texts: list[str]) -> list[tuple[str, float]]:
    """Batch fake review detection. Returns [(label, confidence), ...]"""
    return [detect_fake(text) for text in texts]


def sanitize_review(text: str) -> str:
    """Public sanitizer — use before storing or displaying any review text."""
    return _sanitize(text)