"""
Absamodel.py — Aspect-Based Sentiment Analysis (Memory-Optimised for Render 512MB)

Changes vs previous version:
  ✦ low_cpu_mem_usage=True passed via model_kwargs so the pipeline loader
    doesn't transiently double RAM during weight allocation
  ✦ torch.cuda.empty_cache() + gc.collect() after load to reclaim any
    temporary allocations made by HuggingFace internals
  ✦ Explicit unload() function so main.py can free ABSA RAM before
    reloading the sentiment model after an /absa call
  ✦ All other logic (keyword gate, INT8 quant, NLI hypotheses) unchanged

Strategy: zero-shot NLI approach using a lightweight model.
  Model : typeform/distilbert-base-uncased-mnli  (~135MB, fast on CPU)
  Why   : No fine-tuning needed. We frame aspect sentiment as a
          natural language inference task:
            premise   = the review text
            hypothesis = "The {aspect} is {polarity}."
          The model scores entailment vs contradiction to decide sentiment.

i3 / free-tier optimisations:
  - DistilBERT (not BERT/RoBERTa) — 40% smaller, 60% faster on CPU
  - INT8 dynamic quantization applied at load time
  - max_length=128 for the NLI classifier
  - Only run aspects that are actually mentioned in the text (keyword gate)
    — avoids ~60-70% of NLI calls on typical reviews
  - low_cpu_mem_usage=True prevents transient 2× RAM spike during load

Aspects covered (extensible — just add to ASPECTS):
  battery, camera, display, sound, performance, price, delivery,
  build quality, customer service, software, design, size

Returns list of:
  { aspect, sentiment, confidence, mentioned, snippet }
"""

from __future__ import annotations
import gc
import re
import torch
from transformers import pipeline as hf_pipeline

# ── Model ─────────────────────────────────────────────────────────────────────

MODEL_NAME = "typeform/distilbert-base-uncased-mnli"

_pipe = None   # lazy-loaded


def _load() -> None:
    global _pipe

    if _pipe is not None:
        return

    print(f"[absa_model] Loading {MODEL_NAME}…")
    try:
        # ── KEY FIX: low_cpu_mem_usage=True via model_kwargs ─────────────────
        # The pipeline() API forwards model_kwargs to from_pretrained(), which
        # enables layer-by-layer allocation and prevents the transient 2× RAM
        # spike that was killing Render instances during ABSA load.
        _pipe = hf_pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            device=-1,                              # CPU
            model_kwargs={"low_cpu_mem_usage": True},
        )

        # INT8 quantization — ~2× faster on i3/free-tier CPU
        try:
            _pipe.model = torch.quantization.quantize_dynamic(
                _pipe.model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            print("[absa_model] INT8 quantization applied ✓")
        except Exception as e:
            print(f"[absa_model] Quantization skipped: {e}")

        # Reclaim any scratch memory used by HuggingFace during load
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[absa_model] Ready")

    except Exception as e:
        print(f"[absa_model] Failed to load: {e}")
        _pipe = "fallback"


def unload() -> None:
    """
    Release the ABSA model from RAM so the sentiment model can reload.

    Call this at the end of every /absa handler so memory is freed
    before the next request arrives (which may be a batch job that
    needs the sentiment model).
    """
    global _pipe

    if _pipe is None or _pipe == "fallback":
        return

    print("[absa_model] Unloading to free RAM…")
    del _pipe
    _pipe = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[absa_model] Unloaded ✓")


# ── Aspect definitions ─────────────────────────────────────────────────────────
# Each aspect has a list of trigger keywords. If NONE appear in the review text
# (case-insensitive) we skip NLI for that aspect entirely — the biggest speed
# win on CPU, avoiding ~65% of forward passes on typical reviews.

ASPECTS: dict[str, list[str]] = {
    "Battery":          ["battery", "charge", "charging", "power", "drain", "last", "life"],
    "Camera":           ["camera", "photo", "picture", "image", "video", "lens", "shot", "selfie", "megapixel"],
    "Display":          ["screen", "display", "resolution", "brightness", "color", "colour", "touch", "panel"],
    "Sound":            ["sound", "audio", "speaker", "headphone", "noise", "volume", "bass", "mic", "microphone"],
    "Performance":      ["speed", "fast", "slow", "lag", "hang", "crash", "performance", "processor", "ram", "smooth", "freeze"],
    "Price":            ["price", "cost", "value", "worth", "cheap", "expensive", "afford", "money", "budget"],
    "Delivery":         ["delivery", "shipping", "arrived", "package", "packaging", "days", "courier", "dispatch"],
    "Build Quality":    ["build", "quality", "material", "plastic", "metal", "sturdy", "fragile", "durable", "break", "crack", "feel"],
    "Customer Service": ["service", "support", "refund", "return", "helpdesk", "customer", "response", "reply", "staff"],
    "Software":         ["app", "software", "update", "ui", "interface", "bug", "os", "system", "feature"],
    "Design":           ["design", "look", "beautiful", "slim", "thin", "colour", "color", "style", "aesthetic"],
    "Size / Weight":    ["size", "weight", "heavy", "light", "compact", "large", "small", "portable", "pocket"],
}

# NLI hypothesis templates — compare "positive" vs "negative" entailment scores
_HYP_POS = "The {aspect} is good."
_HYP_NEG = "The {aspect} is bad."


# ── Public API ─────────────────────────────────────────────────────────────────

def analyse_aspects(text: str) -> list[dict]:
    """
    Run ABSA on a single review text.

    Returns a list of dicts — one per detected aspect:
      {
        "aspect":      str,           # e.g. "Battery"
        "sentiment":   str,           # "Positive" | "Negative" | "Neutral"
        "confidence":  float,         # 0.0–1.0
        "mentioned":   bool,          # True = keyword matched in text
        "snippet":     str | None,    # sentence fragment containing the keyword
      }

    Only aspects whose keywords appear in the text are evaluated.
    Results are sorted: Negative first (most actionable), then Positive, Neutral.
    """
    if not text or not text.strip():
        return []

    _load()

    text_lower = text.lower()
    results: list[dict] = []

    for aspect, keywords in ASPECTS.items():

        # ── Keyword gate ──────────────────────────────────────────────────────
        # Skip NLI entirely if no keyword is present — avoids ~65% of
        # expensive forward passes on typical product reviews.
        matched_kw = next((kw for kw in keywords if kw in text_lower), None)
        if matched_kw is None:
            continue

        snippet = _extract_snippet(text, matched_kw)

        # ── Fallback mode (model failed to load) ──────────────────────────────
        if _pipe == "fallback":
            results.append({
                "aspect":     aspect,
                "sentiment":  "Neutral",
                "confidence": 0.5,
                "mentioned":  True,
                "snippet":    snippet,
            })
            continue

        # ── NLI inference ─────────────────────────────────────────────────────
        try:
            out = _pipe(
                text,
                candidate_labels=[
                    _HYP_POS.format(aspect=aspect),
                    _HYP_NEG.format(aspect=aspect),
                ],
                multi_label=False,
                truncation=True,
                max_length=128,
            )

            label_scores = dict(zip(out["labels"], out["scores"]))
            pos_score    = label_scores.get(_HYP_POS.format(aspect=aspect), 0.0)
            neg_score    = label_scores.get(_HYP_NEG.format(aspect=aspect), 0.0)

            gap = abs(pos_score - neg_score)

            if gap < 0.15:
                sentiment  = "Neutral"
                confidence = round(1.0 - gap, 3)
            elif pos_score > neg_score:
                sentiment  = "Positive"
                confidence = round(pos_score, 3)
            else:
                sentiment  = "Negative"
                confidence = round(neg_score, 3)

            results.append({
                "aspect":     aspect,
                "sentiment":  sentiment,
                "confidence": confidence,
                "mentioned":  True,
                "snippet":    snippet,
            })

        except Exception as e:
            print(f"[absa_model] NLI failed for aspect '{aspect}': {e}")
            results.append({
                "aspect":     aspect,
                "sentiment":  "Neutral",
                "confidence": 0.0,
                "mentioned":  True,
                "snippet":    snippet,
            })

    # Sort: Negative first (most actionable), then Positive, then Neutral
    order = {"Negative": 0, "Positive": 1, "Neutral": 2}
    results.sort(key=lambda r: (order[r["sentiment"]], -r["confidence"]))

    return results


def analyse_aspects_batch(texts: list[str]) -> list[list[dict]]:
    """Run ABSA on a list of texts. Returns list of aspect-result lists."""
    return [analyse_aspects(t) for t in texts]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_snippet(text: str, keyword: str) -> str | None:
    """
    Extract the sentence (or up to ~90 chars) containing a keyword.
    Used for displaying context snippets in the UI.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        if keyword in sent.lower():
            sent = sent.strip()
            if len(sent) > 90:
                idx   = sent.lower().find(keyword)
                start = max(0, idx - 30)
                end   = min(len(sent), idx + 60)
                sent  = ("…" if start > 0 else "") + sent[start:end] + ("…" if end < len(sent) else "")
            return sent
    return None