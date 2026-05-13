"""
sentiment_model.py — RoBERTa Sentiment (Memory-Optimised for Render 512MB)

Changes vs previous version:
  ✦ low_cpu_mem_usage=True  — prevents the transient RAM doubling during load
    (HuggingFace used to hold file bytes + model weights simultaneously)
  ✦ INT8 quantization still applied on CPU for ~2x inference speedup
  ✦ unload() clears gc AND torch internal caches (torch.cuda.empty_cache
    is a no-op on CPU but kept for GPU compatibility)
  ✦ predict_sentiment_batch accepts an explicit batch_size param so
    main.py can tune it via the BATCH_SIZE env-var without touching this file
  ✦ max_length stays at 128 — reviews rarely exceed this; saves ~16x
    attention computation vs the old 512 default
"""

from __future__ import annotations
import gc
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.set_grad_enabled(False)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
LABELS     = ["Negative", "Neutral", "Positive"]

_tokenizer = None
_model     = None
_device    = None


def _load() -> None:
    global _tokenizer, _model, _device

    if _model is not None:
        return

    print(f"[sentiment_model] Loading {MODEL_NAME}…")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── KEY FIX: low_cpu_mem_usage=True ──────────────────────────────────────
    # Without this flag HuggingFace allocates a full copy of the weights in
    # Python bytes, THEN copies them into the model — temporarily using 2×
    # the model's RAM during load.  This flag allocates each layer directly
    # into the final tensor, keeping peak RAM at ~1× instead of ~2×.
    _model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True,
    )

    # INT8 dynamic quantization — ~40% smaller RAM, ~2× faster on CPU.
    # Must happen before .to(_device) when targeting CPU.
    if _device.type == "cpu":
        try:
            _model = torch.quantization.quantize_dynamic(
                _model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            print("[sentiment_model] INT8 quantization applied ✓")
        except Exception as e:
            print(f"[sentiment_model] Quantization skipped: {e}")

    _model.to(_device)
    _model.eval()
    print("[sentiment_model] Ready")


def unload() -> None:
    """
    Release model from RAM completely.

    Call this:
      • before loading the ABSA model (both together exceed 512 MB)
      • at the end of every batch job (model reloads on next request)

    The next predict_sentiment / predict_sentiment_batch call will
    reload the model automatically.
    """
    global _tokenizer, _model, _device

    if _model is None:
        return  # already unloaded — nothing to do

    print("[sentiment_model] Unloading to free RAM…")
    del _model
    del _tokenizer
    _model     = None
    _tokenizer = None
    _device    = None

    gc.collect()
    # no-op on CPU but frees GPU VRAM if ever run on GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[sentiment_model] Unloaded ✓")


def predict_sentiment(text: str) -> tuple[str, float]:
    """Predict sentiment for a single text. Returns (label, confidence)."""
    if not text or not text.strip():
        return ("Neutral", 0.0)
    results = predict_sentiment_batch([text])
    return results[0] if results else ("Neutral", 0.0)


def predict_sentiment_batch(
    texts: list[str],
    batch_size: int = 32,
) -> list[tuple[str, float]]:
    """
    Predict sentiment for a list of texts.

    Args:
        texts:      list of review strings
        batch_size: number of reviews per forward pass (tune via BATCH_SIZE
                    env-var in main.py; lower = less RAM per pass)

    Returns:
        list of (label, confidence) tuples in the same order as `texts`.
        label is one of "Negative", "Neutral", "Positive".
        confidence is a float in [0.0, 1.0].
    """
    if not texts:
        return []

    _load()

    results: list[tuple[str, float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        inputs = _tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,         # reviews rarely exceed 128 tokens
            return_tensors="pt",
        ).to(_device)

        with torch.no_grad():
            logits = _model(**inputs).logits

        probs = torch.softmax(logits, dim=1)

        for prob_row in probs:
            idx        = int(torch.argmax(prob_row).item())
            label      = LABELS[idx]
            confidence = round(float(prob_row[idx].item()), 4)
            results.append((label, confidence))

        # ── Per-batch memory hygiene ──────────────────────────────────────────
        # Delete intermediate tensors so they're eligible for GC before the
        # next batch allocation. On a 512 MB host this noticeably lowers the
        # peak RSS during large CSV jobs.
        del inputs, logits, probs

    return results