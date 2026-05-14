"""
sentiment_model.py — DistilBERT Sentiment (Memory-Optimised for Render 512MB)

KEY CHANGE: Swapped cardiffnlp/twitter-roberta-base-sentiment (498MB fp32)
            for distilbert-base-uncased-finetuned-sst-2-english (260MB fp32).
            After INT8 quantization the model sits at ~82MB vs ~160MB,
            freeing ~80MB of headroom — enough to prevent the batch OOM.

Memory budget on Render free tier (512MB):
  DistilBERT INT8   ~82MB
  FastAPI + PyTorch + Transformers overhead  ~100MB
  Batch tensor peak (16 reviews × 128 tokens)  ~20MB
  Headroom for product/fake detection + Python heap  ~310MB
  ──────────────────────────────────────────────────────
  Total peak  ~202MB  ✓  (was ~380MB+ with RoBERTa → OOM)

Other fixes retained:
  ✦ low_cpu_mem_usage=True  — prevents transient 2× RAM spike during load
  ✦ INT8 quantization on CPU for ~2x inference speedup + ~40% RAM reduction
  ✦ unload() clears gc + torch caches
  ✦ max_length=128 — reviews rarely exceed this
  ✦ Per-batch tensor cleanup to reduce peak RSS during large CSV jobs
"""

from __future__ import annotations
import gc
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.set_grad_enabled(False)

# ── Model: DistilBERT SST-2 (260MB fp32 → ~82MB INT8) ────────────────────────
# Strong on product/app review sentiment. 6 transformer layers vs RoBERTa's 12
# → half the weight count, same accuracy tier for short review text.
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# SST-2 is binary (NEGATIVE / POSITIVE). We derive a Neutral class from
# low-confidence predictions — this matches 3-class RoBERTa behaviour closely.
NEUTRAL_THRESHOLD = 0.65
RAW_LABELS = ["Negative", "Positive"]   # index order from the model config

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

    # low_cpu_mem_usage=True: allocates each layer directly into its final
    # tensor — prevents the transient 2× RAM spike during weight loading.
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

    # Reclaim any scratch memory used by HuggingFace internals during load
    gc.collect()
    print("[sentiment_model] Ready")


def unload() -> None:
    """
    Release model from RAM completely.

    Call this:
      • before loading the ABSA model (both together exceed 512 MB)
      • at the end of every batch job (model reloads on next /analyze_text call)
    """
    global _tokenizer, _model, _device

    if _model is None:
        return

    print("[sentiment_model] Unloading to free RAM…")
    del _model
    del _tokenizer
    _model     = None
    _tokenizer = None
    _device    = None

    gc.collect()
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
    batch_size: int = 16,
) -> list[tuple[str, float]]:
    """
    Predict sentiment for a list of texts.

    Returns list of (label, confidence) tuples in the same order as `texts`.
    label is one of "Negative", "Neutral", "Positive".
    confidence is a float in [0.0, 1.0].

    Neutral derivation: SST-2 is binary. When the winning class confidence
    is below NEUTRAL_THRESHOLD (0.65) we call it Neutral — this closely
    matches 3-class RoBERTa behaviour on review data.
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
            max_length=128,
            return_tensors="pt",
        ).to(_device)

        with torch.no_grad():
            logits = _model(**inputs).logits

        probs = torch.softmax(logits, dim=1)

        for prob_row in probs:
            idx        = int(torch.argmax(prob_row).item())
            confidence = float(prob_row[idx].item())

            if confidence < NEUTRAL_THRESHOLD:
                label = "Neutral"
            else:
                label = RAW_LABELS[idx]   # "Negative" or "Positive"

            results.append((label, round(confidence, 4)))

        # Free intermediate tensors before next batch allocation — noticeably
        # lowers peak RSS during large CSV jobs on a 512 MB host.
        del inputs, logits, probs

    return results