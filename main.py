"""
main.py — Review Analyzer API v5.0
────────────────────────────────────────────────────────────────────────────────
Fixes vs v4:
  ─ /analyze_text now returns fake_reasons list for frontend explainability
  ─ confidence stored as 0.0–1.0 throughout (never pre-multiplied)
  ─ fake_score stored as 0.0–1.0 throughout
  ─ version bumped to 5.0.0
  ─ ABSA endpoint added (/absa) for single-text aspect analysis
  ─ insights fallback threshold lowered to 10% fake (matches frontend alert)
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import threading
import concurrent.futures
import time
import io
import torch
from typing import Any
import os
from collections import Counter
import re
import httpx

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from sentiment_model import predict_sentiment, predict_sentiment_batch
from product_detection import detect_product
from fake_review import detect_fake, detect_fake_explained
from Duplicatedetection import detect_duplicates


# ── REQUEST SCHEMA ────────────────────────────────────────────────────────────


class TextRequest(BaseModel):
    text: str

    class Config:
        str_strip_whitespace = True
        str_min_length = 1


# ── APP + CORS ────────────────────────────────────────────────────────────────


app = FastAPI(title="Review Analyzer API", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── CONSTANTS ─────────────────────────────────────────────────────────────────


MAX_ROWS = 50_000
MAX_MB   = 50

# Smaller batches on CPU keep each chunk short and progress updates frequent.
# On GPU, larger batches amortise kernel launch overhead.
_default_batch = 64 if torch.cuda.is_available() else 32
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", _default_batch))

# EMA smoothing factor — higher = faster to react to speed changes
EMA_ALPHA = 0.2

# 4 workers: sentiment, product, fake all run simultaneously
_model_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Use all available CPU threads for PyTorch operations
torch.set_num_threads(os.cpu_count() or 4)

print(f"[main] Device    : {'GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"[main] Batch size: {BATCH_SIZE}")
print(f"[main] CPU cores : {os.cpu_count()}")


# ── SHARED STATE ──────────────────────────────────────────────────────────────


_lock = threading.Lock()

_progress: dict[str, Any] = {
    "total":      0,
    "processed":  0,
    "start_time": None,
    "eta":        0,
    "speed":      0.0,
    "running":    False,
    "ema_speed":  None,
}

_results:   list[dict] = []
_file_name: str        = ""


# ── SAFE UNPACK ───────────────────────────────────────────────────────────────


def _safe_unpack(value: Any, fallback_label: str, fallback_score: float = 0.0) -> tuple:
    """
    Safely unpacks any function return into exactly (label, score).
    Prevents 'too many values to unpack' regardless of what the
    model function actually returns.
    """
    if isinstance(value, str):
        return value, fallback_score
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return fallback_label, fallback_score
        if len(value) == 1:
            return str(value[0]), fallback_score
        return value[0], value[1]
    return fallback_label, fallback_score


# ── BATCH WRAPPERS FOR RULE-BASED MODELS ─────────────────────────────────────


def _detect_product_batch(texts: list[str]) -> list[tuple]:
    """Run detect_product over a list and return list of (label, score) tuples."""
    return [_safe_unpack(detect_product(t), "General", 0) for t in texts]


def _detect_fake_batch(texts: list[str]) -> list[tuple]:
    """Run detect_fake over a list and return list of (label, score) tuples."""
    return [_safe_unpack(detect_fake(t), "Real", 0.0) for t in texts]


# ── PROGRESS HELPERS ──────────────────────────────────────────────────────────


def _reset_progress(total: int) -> None:
    with _lock:
        _progress.update({
            "total":      total,
            "processed":  0,
            "start_time": time.time(),
            "eta":        0,
            "speed":      0.0,
            "running":    True,
            "ema_speed":  None,
        })


def _update_progress(processed: int, total: int, batch_elapsed: float, batch_count: int) -> None:
    """
    EMA-based speed and ETA using per-batch timing.

    batch_elapsed  — wall-clock seconds this batch took
    batch_count    — number of reviews in this batch
    """
    raw_speed = batch_count / batch_elapsed if batch_elapsed > 0 else 1.0

    with _lock:
        if _progress["ema_speed"] is None:
            _progress["ema_speed"] = raw_speed
        else:
            _progress["ema_speed"] = (
                EMA_ALPHA * raw_speed +
                (1 - EMA_ALPHA) * _progress["ema_speed"]
            )

        ema_speed = _progress["ema_speed"]
        remaining = total - processed

        _progress["processed"] = processed
        _progress["speed"]     = round(ema_speed, 2)
        _progress["eta"]       = int(remaining / ema_speed) if ema_speed > 0 else 0


# ── ANALYSIS HELPERS ──────────────────────────────────────────────────────────


def _analyse_single(text: str) -> dict:
    """
    Single review — used by /analyze_text.
    Returns fake_reasons list so the frontend can show rule-by-rule explainability.
    confidence and fake_score are stored as 0.0–1.0 (raw fractions).
    """
    sentiment, sent_score = _safe_unpack(predict_sentiment(text), "Neutral", 0.0)
    emotion               = "Neutral"   # Emotion model disabled
    product, _            = _safe_unpack(detect_product(text), "General", 0)

    # Use the explained variant so the UI gets the reasons list
    fake_label, fake_score, fake_reasons = detect_fake_explained(text)

    return {
        "review":       text,
        "sentiment":    sentiment,
        "emotion":      emotion,
        "product":      product,
        "fake_review":  fake_label,
        "confidence":   round(float(sent_score), 4),   # 0.0–1.0
        "fake_score":   round(float(fake_score), 4),   # 0.0–1.0
        "fake_reasons": fake_reasons,                  # list[dict] for UI
    }


def _build_error_row(review: str) -> dict:
    return {
        "review":      review,
        "sentiment":   "Error",
        "emotion":     "Error",
        "product":     "Unknown",
        "fake_review": "Unknown",
        "confidence":  0.0,
        "fake_score":  0.0,
    }


# ── ROUTES ────────────────────────────────────────────────────────────────────


@app.get("/")
def home():
    return {
        "message":    "Review Analyzer API is running",
        "version":    "5.0.0",
        "device":     "GPU" if torch.cuda.is_available() else "CPU",
        "batch_size": BATCH_SIZE,
        "cpu_cores":  os.cpu_count(),
        "endpoints":  [
            "/analyze_text", "/upload_csv", "/progress",
            "/results", "/insights", "/duplicates", "/absa", "/debug",
        ],
    }


# ── SINGLE TEXT ───────────────────────────────────────────────────────────────


@app.post("/analyze_text")
async def analyze_text(request: TextRequest):
    """
    Analyse a single review text.
    Returns sentiment, emotion, product, fake label + score + reasons, confidence.
    All scores are 0.0–1.0 fractions — the frontend multiplies by 100 for display.
    """
    try:
        result = _analyse_single(request.text)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ── ABSA ENDPOINT ─────────────────────────────────────────────────────────────


@app.post("/absa")
async def absa_single(request: TextRequest):
    """
    Run Aspect-Based Sentiment Analysis on a single review.
    Returns a list of aspect dicts:
      { aspect, sentiment, confidence, mentioned, snippet }
    Lazy-loads the ABSA model on first call.
    """
    try:
        from Absamodel import analyse_aspects
        aspects = analyse_aspects(request.text)
        return {"aspects": aspects}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"ABSA failed: {str(e)}")


# ── CSV UPLOAD ────────────────────────────────────────────────────────────────


@app.post("/upload_csv")
async def upload_csv(
    file:   UploadFile = File(...),
    column: str        = Form(...)
):
    global _results, _file_name

    with _lock:
        if _progress["running"]:
            raise HTTPException(
                status_code=409,
                detail="A batch is already processing. Wait for it to finish."
            )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_MB} MB."
        )

    try:
        df = pd.read_csv(io.BytesIO(content), encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(io.BytesIO(content), encoding="latin1")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not parse CSV: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {str(e)}")

    if column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{column}' not found. Available: {list(df.columns)}"
        )

    if len(df) > MAX_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"CSV has {len(df):,} rows. Maximum: {MAX_ROWS:,}."
        )

    reviews = df[column].dropna().astype(str).str.strip().tolist()
    reviews = [r for r in reviews if r and r.lower() != "nan"]

    if not reviews:
        raise HTTPException(status_code=400, detail="No valid reviews found in that column.")

    _file_name = file.filename
    _results   = []

    thread = threading.Thread(target=_process_batch, args=(reviews,), daemon=True)
    thread.start()

    return {
        "message":       "Processing started",
        "file_name":     _file_name,
        "total_reviews": len(reviews),
    }


# ── BACKGROUND BATCH PROCESSING ───────────────────────────────────────────────


def _process_batch(reviews: list[str]) -> None:
    """
    Processes reviews in parallel batches:
      1. Sentiment model (transformer, batched)
      2. Product detection (rule-based, batched)
      3. Fake detection (rule-based, batched)
    All three run concurrently via ThreadPoolExecutor.
    Results written incrementally after each batch.
    confidence and fake_score stored as 0.0–1.0 fractions.
    """
    global _results

    total = len(reviews)
    _reset_progress(total)
    results = []

    try:
        for batch_start in range(0, total, BATCH_SIZE):
            batch = reviews[batch_start: batch_start + BATCH_SIZE]

            batch_t0 = time.time()

            # Submit all models in parallel
            future_sent    = _model_executor.submit(predict_sentiment_batch, batch)
            future_product = _model_executor.submit(_detect_product_batch,   batch)
            future_fake    = _model_executor.submit(_detect_fake_batch,      batch)

            # Collect with individual fallbacks
            try:
                raw_sentiments = future_sent.result(timeout=120)
            except Exception as e:
                print(f"[WARN] Sentiment batch {batch_start} failed: {e}")
                raw_sentiments = [("Error", 0.0)] * len(batch)

            raw_emotions = ["Neutral"] * len(batch)

            try:
                products = future_product.result(timeout=120)
            except Exception as e:
                print(f"[WARN] Product batch {batch_start} failed: {e}")
                products = [("General", 0)] * len(batch)

            try:
                fakes = future_fake.result(timeout=120)
            except Exception as e:
                print(f"[WARN] Fake batch {batch_start} failed: {e}")
                fakes = [("Real", 0.0)] * len(batch)

            # Assemble results
            batch_results = []
            for j, review in enumerate(batch):
                try:
                    product,   _          = products[j]
                    fake,      fake_score = fakes[j]
                    sentiment, sent_score = _safe_unpack(raw_sentiments[j], "Neutral", 0.0)
                    emotion               = raw_emotions[j]

                    batch_results.append({
                        "review":      review,
                        "sentiment":   sentiment,
                        "emotion":     emotion,
                        "product":     product,
                        "fake_review": fake,
                        "confidence":  round(float(sent_score), 4),  # 0.0–1.0
                        "fake_score":  round(float(fake_score), 4),  # 0.0–1.0
                    })

                except Exception as e:
                    print(f"[WARN] Row {batch_start + j} failed: {e}")
                    batch_results.append(_build_error_row(review))

            results.extend(batch_results)

            # Write incrementally so /results always returns fresh partial data
            with _lock:
                _results = results[:]

            batch_elapsed    = time.time() - batch_t0
            processed_so_far = min(batch_start + len(batch), total)
            _update_progress(processed_so_far, total, batch_elapsed, len(batch))

            print(
                f"[INFO] Batch {batch_start // BATCH_SIZE + 1} done — "
                f"{processed_so_far}/{total} reviews — "
                f"{round(len(batch) / batch_elapsed, 1)} reviews/sec"
            )

        elapsed = round(time.time() - _progress["start_time"], 1)
        print(f"[INFO] Batch complete — {total} reviews in {elapsed}s")

    except Exception as e:
        print(f"[ERROR] Batch processing crashed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        with _lock:
            _progress["running"]   = False
            _progress["processed"] = total


# ── PROGRESS ENDPOINT ─────────────────────────────────────────────────────────


@app.get("/progress")
def get_progress():
    """
    elapsed — real seconds since batch started (counts up)
    eta     — EMA-smoothed seconds remaining   (counts down)
    speed   — EMA-smoothed reviews/sec
    running — False when complete
    """
    with _lock:
        elapsed = 0.0
        if _progress["start_time"]:
            elapsed = round(time.time() - _progress["start_time"], 1)

        total     = _progress["total"]
        processed = _progress["processed"]
        percent   = round((processed / total * 100), 1) if total > 0 else 0.0

        return {
            "total":     total,
            "processed": processed,
            "percent":   percent,
            "elapsed":   elapsed,
            "eta":       _progress["eta"],
            "speed":     _progress["speed"],
            "running":   _progress["running"],
        }


# ── RESULTS ENDPOINT ──────────────────────────────────────────────────────────


@app.get("/results")
def get_results():
    """
    Returns partial results mid-batch, full results when done.
    Results are written after every batch so this is always fresh.
    confidence and fake_score are 0.0–1.0 — multiply ×100 in the UI.
    """
    return {
        "file_name": _file_name,
        "total":     len(_results),
        "results":   _results,
    }


# ── INSIGHTS ENDPOINT ─────────────────────────────────────────────────────────

_STOP_WORDS = {
    "the","and","for","are","but","not","you","all","can","had","her","was",
    "one","our","out","day","get","has","him","his","how","its","may","new",
    "now","old","see","two","way","who","boy","did","let","put","say","she",
    "too","use","this","that","with","from","have","very","well","after",
    "like","just","been","more","when","than","then","they","were","what",
    "will","your","also","each","much","over","such","into","only","other",
    "some","these","would","could","should","really","great","good","product",
    "item","order","bought","also","even","got","its","was",
}


def _top_keywords(texts: list[str], n: int = 8) -> list[str]:
    words: Counter = Counter()
    for t in texts:
        for w in re.findall(r"[a-z]{3,}", t.lower()):
            if w not in _STOP_WORDS:
                words[w] += 1
    return [w for w, _ in words.most_common(n)]


@app.get("/insights")
async def get_insights():
    """
    Aggregate stats from _results and call the Anthropic API to generate
    a plain-English executive summary.
    avg_conf is computed from 0.0–1.0 confidence values and shown as %.
    """
    with _lock:
        results = _results[:]

    if not results:
        raise HTTPException(status_code=404, detail="No results available yet.")

    total        = len(results)
    sent_counts  : Counter = Counter(r.get("sentiment", "Neutral") for r in results)
    fake_count   = sum(1 for r in results if r.get("fake_review") == "Fake")
    emotion_counts: Counter = Counter(r.get("emotion", "Unknown") for r in results)
    product_counts: Counter = Counter(r.get("product", "General") for r in results)

    # confidence stored as 0.0–1.0 → multiply by 100 for display
    avg_conf = round(
        sum(float(r.get("confidence", 0)) for r in results) / total * 100, 1
    )

    neg_texts  = [r["review"] for r in results if r.get("sentiment") == "Negative"]
    pos_texts  = [r["review"] for r in results if r.get("sentiment") == "Positive"]
    fake_texts = [r["review"] for r in results if r.get("fake_review") == "Fake"]

    neg_keywords  = _top_keywords(neg_texts,  8)
    pos_keywords  = _top_keywords(pos_texts,  8)
    fake_keywords = _top_keywords(fake_texts, 6)

    top_emotions = emotion_counts.most_common(3)
    top_products = product_counts.most_common(5)

    stats_block = f"""
Dataset summary ({total:,} reviews):
- Positive: {sent_counts['Positive']} ({sent_counts['Positive']/total*100:.1f}%)
- Neutral:  {sent_counts['Neutral']}  ({sent_counts['Neutral']/total*100:.1f}%)
- Negative: {sent_counts['Negative']} ({sent_counts['Negative']/total*100:.1f}%)
- Fake reviews flagged: {fake_count} ({fake_count/total*100:.1f}%)
- Average confidence: {avg_conf}%
- Top emotions: {', '.join(f"{e} ({c})" for e, c in top_emotions)}
- Top products: {', '.join(f"{p} ({c})" for p, c in top_products)}
- Top negative keywords: {', '.join(neg_keywords) or 'none'}
- Top positive keywords: {', '.join(pos_keywords) or 'none'}
- Top fake-review keywords: {', '.join(fake_keywords) or 'none'}
""".strip()

    prompt = f"""You are a concise business analyst. Given the following review dataset statistics, write a sharp executive summary in plain English.

{stats_block}

Rules:
- Write 4–6 punchy sentences (no bullet points, no headers).
- Lead with the most important insight.
- Mention specific percentages or numbers.
- Call out any red flags (high fake rate, high negativity).
- Reference top keywords naturally if they reveal a theme (e.g. "battery life" or "delivery").
- End with one actionable recommendation.
- Keep it under 120 words."""

    ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                ANTHROPIC_API_URL,
                headers={"Content-Type": "application/json"},
                json={
                    "model":      "claude-sonnet-4-20250514",
                    "max_tokens": 300,
                    "messages":   [{"role": "user", "content": prompt}],
                },
            )
        resp.raise_for_status()
        body    = resp.json()
        summary = body["content"][0]["text"].strip()

    except Exception as e:
        print(f"[WARN] Anthropic API call failed: {e}. Falling back to template.")
        pos_pct  = round(sent_counts["Positive"] / total * 100, 1)
        neg_pct  = round(sent_counts["Negative"] / total * 100, 1)
        fake_pct = round(fake_count / total * 100, 1)
        top_emo  = top_emotions[0][0] if top_emotions else "Neutral"
        neg_kw   = ", ".join(neg_keywords[:3]) if neg_keywords else "general quality"
        summary = (
            f"Analysis of {total:,} reviews shows {pos_pct}% positive and {neg_pct}% negative sentiment, "
            f"with an average confidence of {avg_conf}%. "
            f"The most common emotion is {top_emo}. "
            f"{fake_pct}% of reviews were flagged as potentially fake"
            + (f" — above the 10% threshold, which warrants attention." if fake_pct >= 10 else ".") +
            f" Negative reviews frequently mention themes around: {neg_kw}. "
            f"Consider addressing these pain points to improve overall satisfaction."
        )

    return {
        "summary": summary,
        "stats": {
            "total":        total,
            "positive":     sent_counts["Positive"],
            "negative":     sent_counts["Negative"],
            "neutral":      sent_counts["Neutral"],
            "fake":         fake_count,
            "avg_conf":     avg_conf,
            "neg_keywords": neg_keywords,
            "pos_keywords": pos_keywords,
        },
    }


# ── DUPLICATES ENDPOINT ───────────────────────────────────────────────────────


@app.get("/duplicates")
def get_duplicates(
    near_threshold: float = 0.80,
    sem_threshold:  float = 0.85,
):
    """
    Run 3-stage duplicate detection over the current results:
      Stage 1 — Exact match      (MD5 hash, O(n))
      Stage 2 — Near-duplicate   (Jaccard on char 3-shingles, ≥ near_threshold)
      Stage 3 — Semantic clone   (TF-IDF cosine similarity, ≥ sem_threshold)

    Query params:
      near_threshold  float 0-1  (default 0.80)
      sem_threshold   float 0-1  (default 0.85)
    """
    with _lock:
        results = _results[:]

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No results available yet. Run a batch analysis first."
        )

    reviews = [r.get("review", "") for r in results]
    report  = detect_duplicates(
        reviews,
        near_threshold=near_threshold,
        sem_threshold=sem_threshold,
    )
    return report


# ── DEBUG ENDPOINT ────────────────────────────────────────────────────────────


@app.get("/debug")
def debug():
    results = {}
    test = "This product is amazing and works perfectly!"

    try:
        results["sentiment"] = str(predict_sentiment(test))
    except Exception as e:
        results["sentiment"] = f"FAILED: {e}"

    results["emotion"] = "Neutral"  # Emotion model disabled

    try:
        results["product"] = str(detect_product(test))
    except Exception as e:
        results["product"] = f"FAILED: {e}"

    try:
        results["fake"] = str(detect_fake(test))
    except Exception as e:
        results["fake"] = f"FAILED: {e}"

    try:
        results["product_batch"] = str(_detect_product_batch([test]))
    except Exception as e:
        results["product_batch"] = f"FAILED: {e}"

    try:
        results["fake_batch"] = str(_detect_fake_batch([test]))
    except Exception as e:
        results["fake_batch"] = f"FAILED: {e}"

    return results