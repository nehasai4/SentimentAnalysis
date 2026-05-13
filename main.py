"""
main.py — Review Analyzer API v6.1
────────────────────────────────────────────────────────────────────────────────
Memory fixes vs v6.0 (targeting Render free tier 512MB limit):

  ✦ _process_batch now calls _unload_sentiment() in its `finally` block so
    the RoBERTa model is freed from RAM as soon as a CSV job completes.
    The model reloads automatically on the next /analyze_text request.

  ✦ /absa endpoint guards against concurrent batch jobs (both models in RAM
    simultaneously would exceed 512MB) and unloads ABSA when done so the
    sentiment model can reload for the next batch.

  ✦ BATCH_SIZE env-var defaults to 16 on CPU (was 32) — halves the peak
    tensor memory during tokenization without meaningfully hurting throughput.

  ✦ torch.set_num_threads capped at min(cpu_count, 4) — over-threading on
    Render's shared CPUs wastes RAM on thread stacks and hurts latency.

Everything else (rate limiting, CORS, progress, insights, duplicates) is
unchanged from v6.0.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import pandas as pd
import threading
import time
import io
import torch
from typing import Any
import os
from collections import Counter
import re
import httpx

from sentiment_model import predict_sentiment, predict_sentiment_batch, unload as _unload_sentiment
from product_detection import detect_product
from fake_review import detect_fake, detect_fake_explained
from Duplicatedetection import detect_duplicates


# ── REQUEST SCHEMA ────────────────────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str

    class Config:
        str_strip_whitespace = True
        str_min_length = 1


# ── RATE LIMITER ──────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


# ── APP + CORS ────────────────────────────────────────────────────────────────

_ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")

app = FastAPI(title="Review Analyzer API", version="6.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[_ALLOWED_ORIGIN] if _ALLOWED_ORIGIN != "*" else ["*"],
    allow_credentials=_ALLOWED_ORIGIN != "*",
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

MAX_ROWS = 50_000
MAX_MB   = 50

# ── MEMORY FIX: default batch size reduced from 32 → 16 on CPU ───────────────
# Each batch allocates tokenizer tensors proportional to batch_size × seq_len.
# 16 reviews × 128 tokens is much cheaper than 32 × 128 with no meaningful
# throughput loss since the bottleneck is model inference, not data loading.
_default_batch = 64 if torch.cuda.is_available() else 16
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", _default_batch))

EMA_ALPHA = 0.2

# Serialized batch processing — no ThreadPoolExecutor needed

# ── MEMORY FIX: cap PyTorch threads ──────────────────────────────────────────
# On Render's shared CPUs, spawning too many threads wastes RAM on stacks and
# causes context-switch overhead.  4 is the sweet spot for free-tier instances.
_num_threads = min(os.cpu_count() or 4, 4)
torch.set_num_threads(_num_threads)

print(f"[main] Version   : 6.1.0")
print(f"[main] Device    : {'GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"[main] Batch size: {BATCH_SIZE}")
print(f"[main] CPU cores : {os.cpu_count()}  (torch threads: {_num_threads})")


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
    return [_safe_unpack(detect_product(t), "General", 0) for t in texts]


def _detect_fake_batch(texts: list[str]) -> list[tuple]:
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
    sentiment, sent_score = _safe_unpack(predict_sentiment(text), "Neutral", 0.0)
    emotion               = "Neutral"
    product, _            = _safe_unpack(detect_product(text), "General", 0)
    fake_label, fake_score, fake_reasons = detect_fake_explained(text)

    return {
        "review":       text,
        "sentiment":    sentiment,
        "emotion":      emotion,
        "product":      product,
        "fake_review":  fake_label,
        "confidence":   round(float(sent_score), 4),
        "fake_score":   round(float(fake_score), 4),
        "fake_reasons": fake_reasons,
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
        "version":    "6.1.0",
        "device":     "GPU" if torch.cuda.is_available() else "CPU",
        "batch_size": BATCH_SIZE,
        "cpu_cores":  os.cpu_count(),
        "endpoints":  [
            "/health", "/analyze_text", "/upload_csv", "/progress",
            "/results", "/insights", "/duplicates", "/absa", "/debug",
        ],
    }


@app.get("/health")
def health():
    """
    Lightweight warmup endpoint — returns 200 immediately with no model calls.
    Reports RAM usage so memory issues can be spotted in Render logs.
    """
    mem_mb = None
    try:
        import psutil as _psutil, os as _os2
        mem_mb = round(_psutil.Process(_os2.getpid()).memory_info().rss / 1024 / 1024, 1)
    except Exception:
        pass

    return {
        "status":  "ok",
        "version": "6.1.0",
        "device":  "GPU" if torch.cuda.is_available() else "CPU",
        "ram_mb":  mem_mb,
    }


# ── SINGLE TEXT ───────────────────────────────────────────────────────────────

@app.post("/analyze_text")
@limiter.limit("10/minute")
async def analyze_text(request: Request, req: TextRequest):
    try:
        result = _analyse_single(req.text)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ── ABSA ENDPOINT ─────────────────────────────────────────────────────────────

@app.post("/absa")
@limiter.limit("10/minute")
async def absa_single(request: Request, req: TextRequest):
    """
    Run Aspect-Based Sentiment Analysis on a single review.

    Memory safety:
      1. Reject if a batch job is already running (both models in RAM = OOM).
      2. Unload the sentiment model before loading ABSA.
      3. Unload ABSA after the call so sentiment can reload for the next request.
    """
    # ── MEMORY FIX: block ABSA during batch jobs ──────────────────────────────
    with _lock:
        if _progress["running"]:
            raise HTTPException(
                status_code=503,
                detail=(
                    "A batch analysis is currently running. "
                    "ABSA is unavailable during batch processing to prevent "
                    "out-of-memory errors. Please try again once the batch completes."
                ),
            )

    try:
        from Absamodel import analyse_aspects, unload as _unload_absa

        # Free sentiment model first — both models together exceed 512MB
        _unload_sentiment()

        aspects = analyse_aspects(req.text)
        return {"aspects": aspects}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"ABSA failed: {str(e)}")

    finally:
        # ── MEMORY FIX: always unload ABSA after the call ─────────────────────
        # This ensures RAM is freed even if the call raised an exception,
        # so the next request (likely a batch job) can reload sentiment.
        try:
            from Absamodel import unload as _unload_absa
            _unload_absa()
        except Exception:
            pass


# ── CSV UPLOAD ────────────────────────────────────────────────────────────────

@app.post("/upload_csv")
@limiter.limit("2/minute")
async def upload_csv(
    request: Request,
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
    Processes reviews in sequential batches:
      1. Sentiment model (DistilBERT INT8, batched)
      2. Product detection (rule-based, sequential — no RAM cost)
      3. Fake detection (rule-based, sequential — no RAM cost)

    Sequential instead of parallel: running all three concurrently caused
    peak RSS spikes that OOM-killed Render free-tier (512MB).
    Product/fake are pure-Python; serializing them costs <1ms per batch.
    Calls _unload_sentiment() in the finally block to free RAM after the job.
    """
    global _results

    total = len(reviews)
    _reset_progress(total)
    results = []

    try:
        for batch_start in range(0, total, BATCH_SIZE):
            batch = reviews[batch_start: batch_start + BATCH_SIZE]

            batch_t0 = time.time()

            # ── MEMORY FIX: run sequentially, not concurrently ────────────────────
            # Running sentiment + product + fake in 3 threads simultaneously
            # caused peak RSS spikes that OOM-killed the Render free-tier instance.
            # Product and fake detection are pure-Python rule-based (no tensors),
            # so serializing them costs <1ms per batch — the throughput impact is
            # negligible compared to the transformer forward pass.
            try:
                raw_sentiments = predict_sentiment_batch(batch)
            except Exception as e:
                print(f"[WARN] Sentiment batch {batch_start} failed: {e}")
                raw_sentiments = [("Error", 0.0)] * len(batch)

            raw_emotions = ["Neutral"] * len(batch)

            try:
                products = _detect_product_batch(batch)
            except Exception as e:
                print(f"[WARN] Product batch {batch_start} failed: {e}")
                products = [("General", 0)] * len(batch)

            try:
                fakes = _detect_fake_batch(batch)
            except Exception as e:
                print(f"[WARN] Fake batch {batch_start} failed: {e}")
                fakes = [("Real", 0.0)] * len(batch)

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
                        "confidence":  round(float(sent_score), 4),
                        "fake_score":  round(float(fake_score), 4),
                    })

                except Exception as e:
                    print(f"[WARN] Row {batch_start + j} failed: {e}")
                    batch_results.append(_build_error_row(review))

            results.extend(batch_results)

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

        # ── MEMORY FIX: unload sentiment model after every batch ──────────────
        # Frees ~125MB of RoBERTa weights + quantization overhead.
        # The model reloads automatically on the next /analyze_text call.
        # Without this, the model stays resident between batches and leaves
        # no headroom for ABSA or other allocations.
        _unload_sentiment()
        print("[INFO] Sentiment model unloaded after batch — RAM freed.")


# ── PROGRESS ENDPOINT ─────────────────────────────────────────────────────────

@app.get("/progress")
def get_progress():
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
    with _lock:
        results = _results[:]

    if not results:
        raise HTTPException(status_code=404, detail="No results available yet.")

    total         = len(results)
    sent_counts   : Counter = Counter(r.get("sentiment", "Neutral") for r in results)
    fake_count    = sum(1 for r in results if r.get("fake_review") == "Fake")
    emotion_counts: Counter = Counter(r.get("emotion", "Unknown") for r in results)
    product_counts: Counter = Counter(r.get("product", "General") for r in results)

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
                headers={
                    "Content-Type":      "application/json",
                    "x-api-key":         os.environ.get("ANTHROPIC_API_KEY", ""),
                    "anthropic-version": "2023-06-01",
                },
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

    results["emotion"] = "Neutral"

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