"""
main.py — Review Analyzer API v6.0  (Production-Ready)
────────────────────────────────────────────────────────────────────────────────
New in v6.0 vs v5.0:
  ─ /reset endpoint  → clears stuck running=True state (fixes 409 on Render)
  ─ /upload_csv now accepts ?force=true to auto-reset before starting
  ─ Stale-job watchdog: running jobs older than JOB_TIMEOUT_SEC auto-expire
  ─ Optional API key auth via X-API-Key header (set API_SECRET_KEY env var)
  ─ /health endpoint with memory + model status for uptime monitors
  ─ Memory guard: refuses batch if free RAM < MIN_FREE_MB
  ─ Improved fake-review scoring: 3 new research-backed rules added
    (first-person pronoun density, superlative density, vague temporal language)
  ─ /status endpoint: returns current job status without full progress payload
  ─ Graceful CORS: restrict origins via ALLOWED_ORIGINS env var in production
  ─ version bumped to 6.0.0
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import concurrent.futures
import io
import logging
import os
import re
import threading
import time
import traceback
from collections import Counter
from typing import Any

import httpx
import pandas as pd
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Offline mode for HuggingFace (no outbound calls during inference) ─────────
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"

from sentiment_model import predict_sentiment, predict_sentiment_batch
from product_detection import detect_product, detect_product_full
from fake_review import detect_fake, detect_fake_explained
from Duplicatedetection import detect_duplicates

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("review_api")

# ── Environment config ────────────────────────────────────────────────────────
API_SECRET_KEY  = os.environ.get("API_SECRET_KEY", "")          # empty = auth disabled
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
JOB_TIMEOUT_SEC = int(os.environ.get("JOB_TIMEOUT_SEC", "600")) # 10 min stale-job TTL
MIN_FREE_MB     = int(os.environ.get("MIN_FREE_MB", "200"))      # refuse batch below this
MAX_ROWS        = int(os.environ.get("MAX_ROWS", "50000"))
MAX_MB          = int(os.environ.get("MAX_FILE_MB", "50"))

_default_batch  = 64 if torch.cuda.is_available() else 32
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE", _default_batch))
EMA_ALPHA       = 0.2
DEDUP_TIMEOUT   = 90

torch.set_num_threads(os.cpu_count() or 4)
_model_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

log.info("Device    : %s", "GPU" if torch.cuda.is_available() else "CPU")
log.info("Batch size: %d", BATCH_SIZE)
log.info("CPU cores : %s", os.cpu_count())
log.info("Auth      : %s", "enabled" if API_SECRET_KEY else "disabled")
log.info("Origins   : %s", ALLOWED_ORIGINS)


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class TextRequest(BaseModel):
    text: str

    model_config = {"str_strip_whitespace": True}


# ══════════════════════════════════════════════════════════════════════════════
# APP + MIDDLEWARE
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Review Analyzer API",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONAL API KEY AUTH
# ══════════════════════════════════════════════════════════════════════════════

def _verify_api_key(x_api_key: str = Header(default="")) -> None:
    """
    If API_SECRET_KEY env var is set, every mutating endpoint requires the
    X-API-Key header to match. GET endpoints (health, progress, results) are
    always public so uptime monitors work without credentials.
    """
    if API_SECRET_KEY and x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")

_auth = [Depends(_verify_api_key)]


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY GUARD
# ══════════════════════════════════════════════════════════════════════════════

def _check_memory() -> None:
    """Raise 503 if free RAM is below MIN_FREE_MB. Prevents OOM on Render free tier."""
    try:
        import psutil
        free_mb = psutil.virtual_memory().available / 1024 / 1024
        if free_mb < MIN_FREE_MB:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Server memory too low ({free_mb:.0f} MB free, "
                    f"need ≥ {MIN_FREE_MB} MB). Try again in a moment."
                ),
            )
    except ImportError:
        pass  # psutil not installed — skip guard


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE
# ══════════════════════════════════════════════════════════════════════════════

_lock = threading.Lock()

_progress: dict[str, Any] = {
    "total":      0,
    "processed":  0,
    "start_time": None,
    "eta":        0,
    "speed":      0.0,
    "running":    False,
    "ema_speed":  None,
    "error":      None,   # set if batch crashes
}

_results:   list[dict] = []
_file_name: str        = ""


def _is_stale() -> bool:
    """Return True if a job has been 'running' for longer than JOB_TIMEOUT_SEC."""
    with _lock:
        if not _progress["running"]:
            return False
        start = _progress.get("start_time")
        if start is None:
            return False
        return (time.time() - start) > JOB_TIMEOUT_SEC


def _force_reset() -> None:
    """Hard-reset all shared state. Call under _lock or from /reset endpoint."""
    global _results, _file_name
    with _lock:
        _progress.update({
            "total":      0,
            "processed":  0,
            "start_time": None,
            "eta":        0,
            "speed":      0.0,
            "running":    False,
            "ema_speed":  None,
            "error":      None,
        })
        _results   = []
        _file_name = ""


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — safe unpack, batch wrappers, progress
# ══════════════════════════════════════════════════════════════════════════════

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


def _detect_product_batch(texts: list[str]) -> list[tuple]:
    return [(detect_product_full(t).category, detect_product_full(t).sub_category) for t in texts]


def _detect_fake_batch(texts: list[str]) -> list[tuple]:
    return [_safe_unpack(detect_fake(t), "Real", 0.0) for t in texts]


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
            "error":      None,
        })


def _update_progress(processed: int, total: int, batch_elapsed: float, batch_count: int) -> None:
    raw_speed = batch_count / batch_elapsed if batch_elapsed > 0 else 1.0
    with _lock:
        prev = _progress["ema_speed"]
        _progress["ema_speed"] = (
            raw_speed if prev is None
            else EMA_ALPHA * raw_speed + (1 - EMA_ALPHA) * prev
        )
        ema   = _progress["ema_speed"]
        rem   = total - processed
        _progress["processed"] = processed
        _progress["speed"]     = round(ema, 2)
        _progress["eta"]       = int(rem / ema) if ema > 0 else 0


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _analyse_single(text: str) -> dict:
    """Single review analysis — used by /analyze_text."""
    sentiment, sent_score = _safe_unpack(predict_sentiment(text), "Neutral", 0.0)
    prod_result           = detect_product_full(text)
    fake_label, fake_score, fake_reasons = detect_fake_explained(text)
    return {
        "review":       text,
        "sentiment":    sentiment,
        "emotion":      "Neutral",
        "product":      prod_result.category,
        "sub_category": prod_result.sub_category,
        "fake_review":  fake_label,
        "confidence":   round(float(sent_score), 4),
        "fake_score":   round(float(fake_score), 4),
        "fake_reasons": fake_reasons,
    }


def _build_error_row(review: str) -> dict:
    return {
        "review":       review,
        "sentiment":    "Error",
        "emotion":      "Error",
        "product":      "Unknown",
        "sub_category": "Unknown",
        "fake_review":  "Unknown",
        "confidence":   0.0,
        "fake_score":   0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND BATCH WORKER
# ══════════════════════════════════════════════════════════════════════════════

def _process_batch(reviews: list[str]) -> None:
    global _results

    total = len(reviews)
    _reset_progress(total)
    results: list[dict] = []

    try:
        for batch_start in range(0, total, BATCH_SIZE):
            batch    = reviews[batch_start: batch_start + BATCH_SIZE]
            batch_t0 = time.time()

            future_sent    = _model_executor.submit(predict_sentiment_batch, batch)
            future_product = _model_executor.submit(_detect_product_batch,   batch)
            future_fake    = _model_executor.submit(_detect_fake_batch,      batch)

            try:
                raw_sentiments = future_sent.result(timeout=120)
            except Exception as exc:
                log.warning("Sentiment batch %d failed: %s", batch_start, exc)
                raw_sentiments = [("Error", 0.0)] * len(batch)

            try:
                products = future_product.result(timeout=120)
            except Exception as exc:
                log.warning("Product batch %d failed: %s", batch_start, exc)
                products = [("General", "General")] * len(batch)

            try:
                fakes = future_fake.result(timeout=120)
            except Exception as exc:
                log.warning("Fake batch %d failed: %s", batch_start, exc)
                fakes = [("Real", 0.0)] * len(batch)

            batch_results: list[dict] = []
            for j, review in enumerate(batch):
                try:
                    product_cat, product_sub = products[j]
                    fake,        fake_score  = fakes[j]
                    sentiment,   sent_score  = _safe_unpack(raw_sentiments[j], "Neutral", 0.0)
                    batch_results.append({
                        "review":       review,
                        "sentiment":    sentiment,
                        "emotion":      "Neutral",
                        "product":      product_cat,
                        "sub_category": product_sub,
                        "fake_review":  fake,
                        "confidence":   round(float(sent_score), 4),
                        "fake_score":   round(float(fake_score), 4),
                    })
                except Exception as exc:
                    log.warning("Row %d failed: %s", batch_start + j, exc)
                    batch_results.append(_build_error_row(review))

            results.extend(batch_results)

            with _lock:
                _results = results[:]

            batch_elapsed    = max(time.time() - batch_t0, 1e-6)
            processed_so_far = min(batch_start + len(batch), total)
            _update_progress(processed_so_far, total, batch_elapsed, len(batch))

            log.info(
                "Batch %d done — %d/%d — %.1f rev/s",
                batch_start // BATCH_SIZE + 1,
                processed_so_far, total,
                len(batch) / batch_elapsed,
            )

        elapsed = round(time.time() - _progress["start_time"], 1)
        log.info("Batch complete — %d reviews in %.1fs", total, elapsed)

    except Exception as exc:
        log.error("Batch crashed: %s", exc)
        traceback.print_exc()
        with _lock:
            _progress["error"] = str(exc)

    finally:
        with _lock:
            _progress["running"]   = False
            _progress["processed"] = total


# ══════════════════════════════════════════════════════════════════════════════
# KEYWORD HELPERS  (used by /insights)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "message":    "Review Analyzer API is running",
        "version":    "6.0.0",
        "device":     "GPU" if torch.cuda.is_available() else "CPU",
        "batch_size": BATCH_SIZE,
        "cpu_cores":  os.cpu_count(),
        "auth":       bool(API_SECRET_KEY),
        "endpoints":  [
            "/analyze_text", "/upload_csv", "/progress", "/status",
            "/results", "/insights", "/duplicates", "/absa",
            "/reset", "/health", "/debug",
        ],
    }


# ── Health  (public — used by Render uptime checks & frontend backend_ok()) ───

@app.get("/health")
def health():
    """
    Lightweight health check for uptime monitors.
    Returns 200 + memory stats + current job status.
    Never requires auth so Render's health-check probe always works.
    """
    mem_info: dict[str, Any] = {}
    try:
        import psutil
        vm = psutil.virtual_memory()
        mem_info = {
            "total_mb":     round(vm.total / 1024 / 1024),
            "available_mb": round(vm.available / 1024 / 1024),
            "used_pct":     vm.percent,
        }
    except ImportError:
        mem_info = {"note": "psutil not installed"}

    with _lock:
        running   = _progress["running"]
        processed = _progress["processed"]
        total_j   = _progress["total"]

    return {
        "status":    "ok",
        "version":   "6.0.0",
        "memory":    mem_info,
        "job": {
            "running":   running,
            "processed": processed,
            "total":     total_j,
            "stale":     _is_stale(),
        },
    }


# ── Status (public lightweight poll — cheaper than /progress) ─────────────────

@app.get("/status")
def get_status():
    """Minimal job status: is a batch running and is it stale?"""
    with _lock:
        return {
            "running":   _progress["running"],
            "stale":     _is_stale(),
            "processed": _progress["processed"],
            "total":     _progress["total"],
            "error":     _progress.get("error"),
        }


# ── Reset  (protected — clears stuck running=True, fixes 409 on Render) ───────

@app.post("/reset", dependencies=_auth)
def reset_state(clear_results: bool = False):
    """
    Force-reset the job state.

    - Always clears the running flag, progress counters, and any error.
    - Pass ?clear_results=true to also wipe the results list.

    Use this when the frontend shows a stuck 409 error.
    The frontend auto-calls this before every new upload.
    """
    was_running = _progress.get("running", False)
    stale       = _is_stale()

    global _results, _file_name
    with _lock:
        _progress.update({
            "total":      0 if clear_results else _progress["total"],
            "processed":  0 if clear_results else _progress["processed"],
            "start_time": None,
            "eta":        0,
            "speed":      0.0,
            "running":    False,
            "ema_speed":  None,
            "error":      None,
        })
        if clear_results:
            _results   = []
            _file_name = ""

    log.info(
        "State reset — was_running=%s stale=%s clear_results=%s",
        was_running, stale, clear_results,
    )
    return {
        "message":       "State reset successfully.",
        "was_running":   was_running,
        "was_stale":     stale,
        "results_cleared": clear_results,
    }


# ── Single text analysis ───────────────────────────────────────────────────────

@app.post("/analyze_text", dependencies=_auth)
async def analyze_text(request: TextRequest):
    """
    Analyse a single review text.
    Returns sentiment, emotion, product, fake label + score + reasons, confidence.
    All scores are 0.0–1.0 fractions — the frontend multiplies by 100 for display.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=422, detail="Review text cannot be empty.")
    try:
        return _analyse_single(request.text)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")


# ── ABSA ──────────────────────────────────────────────────────────────────────

@app.post("/absa", dependencies=_auth)
async def absa_single(request: TextRequest):
    """
    Run Aspect-Based Sentiment Analysis on a single review.
    Returns a list of aspect dicts: { aspect, sentiment, confidence, mentioned, snippet }
    Lazy-loads the ABSA model on first call.
    """
    try:
        from Absamodel import analyse_aspects
        aspects = analyse_aspects(request.text)
        return {"aspects": aspects}
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"ABSA failed: {exc}")


# ── CSV upload ────────────────────────────────────────────────────────────────

@app.post("/upload_csv", dependencies=_auth)
async def upload_csv(
    file:   UploadFile = File(...),
    column: str        = Form(...),
    force:  bool       = Form(False),   # if True, auto-reset any stuck job first
):
    """
    Upload a CSV file and start batch analysis.

    Pass force=true (or force=True as a form field) to automatically reset
    any stuck running job before starting. This is the recommended default
    for the frontend to avoid manual /reset calls.
    """
    global _results, _file_name

    # ── Auto-reset stale jobs ──────────────────────────────────────────────
    if _is_stale():
        log.warning("Stale job detected (> %ds) — auto-resetting.", JOB_TIMEOUT_SEC)
        _force_reset()

    # ── Honour force flag ─────────────────────────────────────────────────
    if force:
        log.info("force=True — resetting state before upload.")
        _force_reset()

    # ── Guard: refuse if another job is genuinely running ──────────────────
    with _lock:
        if _progress["running"]:
            raise HTTPException(
                status_code=409,
                detail=(
                    "A batch is already processing. "
                    "Wait for it to finish, or POST to /reset first, "
                    "or re-upload with force=true."
                ),
            )

    # ── Memory guard ──────────────────────────────────────────────────────
    _check_memory()

    # ── File size check ───────────────────────────────────────────────────
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum: {MAX_MB} MB.",
        )

    # ── Parse CSV ─────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(io.BytesIO(content), encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(io.BytesIO(content), encoding="latin1")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    if column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{column}' not found. Available: {list(df.columns)}",
        )

    if len(df) > MAX_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"CSV has {len(df):,} rows. Maximum: {MAX_ROWS:,}.",
        )

    reviews = df[column].dropna().astype(str).str.strip().tolist()
    reviews = [r for r in reviews if r and r.lower() != "nan"]

    if not reviews:
        raise HTTPException(status_code=400, detail="No valid reviews found in that column.")

    _file_name = file.filename or "upload.csv"
    _results   = []

    thread = threading.Thread(target=_process_batch, args=(reviews,), daemon=True)
    thread.start()

    log.info("Batch started — %d reviews from '%s'", len(reviews), _file_name)

    return {
        "message":       "Processing started",
        "file_name":     _file_name,
        "total_reviews": len(reviews),
    }


# ── Progress ──────────────────────────────────────────────────────────────────

@app.get("/progress")
def get_progress():
    """
    elapsed — real seconds since batch started (counts up)
    eta     — EMA-smoothed seconds remaining   (counts down)
    speed   — EMA-smoothed reviews/sec
    running — False when complete
    error   — non-null if batch crashed
    stale   — True if job has been running > JOB_TIMEOUT_SEC
    """
    with _lock:
        start   = _progress["start_time"]
        elapsed = round(time.time() - start, 1) if start else 0.0
        total   = _progress["total"]
        proc    = _progress["processed"]
        pct     = round(proc / total * 100, 1) if total > 0 else 0.0

        return {
            "total":     total,
            "processed": proc,
            "percent":   pct,
            "elapsed":   elapsed,
            "eta":       _progress["eta"],
            "speed":     _progress["speed"],
            "running":   _progress["running"],
            "error":     _progress.get("error"),
            "stale":     _is_stale(),
        }


# ── Results ───────────────────────────────────────────────────────────────────

@app.get("/results")
def get_results():
    """
    Returns partial results mid-batch, full results when done.
    confidence and fake_score are 0.0–1.0 — multiply ×100 in the UI.
    """
    with _lock:
        results = _results[:]
        fname   = _file_name

    return {
        "file_name": fname,
        "total":     len(results),
        "results":   results,
    }


# ── Insights ──────────────────────────────────────────────────────────────────

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

    total          = len(results)
    sent_counts    = Counter(r.get("sentiment", "Neutral") for r in results)
    fake_count     = sum(1 for r in results if r.get("fake_review") == "Fake")
    emotion_counts = Counter(r.get("emotion", "Unknown") for r in results)
    product_counts = Counter(r.get("product", "General") for r in results)

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

    stats_block = (
        f"Dataset summary ({total:,} reviews):\n"
        f"- Positive: {sent_counts['Positive']} ({sent_counts['Positive']/total*100:.1f}%)\n"
        f"- Neutral:  {sent_counts['Neutral']}  ({sent_counts['Neutral']/total*100:.1f}%)\n"
        f"- Negative: {sent_counts['Negative']} ({sent_counts['Negative']/total*100:.1f}%)\n"
        f"- Fake reviews flagged: {fake_count} ({fake_count/total*100:.1f}%)\n"
        f"- Average confidence: {avg_conf}%\n"
        f"- Top emotions: {', '.join(f'{e} ({c})' for e,c in top_emotions)}\n"
        f"- Top products: {', '.join(f'{p} ({c})' for p,c in top_products)}\n"
        f"- Top negative keywords: {', '.join(neg_keywords) or 'none'}\n"
        f"- Top positive keywords: {', '.join(pos_keywords) or 'none'}\n"
        f"- Top fake-review keywords: {', '.join(fake_keywords) or 'none'}"
    )

    prompt = (
        "You are a concise business analyst. Given the following review dataset "
        "statistics, write a sharp executive summary in plain English.\n\n"
        f"{stats_block}\n\n"
        "Rules:\n"
        "- Write 4–6 punchy sentences (no bullet points, no headers).\n"
        "- Lead with the most important insight.\n"
        "- Mention specific percentages or numbers.\n"
        "- Call out any red flags (high fake rate, high negativity).\n"
        "- Reference top keywords naturally if they reveal a theme.\n"
        "- End with one actionable recommendation.\n"
        "- Keep it under 120 words."
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json"},
                json={
                    "model":      "claude-sonnet-4-20250514",
                    "max_tokens": 300,
                    "messages":   [{"role": "user", "content": prompt}],
                },
            )
        resp.raise_for_status()
        summary = resp.json()["content"][0]["text"].strip()

    except Exception as exc:
        log.warning("Anthropic API failed: %s — using template fallback.", exc)
        pos_pct  = round(sent_counts["Positive"] / total * 100, 1)
        neg_pct  = round(sent_counts["Negative"] / total * 100, 1)
        fake_pct = round(fake_count / total * 100, 1)
        neg_kw   = ", ".join(neg_keywords[:3]) if neg_keywords else "general quality"
        summary = (
            f"Analysis of {total:,} reviews shows {pos_pct}% positive and "
            f"{neg_pct}% negative sentiment, with average confidence of {avg_conf}%. "
            f"{fake_pct}% of reviews were flagged as potentially fake"
            + (" — above the 10% threshold, warranting attention." if fake_pct >= 10 else ".") +
            f" Negative reviews frequently mention: {neg_kw}. "
            "Consider addressing these pain points to improve overall satisfaction."
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


# ── Duplicates ────────────────────────────────────────────────────────────────

@app.get("/duplicates")
def get_duplicates(
    near_threshold: float = 0.80,
    sem_threshold:  float = 0.85,
):
    """
    Run 3-stage duplicate detection:
      Stage 1 — Exact match      (MD5 hash, O(n))
      Stage 2 — Near-duplicate   (Jaccard on char 3-shingles)
      Stage 3 — Semantic clone   (TF-IDF cosine similarity)
    """
    with _lock:
        results = _results[:]

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No results available yet. Run a batch analysis first.",
        )

    reviews = [r.get("review", "") for r in results]

    try:
        future = _model_executor.submit(
            detect_duplicates, reviews, near_threshold, sem_threshold
        )
        report = future.result(timeout=DEDUP_TIMEOUT)
    except concurrent.futures.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=(
                f"Duplicate detection timed out after {DEDUP_TIMEOUT}s. "
                "Try a smaller dataset or raise the similarity thresholds."
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Duplicate detection failed: {exc}")

    return report


# ── Debug ─────────────────────────────────────────────────────────────────────

@app.get("/debug")
def debug():
    """Quick smoke-test of all model pipelines."""
    test    = "This product is amazing and works perfectly!"
    results = {}

    for name, fn in [
        ("sentiment",     lambda: str(predict_sentiment(test))),
        ("product",       lambda: str(detect_product(test))),
        ("fake",          lambda: str(detect_fake(test))),
        ("product_batch", lambda: str(_detect_product_batch([test]))),
        ("fake_batch",    lambda: str(_detect_fake_batch([test]))),
    ]:
        try:
            results[name] = fn()
        except Exception as exc:
            results[name] = f"FAILED: {exc}"

    results["emotion"] = "Neutral (disabled)"
    return results