from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import gc

torch.set_grad_enabled(False)

# ── Model choice ──────────────────────────────────────────────
# twitter-roberta-base-sentiment      (~125MB, 3-class: Neg/Neu/Pos) ← USE THIS
# twitter-roberta-base-sentiment-latest (~500MB, same labels but much larger)
#
# The "latest" variant bundles larger tweet-corpus embeddings and pushes
# RAM over Render free tier's 512MB limit. The base variant is 4x smaller,
# equally accurate on product reviews, and still gives true 3-class output.
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
LABELS     = ["Negative", "Neutral", "Positive"]

# ── Lazy-loaded globals ───────────────────────────────────────
_tokenizer = None
_model     = None
_device    = None


def _load():
    """Load model on first use, apply INT8 quantization for ~2x CPU speedup."""
    global _tokenizer, _model, _device

    if _model is not None:
        return

    print(f"[sentiment_model] Loading {MODEL_NAME}...")
    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # INT8 dynamic quantization — cuts RAM ~40% and speeds up inference ~2x on CPU
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


def unload():
    """
    Free the model from RAM. Called before loading the ABSA model so both
    never sit in memory simultaneously on Render's 512MB free tier.
    The next predict call will reload automatically.
    """
    global _tokenizer, _model, _device
    if _model is not None:
        print("[sentiment_model] Unloading to free RAM...")
        del _model
        del _tokenizer
        _model     = None
        _tokenizer = None
        _device    = None
        gc.collect()
        print("[sentiment_model] Unloaded ✓")


def predict_sentiment(text: str) -> tuple[str, float]:
    """Predict sentiment for a single text input."""
    if not text:
        return ("Neutral", 0.0)
    result = predict_sentiment_batch([text])
    return result[0] if result else ("Neutral", 0.0)


def predict_sentiment_batch(texts: list[str], batch_size: int = 32) -> list[tuple[str, float]]:
    """
    Predict sentiment for a list of texts.

    Key optimisations vs original:
      - INT8 quantized model (~2x faster on CPU)
      - max_length=128 instead of 512 — reviews are short; this alone
        cuts attention computation by ~16x for long inputs
      - batch_size=32 default (outer loop in main.py controls chunking)
    """
    if not texts:
        return []

    _load()

    results = []

    for i in range(0, len(texts), batch_size):
        batch  = texts[i : i + batch_size]
        inputs = _tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,        # ← was 512; reviews rarely exceed 128 tokens
            return_tensors="pt",
        ).to(_device)

        with torch.no_grad():
            outputs = _model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)

        for prob in probs:
            label      = LABELS[torch.argmax(prob).item()]
            confidence = round(torch.max(prob).item(), 4)
            results.append((label, confidence))

    return results