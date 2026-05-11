"""
Sentiment Analysis Platform — Streamlit Frontend (v5.0 Fixed)

Fixes vs v4:
  ─ HTML tags (</div> etc.) stripped at DISPLAY TIME in review cards
  ─ Avg Conf % in product table now correctly shows e.g. 84% not 7,550%
  ─ Product category icons updated to cover all 16 categories
  ─ Confidence stored as 0–1 in results, multiplied only once for display
  ─ Review text escaped before injection into unsafe_allow_html blocks
"""

import html as _html
import re as _re

import streamlit as st
import requests
import pandas as pd
import time

st.set_page_config(
    page_title="Sentiment AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://127.0.0.1:8000"

# ── session state defaults ─────────────────────────────────────────────────────
for _k, _v in [
    ("page", "Home"),
    ("insight_text", ""),
    ("insight_pos", []),
    ("insight_neg", []),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Syne:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --ink:#0c0c0e; --bg:#f4f2ec; --surface:#fafaf8;
    --teal:#0f7c6e; --teal2:#0a5c52; --teal-pale:#d6f0ec;
    --amber:#c47b0a; --amber-pale:#fdf2db;
    --rose:#b83348;  --rose-pale:#fce8ec;
    --violet:#5040c8; --violet-pale:#ede9ff;
    --muted:#6b6860; --border:rgba(12,12,14,.09);
}
html,body,[data-testid="stAppViewContainer"],[data-testid="stAppViewContainer"]>div>div {
    background:var(--bg) !important; font-family:'Syne',sans-serif !important; color:var(--ink) !important;
}
[data-testid="stHeader"],[data-testid="stToolbar"] { display:none !important; }
#MainMenu,footer { visibility:hidden !important; }
[data-testid="stSidebar"] { background:#0c0c0e !important; border-right:none !important; }
[data-testid="stSidebar"] * { color:rgba(255,255,255,.72) !important; }
[data-testid="stSidebarContent"] { padding:24px 18px !important; }
.block-container { padding:2rem 2.5rem 4rem !important; max-width:1280px !important; }

h1 { font-family:'DM Serif Display',serif !important; font-size:50px !important; letter-spacing:-.02em !important; line-height:1.06 !important; margin:0 0 16px !important; }
h2 { font-family:'DM Serif Display',serif !important; font-size:32px !important; letter-spacing:-.02em !important; margin:0 0 6px !important; }

.stRadio > label { display:none !important; }
.stRadio > div { gap:3px !important; }
.stRadio > div > label {
    background:rgba(255,255,255,.05) !important; border-radius:8px !important;
    padding:9px 14px !important; color:rgba(255,255,255,.6) !important;
    font-family:'Syne',sans-serif !important; font-size:13px !important;
    font-weight:500 !important; cursor:pointer !important; transition:background .15s !important;
    display:flex !important; align-items:center !important;
}
.stRadio > div > label:has(input:checked) { background:rgba(255,255,255,.13) !important; color:#fff !important; }
.stRadio > div > label > div:first-child { display:none !important; }

.stButton > button {
    background:var(--ink) !important; color:var(--surface) !important;
    border:none !important; border-radius:10px !important;
    font-family:'Syne',sans-serif !important; font-weight:600 !important;
    font-size:13px !important; padding:11px 22px !important;
    letter-spacing:.02em !important; transition:opacity .18s !important;
}
.stButton > button:hover:not(:disabled) { opacity:.82 !important; }
.stButton > button:disabled { opacity:.45 !important; }
.stDownloadButton > button {
    background:transparent !important; color:var(--teal) !important;
    border:1px solid rgba(15,124,110,.35) !important;
    font-family:'Syne',sans-serif !important; font-weight:600 !important;
    font-size:12px !important; padding:8px 18px !important; width:auto !important;
    border-radius:8px !important;
}
.stDownloadButton > button:hover { background:var(--teal) !important; color:#fff !important; }

.stTextArea textarea {
    border:1.5px solid var(--border) !important; border-radius:12px !important;
    font-family:'Syne',sans-serif !important; font-size:14px !important;
    background:var(--surface) !important; color:var(--ink) !important;
    padding:12px 14px !important;
}
.stTextArea textarea:focus { border-color:var(--teal) !important; box-shadow:0 0 0 3px rgba(15,124,110,.1) !important; }
.stSelectbox > div > div { border:1.5px solid var(--border) !important; border-radius:10px !important; background:var(--surface) !important; }
[data-testid="stFileUploaderDropzone"] { background:var(--surface) !important; border:2px dashed rgba(12,12,14,.15) !important; border-radius:14px !important; }

.card { background:var(--surface); border:1px solid var(--border); border-radius:20px; padding:26px 30px; box-shadow:0 2px 14px rgba(12,12,14,.05); position:relative; overflow:hidden; }
.card-teal::before   { content:''; position:absolute; top:0; left:0; right:0; height:3px; background:linear-gradient(90deg,var(--teal),#1cbfac); }
.card-amber::before  { content:''; position:absolute; top:0; left:0; right:0; height:3px; background:linear-gradient(90deg,var(--amber),#f0b033); }
.card-violet::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; background:linear-gradient(90deg,var(--violet),#7c6ee0); }

.tag { display:inline-flex; align-items:center; gap:6px; font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.14em; padding:5px 13px; border-radius:100px; margin-bottom:14px; }
.tag-teal   { color:var(--teal2); background:var(--teal-pale); border:1px solid rgba(15,124,110,.2); }
.tag-amber  { color:var(--amber); background:var(--amber-pale); border:1px solid rgba(196,123,10,.2); }
.tag-violet { color:var(--violet); background:var(--violet-pale); border:1px solid rgba(80,64,200,.2); }

.sec { font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.12em; color:var(--muted); margin:28px 0 12px; display:flex; align-items:center; gap:8px; }
.sec::before { content:''; width:5px; height:5px; background:var(--teal); border-radius:50%; flex-shrink:0; }

.kpi { background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:20px 18px; position:relative; overflow:hidden; }
.kpi::after { content:''; position:absolute; bottom:0; left:0; right:0; height:3px; background:var(--kc,var(--teal)); border-radius:0 0 16px 16px; }
.kpi-em  { font-size:18px; margin-bottom:8px; }
.kpi-lbl { font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); margin-bottom:3px; }
.kpi-val { font-family:'DM Serif Display',serif; font-size:34px; line-height:1; color:var(--kc,var(--teal)); }
.kpi-sub { font-size:11px; color:var(--muted); margin-top:4px; }

.mbox { background:var(--bg); border:1px solid var(--border); border-radius:14px; padding:16px 18px; }
.mlbl { font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.12em; color:var(--muted); margin-bottom:5px; }
.mval { font-family:'DM Serif Display',serif; font-size:26px; line-height:1; }
.mbar { height:5px; background:rgba(12,12,14,.08); border-radius:10px; margin-top:10px; overflow:hidden; }
.mfill { height:100%; background:linear-gradient(90deg,var(--teal),#1cbfac); border-radius:10px; }

.bdg { display:inline-flex; align-items:center; padding:3px 10px; border-radius:100px; font-size:11px; font-weight:700; letter-spacing:.03em; }
.pos { background:var(--teal-pale); color:var(--teal2); border:1px solid rgba(15,124,110,.2); }
.neg { background:var(--rose-pale); color:var(--rose); border:1px solid rgba(184,51,72,.2); }
.neu { background:var(--amber-pale); color:var(--amber); border:1px solid rgba(196,123,10,.2); }
.fake-b { background:#fff0e0; color:#b06000; border:1px solid rgba(176,96,0,.2); }
.real-b { background:var(--teal-pale); color:var(--teal2); border:1px solid rgba(15,124,110,.2); }

.alert { display:flex; align-items:flex-start; gap:12px; padding:13px 16px; border-radius:12px; margin-bottom:8px; border:1px solid; border-left-width:4px; font-size:13px; }
.a-ok   { background:#edfaf3; border-color:rgba(21,128,61,.15); border-left-color:#15803d; color:#166534; }
.a-crit { background:var(--rose-pale); border-color:rgba(184,51,72,.15); border-left-color:var(--rose); color:#7a1b2a; }
.at { font-weight:700; margin-bottom:2px; font-size:13px; }
.ab { font-size:12px; opacity:.9; line-height:1.5; }

.drow { display:flex; align-items:center; justify-content:space-between; padding:13px 16px; border-radius:12px; background:var(--bg); border:1px solid var(--border); margin-bottom:8px; }
.dleft { display:flex; align-items:center; gap:12px; }
.dico { width:34px; height:34px; border-radius:9px; display:flex; align-items:center; justify-content:center; font-size:14px; flex-shrink:0; }
.dlbl { font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); margin-bottom:2px; }
.dval { font-size:14px; font-weight:600; }

.rule { border-radius:10px; padding:11px 14px; margin-bottom:7px; border:1px solid; }
.r-trig { background:rgba(176,96,0,.04); border-color:rgba(176,96,0,.18); }
.r-ok   { background:rgba(15,124,110,.04); border-color:rgba(15,124,110,.14); }
.rh { display:flex; justify-content:space-between; align-items:flex-start; gap:10px; }
.rn { font-weight:700; font-size:12px; margin-bottom:2px; }
.rd { font-size:11px; color:var(--muted); line-height:1.5; }
.rm { display:inline-block; margin-top:5px; font-size:10px; font-weight:700; background:rgba(176,96,0,.08); color:#b06000; padding:2px 9px; border-radius:100px; border:1px solid rgba(176,96,0,.18); font-family:'JetBrains Mono',monospace; }
.rw  { font-size:11px; font-weight:700; color:var(--muted); white-space:nowrap; margin-top:2px; }
.rwa { color:#b06000; }

.pbar-wrap { height:10px; background:rgba(12,12,14,.08); border-radius:100px; overflow:hidden; margin:6px 0 16px; }
.pbar-fill  { height:100%; background:linear-gradient(90deg,var(--teal),#1cbfac); border-radius:100px; }

.mini2 { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.mini { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:14px 16px; text-align:center; }
.mnl { font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); margin-bottom:4px; }
.mnv { font-family:'DM Serif Display',serif; font-size:24px; }
.mns { font-size:11px; color:var(--muted); margin-top:3px; }

.insight-wrap { background:linear-gradient(135deg,#f0edff,#e5f5fc); border:1px solid rgba(80,64,200,.14); border-radius:18px; padding:24px 28px; }
.ai-tag { display:inline-flex; align-items:center; gap:6px; margin-bottom:14px; font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.14em; color:#3730a3; background:rgba(80,64,200,.1); border:1px solid rgba(80,64,200,.2); padding:4px 10px; border-radius:100px; }
.itext { font-size:15px; line-height:1.8; color:var(--ink); }
.kw-row { display:flex; flex-wrap:wrap; gap:6px; margin-top:16px; padding-top:16px; border-top:1px solid rgba(80,64,200,.12); }
.kwp { background:var(--teal-pale); color:var(--teal2); border:1px solid rgba(15,124,110,.2); padding:4px 11px; border-radius:100px; font-size:12px; font-weight:600; }
.kwn { background:var(--rose-pale); color:var(--rose); border:1px solid rgba(184,51,72,.2); padding:4px 11px; border-radius:100px; font-size:12px; font-weight:600; }

.empty { text-align:center; padding:50px 20px; display:flex; flex-direction:column; align-items:center; gap:12px; }
.eico { font-size:36px; opacity:.2; }
.etxt { font-size:13px; color:var(--muted); max-width:200px; line-height:1.65; }

.dup-row { display:flex; align-items:center; justify-content:space-between; padding:10px 14px; border-radius:10px; background:var(--bg); border:1px solid var(--border); margin-bottom:6px; font-size:12px; }
.dup-type-exact    { background:#fce8ec; color:#b83348; border:1px solid rgba(184,51,72,.2); padding:2px 9px; border-radius:100px; font-size:10px; font-weight:700; }
.dup-type-near     { background:#fff0e0; color:#b06000; border:1px solid rgba(176,96,0,.2); padding:2px 9px; border-radius:100px; font-size:10px; font-weight:700; }
.dup-type-semantic { background:var(--violet-pale); color:var(--violet); border:1px solid rgba(80,64,200,.2); padding:2px 9px; border-radius:100px; font-size:10px; font-weight:700; }

/* ── Customer View ─────────────────────────────────────────── */
.cv-card {
    background:var(--surface); border:1px solid var(--border);
    border-radius:18px; padding:20px 22px; margin-bottom:14px;
    position:relative; overflow:hidden;
    transition:box-shadow .18s;
}
.cv-card:hover { box-shadow:0 6px 24px rgba(12,12,14,.09); }
.cv-card-pos::before { content:''; position:absolute; left:0; top:0; bottom:0; width:4px; background:linear-gradient(180deg,var(--teal),#1cbfac); border-radius:4px 0 0 4px; }
.cv-card-neg::before { content:''; position:absolute; left:0; top:0; bottom:0; width:4px; background:linear-gradient(180deg,var(--rose),#e05070); border-radius:4px 0 0 4px; }
.cv-card-neu::before { content:''; position:absolute; left:0; top:0; bottom:0; width:4px; background:linear-gradient(180deg,var(--amber),#f0b033); border-radius:4px 0 0 4px; }
.cv-header { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; margin-bottom:12px; }
.cv-badges { display:flex; flex-wrap:wrap; gap:6px; align-items:center; flex-shrink:0; }
.cv-text { font-size:14px; line-height:1.7; color:var(--ink); margin-bottom:14px; padding-left:8px; word-break:break-word; }
.cv-meta { display:flex; align-items:center; gap:16px; font-size:11px; color:var(--muted); padding-left:8px; }
.cv-meta-item { display:flex; align-items:center; gap:5px; }
.cv-conf-bar { height:3px; border-radius:100px; background:rgba(12,12,14,.08); width:60px; overflow:hidden; display:inline-block; vertical-align:middle; margin-left:5px; }
.cv-conf-fill { height:100%; background:linear-gradient(90deg,var(--teal),#1cbfac); border-radius:100px; }
.cv-aspects { display:flex; flex-wrap:wrap; gap:6px; padding-left:8px; margin-top:10px; }
.cv-asp { display:inline-flex; align-items:center; gap:5px; font-size:11px; font-weight:600; padding:3px 10px; border-radius:100px; border:1px solid; }
.cv-asp-pos { color:var(--teal2); background:var(--teal-pale); border-color:rgba(15,124,110,.2); }
.cv-asp-neg { color:var(--rose); background:var(--rose-pale); border-color:rgba(184,51,72,.2); }
.cv-asp-neu { color:var(--amber); background:var(--amber-pale); border-color:rgba(196,123,10,.2); }
.cv-filter-row { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:20px; align-items:center; }
.cv-stat-strip { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:24px; }
.cv-stat { background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:14px 16px; text-align:center; }
.cv-stat-lbl { font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); margin-bottom:4px; }
.cv-stat-val { font-family:'DM Serif Display',serif; font-size:28px; line-height:1; }
.cv-stat-sub { font-size:11px; color:var(--muted); margin-top:3px; }
.cv-fake-warn { display:inline-flex; align-items:center; gap:5px; font-size:10px; font-weight:700; color:#b06000; background:#fff0e0; border:1px solid rgba(176,96,0,.2); padding:2px 9px; border-radius:100px; }
.cv-search { width:100%; }
.cv-page-btn { display:inline-flex; align-items:center; justify-content:center; gap:6px; padding:8px 18px; border-radius:9px; background:var(--ink); color:var(--surface); font-size:12px; font-weight:700; border:none; cursor:pointer; }
.cv-no-results { text-align:center; padding:40px 20px; color:var(--muted); font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
EMO = {"Joy":"😊","Anger":"😠","Sadness":"😢","Fear":"😨","Surprise":"😲",
       "Disgust":"🤢","Neutral":"😐","Love":"❤️","Unknown":"❓"}

# ── Extended product icon map covering all 16 taxonomy categories ──────────────
PROD_ICO = {
    "Electronics":      "📱",
    "Food & Drink":     "🍔",
    "Fashion":          "👗",
    "Home & Living":    "🏠",
    "Beauty":           "💄",
    "Healthcare":       "💊",
    "Automotive":       "🚗",
    "Books & Media":    "📚",
    "Movies & OTT":     "🎬",
    "Music":            "🎵",
    "Gaming":           "🎮",
    "Travel":           "✈️",
    "Education":        "🎓",
    "Software & App":   "💻",
    "Finance":          "💰",
    "Grocery & FMCG":   "🛒",
    "Sellers & Stores": "🏪",
    "General":          "📦",
}


def _sanitize_for_display(text: str) -> str:
    """
    Strip HTML tags and escape special chars so review text is ALWAYS safe
    to inject inside unsafe_allow_html blocks.
    This fixes the </div> rendering bug — run on every review before display.
    """
    # 1. Strip all HTML/XML tags (e.g. </div>, <br>, <script>)
    text = _re.sub(r"<[^>]+>", " ", text)
    # 2. Collapse extra whitespace left by stripping
    text = _re.sub(r"\s+", " ", text).strip()
    # 3. Escape remaining &, <, > so they render as text not HTML
    text = _html.escape(text)
    return text


def eico(e): return EMO.get(e, "❓")
def pct(a, b): return f"{a/b*100:.1f}%" if b else "0%"
def sent_badge(s):
    c = {"Positive": "pos", "Negative": "neg", "Neutral": "neu"}.get(s, "neu")
    return f'<span class="bdg {c}">{s}</span>'
def fake_badge(f):
    return f'<span class="bdg {"fake-b" if f=="Fake" else "real-b"}">{f}</span>'
def fmt_time(s):
    s = int(s)
    if s < 60:   return f"{s}s"
    if s < 3600: return f"{s//60}m {s%60}s"
    return f"{s//3600}h {(s%3600)//60}m"
def backend_ok():
    try: return requests.get(f"{API_BASE}/", timeout=3).status_code == 200
    except: return False

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0 0 28px">
      <div style="font-family:'DM Serif Display',serif;font-size:26px;color:#fff;line-height:1.1;margin-bottom:5px">Sentiment<br><em style="opacity:.45">AI</em></div>
      <div style="font-size:10px;color:rgba(255,255,255,.35);text-transform:uppercase;letter-spacing:.13em">Review Intelligence</div>
    </div>""", unsafe_allow_html=True)

    pages    = ["◈  Home", "◉  Text Analysis", "▦  Batch Upload", "▩  Dashboard", "◎  Product Deep Dive"]
    page_map = {"◈  Home":"Home","◉  Text Analysis":"Text Analysis","▦  Batch Upload":"Batch Upload","▩  Dashboard":"Dashboard","◎  Product Deep Dive":"Product Deep Dive"}
    idx_map  = {"Home":0,"Text Analysis":1,"Batch Upload":2,"Dashboard":3,"Product Deep Dive":4}

    sel = st.radio("nav", pages, index=idx_map.get(st.session_state.page, 0), label_visibility="collapsed")
    st.session_state.page = page_map[sel]

    st.markdown('<div style="height:1px;background:rgba(255,255,255,.08);margin:18px 0"></div>', unsafe_allow_html=True)
    ok = backend_ok()
    bc = "rgba(15,124,110,.9)" if ok else "rgba(184,51,72,.9)"
    bg = "rgba(15,124,110,.1)" if ok else "rgba(184,51,72,.1)"
    bd = "rgba(15,124,110,.2)" if ok else "rgba(184,51,72,.2)"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;font-size:11px;color:{bc};background:{bg};border:1px solid {bd};border-radius:8px;padding:8px 12px">
      <div style="width:6px;height:6px;border-radius:50%;background:{bc};flex-shrink:0"></div>
      {"Backend online" if ok else "Backend offline"}
    </div>
    <div style="margin-top:8px;font-size:10px;color:rgba(255,255,255,.28);font-family:'JetBrains Mono',monospace">{API_BASE}</div>
    """, unsafe_allow_html=True)

page = st.session_state.page

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown('<div class="tag tag-teal">◈ AI-Powered</div>', unsafe_allow_html=True)
    st.markdown("# Understand what\nreviews *really* say.")
    st.markdown('<p style="font-size:16px;color:#6b6860;line-height:1.7;max-width:540px;margin:0 0 36px">Detect sentiment, emotion, and authenticity in seconds — from a single review or a 50,000-row dataset.</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="large")
    for col, (acc, ico, title, desc) in zip([c1, c2, c3], [
        ("teal",   "◉", "Text Analysis",  "Paste any review. Get sentiment, emotion, product category, and a rule-by-rule authenticity breakdown instantly."),
        ("amber",  "▦", "Batch Upload",   "Upload a CSV with up to 50k reviews. Live ETA and speed metrics while models run in parallel."),
        ("violet", "▩", "Dashboard",      "Charts, word clouds, AI executive summary, product tables, duplicate detection, and CSV export."),
    ]):
        with col:
            st.markdown(f'<div class="card card-{acc}" style="min-height:200px"><div style="font-size:26px;margin-bottom:12px;opacity:.65">{ico}</div><div style="font-family:\'DM Serif Display\',serif;font-size:22px;margin-bottom:10px">{title}</div><div style="font-size:13px;color:#6b6860;line-height:1.65">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    caps = [
        ("RoBERTa",   "Sentiment",      "Pos / Neg / Neutral + confidence"),
        ("6 rules",   "Fake detect",    "Full explainability per rule"),
        ("12 aspects","ABSA",           "Battery · Camera · Price · more"),
        ("3-stage",   "Deduplication",  "Exact · Near · Semantic"),
        ("Claude API","AI summaries",   "Executive insights on demand"),
        ("50k rows",  "Batch cap",      "CSV pipeline with live ETA"),
        ("TF-IDF",    "Similarity",     "Cosine & Jaccard indexing"),
        ("16 cats",   "Product detect", "Electronics → Smartphone etc."),
    ]
    rows = "".join(
        f'<div style="padding:14px 16px;border-radius:12px;background:var(--bg);border:1px solid var(--border)">'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:var(--teal);margin-bottom:4px">{s}</div>'
        f'<div style="font-size:12px;font-weight:700;margin-bottom:2px">{n}</div>'
        f'<div style="font-size:11px;color:var(--muted)">{d}</div></div>'
        for s, n, d in caps
    )
    st.markdown(f'<div style="background:var(--surface);border:1px solid var(--border);border-radius:20px;padding:26px 30px"><div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin-bottom:18px">Platform capabilities</div><div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px">{rows}</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TEXT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Text Analysis":
    st.markdown('<div class="tag tag-teal">◉ Single Review</div>', unsafe_allow_html=True)
    st.markdown("## Analyse a review")
    st.markdown('<p style="font-size:13px;color:#6b6860;margin-bottom:22px">Paste any product review for instant sentiment, emotion, product detection and fake-review analysis.</p>', unsafe_allow_html=True)

    col_in, col_out = st.columns([10, 12], gap="large")

    with col_in:
        examples = [
            "— try an example —",
            "Best product ever!! Buy now!!",
            "Battery died after 2 days. Very disappointed.",
            "good good value for money, highly recommended!!!",
            "The camera is decent but the charging cable broke in a week.",
            "AMAZING AMAZING PRODUCT BUY NOW!!!",
        ]
        sel  = st.selectbox("Examples", examples, label_visibility="collapsed")
        seed = "" if sel.startswith("—") else sel
        review_text = st.text_area("Review", value=seed, height=190, max_chars=1000,
                                   placeholder="Paste your review here…", label_visibility="collapsed")
        char_col = "#b83348" if len(review_text) > 900 else "#c47b0a" if len(review_text) > 700 else "#aaa"
        st.markdown(f'<div style="font-size:11px;color:{char_col};text-align:right;font-family:JetBrains Mono,monospace;margin-top:-8px;margin-bottom:12px">{len(review_text)}/1000</div>', unsafe_allow_html=True)
        do_analyze = st.button("Analyse →", use_container_width=True)

    with col_out:
        if do_analyze and review_text.strip():
            if not backend_ok():
                st.error(f"Backend not reachable at `{API_BASE}`")
            else:
                with st.spinner("Running models…"):
                    try:
                        resp = requests.post(f"{API_BASE}/analyze_text", json={"text": review_text}, timeout=30)
                        resp.raise_for_status()
                        d = resp.json()

                        sent       = d.get("sentiment", "Neutral")
                        conf       = round(d.get("confidence", 0) * 100)   # ← multiply once here
                        emotion    = d.get("emotion", "Neutral")
                        product    = d.get("product", "General")
                        fake       = d.get("fake_review", "Real")
                        fake_score = round(d.get("fake_score", 0) * 100)
                        reasons    = d.get("fake_reasons", [])

                        sc = {"Positive": "var(--teal)", "Negative": "var(--rose)", "Neutral": "var(--amber)"}.get(sent, "var(--muted)")
                        fc = "#b06000" if fake == "Fake" else "var(--teal)"

                        m1, m2, m3 = st.columns(3)
                        m1.markdown(f'<div class="mbox"><div class="mlbl">Sentiment</div><div class="mval" style="color:{sc}">{sent}</div><div style="margin-top:8px">{sent_badge(sent)}</div></div>', unsafe_allow_html=True)
                        m2.markdown(f'<div class="mbox"><div class="mlbl">Confidence</div><div class="mval">{conf}%</div><div class="mbar"><div class="mfill" style="width:{conf}%"></div></div></div>', unsafe_allow_html=True)
                        m3.markdown(f'<div class="mbox"><div class="mlbl">Authenticity</div><div class="mval" style="color:{fc}">{fake}</div><div style="margin-top:8px">{fake_badge(fake)}</div></div>', unsafe_allow_html=True)

                        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

                        prod_ico_single = PROD_ICO.get(product, "📦")
                        st.markdown(f"""
                        <div class="drow">
                          <div class="dleft">
                            <div class="dico" style="background:var(--violet-pale)">{prod_ico_single}</div>
                            <div><div class="dlbl">Product Category</div><div class="dval">{product}</div></div>
                          </div>
                        </div>
                        <div class="drow">
                          
                          <div style="text-align:right">
                            <div style="font-size:10px;color:var(--muted);font-family:'JetBrains Mono',monospace">fake score</div>
                            <div style="font-family:'DM Serif Display',serif;font-size:20px;color:{fc}">{fake_score}%</div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if reasons:
                            trig_n = sum(1 for r in reasons if r["triggered"])
                            with st.expander(f"🕵️ {'Flagged fake' if fake=='Fake' else 'Looks real'} — {trig_n} rule{'s' if trig_n != 1 else ''} triggered", expanded=True):
                                bar_c = "#b06000" if fake == "Fake" else "var(--teal)"
                                st.markdown(f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:14px"><div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);white-space:nowrap">Fake score</div><div style="flex:1;height:8px;background:rgba(12,12,14,.08);border-radius:100px;overflow:hidden"><div style="height:100%;width:{fake_score}%;background:{bar_c};border-radius:100px"></div></div><div style="font-size:12px;font-weight:700;color:{bar_c};font-family:\'JetBrains Mono\',monospace">{fake_score}%</div></div>', unsafe_allow_html=True)
                                for r in reasons:
                                    trig = r["triggered"]
                                    ico  = "⚠" if trig else "✓"
                                    wt   = int(r.get("weight", 0) * 100)
                                    mc   = f'<div class="rm">↳ {_html.escape(str(r["matched"]))}</div>' if trig and r.get("matched") else ""
                                    st.markdown(f'<div class="rule {"r-trig" if trig else "r-ok"}"><div class="rh"><div><div class="rn">{ico}  {r["rule"]}</div><div class="rd">{r["description"]}</div>{mc}</div><div class="{"rwa" if trig else "rw"}">+{wt}%</div></div></div>', unsafe_allow_html=True)

                    except requests.exceptions.ConnectionError:
                        st.error(f"Backend not reachable at `{API_BASE}`")
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

        elif do_analyze:
            st.warning("Enter a review first.")
        else:
            st.markdown('<div class="empty"><div class="eico">◉</div><div class="etxt">Enter a review on the left and click Analyse to see results.</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# BATCH UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Batch Upload":
    st.markdown('<div class="tag tag-amber">▦ Batch Processing</div>', unsafe_allow_html=True)
    st.markdown("## Upload a dataset")
    st.markdown('<p style="font-size:13px;color:#6b6860;margin-bottom:22px">Upload a CSV and pick the review column. All models run in parallel with live progress tracking.</p>', unsafe_allow_html=True)

    col_l, col_r = st.columns([9, 11], gap="large")

    with col_l:
        uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
        col_sel  = None
        if uploaded:
            try:
                pdf = pd.read_csv(uploaded, nrows=1)
                uploaded.seek(0)
                col_sel = st.selectbox("Review column", list(pdf.columns))
            except Exception as e:
                st.error(f"Parse error: {e}")
        run_batch = st.button("Analyse Dataset →", use_container_width=True,
                              disabled=(uploaded is None or col_sel is None))

    if not run_batch:
        with col_r:
            st.markdown('<div class="empty" style="padding:40px 0"><div class="eico">▦</div><div class="etxt">Upload a CSV and start the analysis to see live progress here.</div></div>', unsafe_allow_html=True)

    if run_batch and uploaded and col_sel:
        if not backend_ok():
            st.error(f"Backend not reachable at `{API_BASE}`")
        else:
            uploaded.seek(0)
            try:
                resp = requests.post(
                    f"{API_BASE}/upload_csv",
                    files={"file": (uploaded.name, uploaded, "text/csv")},
                    data={"column": col_sel},
                    timeout=15,
                )
                resp.raise_for_status()
                total = resp.json().get("total_reviews", 0)
            except Exception as e:
                st.error(f"Upload failed: {e}")
                st.stop()

            start = time.time()
            last_speed, last_eta = 0.0, "—"

            with col_r:
                s_el   = st.empty()
                pct_el = st.empty()
                bar_el = st.empty()
                met_el = st.empty()

            while True:
                try:
                    pr = requests.get(f"{API_BASE}/progress", timeout=5).json()
                except Exception:
                    time.sleep(0.5)
                    continue

                processed = pr.get("processed", 0)
                percent   = pr.get("percent", 0.0)
                speed     = pr.get("speed", 0)
                eta       = pr.get("eta", 0)
                running   = pr.get("running", True)
                elapsed   = time.time() - start

                if speed > 0: last_speed = speed
                if eta   > 0: last_eta   = fmt_time(eta)

                done = not running and processed >= total
                sc_  = "var(--teal)" if done else "var(--amber)"

                s_el.markdown(f'<div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:{sc_};margin-bottom:8px">{"Complete ✓" if done else "Processing…"}</div>', unsafe_allow_html=True)
                pct_el.markdown(f'<div style="display:flex;justify-content:space-between;font-size:11px;font-family:\'JetBrains Mono\',monospace;color:var(--muted);margin-bottom:4px"><span>{processed:,} / {total:,}</span><span style="color:var(--teal);font-weight:700">{percent:.1f}%</span></div>', unsafe_allow_html=True)
                bar_el.markdown(f'<div class="pbar-wrap"><div class="pbar-fill" style="width:{percent}%"></div></div>', unsafe_allow_html=True)
                met_el.markdown(f"""
                <div class="mini2">
                  <div class="mini"><div class="mnl">Speed</div><div class="mnv">{last_speed:.1f}</div><div class="mns">reviews/sec</div></div>
                  <div class="mini"><div class="mnl">Elapsed</div><div class="mnv">{fmt_time(elapsed)}</div><div class="mns">real time</div></div>
                  <div class="mini"><div class="mnl">Remaining</div><div class="mnv">{max(0,total-processed):,}</div><div class="mns">reviews</div></div>
                  <div class="mini"><div class="mnl">ETA</div><div class="mnv">{"Done!" if done else last_eta}</div><div class="mns">estimated</div></div>
                </div>""", unsafe_allow_html=True)

                if done:
                    time.sleep(1.2)
                    st.session_state.page = "Dashboard"
                    st.session_state.insight_text = ""
                    st.rerun()

                time.sleep(0.5)

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dashboard":
    import plotly.graph_objects as go

    st.markdown('<div class="tag tag-violet">▩ Analytics</div>', unsafe_allow_html=True)
    st.markdown("## Review Intelligence")

    if not backend_ok():
        st.error(f"Backend not reachable at `{API_BASE}`")
        st.stop()

    with st.spinner("Loading results…"):
        try:
            res  = requests.get(f"{API_BASE}/results", timeout=10)
            res.raise_for_status()
            data    = res.json()
            results = data.get("results", [])
        except Exception as e:
            st.error(f"Could not load: {e}")
            st.stop()

    if not results:
        st.warning("No data yet. Run a Batch Analysis first.")
        st.stop()

    df     = pd.DataFrame(results)
    total  = len(df)
    sc_    = df["sentiment"].value_counts().to_dict()
    pos_, neg_, neu_ = sc_.get("Positive", 0), sc_.get("Negative", 0), sc_.get("Neutral", 0)
    fake_n = int((df["fake_review"] == "Fake").sum())
    fname  = data.get("file_name", "")

    st.markdown(f'<p style="font-size:12px;color:var(--muted);margin-bottom:18px;font-family:\'JetBrains Mono\',monospace">{total:,} reviews{(" · " + fname) if fname else ""}</p>', unsafe_allow_html=True)

    # ── Alerts ────────────────────────────────────────────────────────────────
    ah = ""
    if fake_n / total > 0.10:
        ah += f'<div class="alert a-crit"><div style="font-weight:700;flex-shrink:0">⚠</div><div><div class="at">Elevated fake review rate</div><div class="ab">Fake reviews at {pct(fake_n,total)}, above the 10% threshold.</div></div></div>'
    if neg_ / total > 0.50:
        ah += f'<div class="alert a-crit"><div style="font-weight:700;flex-shrink:0">↓</div><div><div class="at">Negative sentiment spike</div><div class="ab">{pct(neg_,total)} of reviews are negative.</div></div></div>'
    if not ah:
        ah = '<div class="alert a-ok"><div style="font-weight:700;flex-shrink:0">✓</div><div><div class="at">No issues detected</div><div class="ab">Dataset is within normal thresholds.</div></div></div>'
    st.markdown(ah, unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    kpis = [
        ("#2563eb", "📊", "Total",    f"{total:,}",  "reviews"),
        ("var(--teal)",  "😊", "Positive", str(pos_),     pct(pos_, total)),
        ("var(--amber)", "😐", "Neutral",  str(neu_),     pct(neu_, total)),
        ("var(--rose)",  "😞", "Negative", str(neg_),     pct(neg_, total)),
        ("#b06000",      "🕵️", "Fake",     str(fake_n),   pct(fake_n, total) + " flagged"),
    ]
    for col, (color, emoji, label, val, sub) in zip(st.columns(5), kpis):
        col.markdown(f'<div class="kpi" style="--kc:{color}"><div class="kpi-em">{emoji}</div><div class="kpi-lbl">{label}</div><div class="kpi-val">{val}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

    # ── AI Insights ───────────────────────────────────────────────────────────
    st.markdown('<div class="sec">AI Insights</div>', unsafe_allow_html=True)
    regen = st.button("↻ Regenerate Summary", key="regen")

    if not st.session_state.insight_text or regen:
        with st.spinner("Generating executive summary…"):
            try:
                ir = requests.get(f"{API_BASE}/insights", timeout=30)
                ir.raise_for_status()
                idata = ir.json()
                st.session_state.insight_text = idata.get("summary", "")
                st.session_state.insight_pos  = idata.get("stats", {}).get("pos_keywords", [])
                st.session_state.insight_neg  = idata.get("stats", {}).get("neg_keywords", [])
            except Exception as e:
                st.session_state.insight_text = f"Could not generate insights: {e}"

    pp  = "".join(f'<span class="kwp">+ {k}</span>' for k in st.session_state.insight_pos[:6])
    np_ = "".join(f'<span class="kwn">− {k}</span>' for k in st.session_state.insight_neg[:6])
    st.markdown(f'<div class="insight-wrap"><div class="ai-tag"><div style="width:5px;height:5px;border-radius:50%;background:#4f46e5"></div>AI Generated</div><div class="itext">{st.session_state.insight_text}</div><div class="kw-row">{pp}{np_}</div></div>', unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Visualisations</div>', unsafe_allow_html=True)

    plo = dict(margin=dict(t=36, b=16, l=0, r=0), height=240,
               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
               font=dict(family="Syne, sans-serif", size=11, color="#6b6860"),
               showlegend=True, legend=dict(font=dict(size=10), orientation="h", y=-0.2))

    cc1, cc2 = st.columns(2)
    with cc1:
        fig = go.Figure(go.Pie(
            labels=["Positive", "Neutral", "Negative"], values=[pos_, neu_, neg_],
            hole=0.65, marker=dict(colors=["#0f7c6e","#c47b0a","#b83348"], line=dict(color="white", width=3)),
            textinfo="none", hoverinfo="label+percent+value"))
        fig.update_layout(title=dict(text="Sentiment split", font=dict(size=11), x=0.5), **plo)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with cc2:
        fig2 = go.Figure(go.Pie(
            labels=["Real", "Fake"], values=[total - fake_n, fake_n],
            hole=0.65, marker=dict(colors=["#0f7c6e","#b06000"], line=dict(color="white", width=3)),
            textinfo="none", hoverinfo="label+percent+value"))
        fig2.update_layout(title=dict(text="Authenticity", font=dict(size=11), x=0.5), **plo)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ── Word clouds ───────────────────────────────────────────────────────────
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        STOP = {"the","and","for","are","but","not","you","all","can","had","her","was","one","our",
                "out","day","get","has","him","his","how","its","may","new","now","old","see","two",
                "way","who","did","let","put","say","she","too","use","this","that","with","from",
                "have","very","well","after","like","just","been","more","when","than","then","they",
                "were","what","will","your","also","each","much","over","such","into","only","other",
                "some","these","would","could","should","really","great","good","product","item","order","bought"}

        def wc_text(filt):
            raw = " ".join(r["review"] for r in results if r.get("sentiment") == filt).lower()
            return " ".join(w for w in raw.split() if w.isalpha() and len(w) > 2 and w not in STOP)

        st.markdown('<div class="sec">Keyword clouds</div>', unsafe_allow_html=True)
        wc1, wc2 = st.columns(2)
        for col, (filt, label, cmap, lc) in [
            (wc1, ("Positive", "Positive keywords", "YlGn",  "var(--teal)")),
            (wc2, ("Negative", "Negative keywords", "OrRd",  "var(--rose)")),
        ]:
            with col:
                st.markdown(f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:{lc};margin-bottom:6px">{label}</div>', unsafe_allow_html=True)
                wct = wc_text(filt)
                if wct.strip():
                    wc = WordCloud(width=520, height=200, background_color="white",
                                   colormap=cmap, max_words=55, prefer_horizontal=0.88,
                                   collocations=False).generate(wct)
                    fig, ax = plt.subplots(figsize=(5.2, 2))
                    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
                    fig.patch.set_facecolor("white"); plt.tight_layout(pad=0)
                    st.pyplot(fig, use_container_width=True); plt.close()
                else:
                    st.markdown(f'<div style="background:var(--bg);border-radius:12px;height:140px;display:flex;align-items:center;justify-content:center;font-size:12px;color:var(--muted)">No {filt.lower()} reviews</div>', unsafe_allow_html=True)
    except ImportError:
        st.info("Install `wordcloud` and `matplotlib` for keyword clouds.")

    # ── Product table ─────────────────────────────────────────────────────────
    # FIX: Avg_Conf stored as 0.0–1.0 in results; multiply ×100 here ONCE
    # then use NumberColumn (not ProgressColumn) to avoid the ×100 double-multiply
    st.markdown('<div class="sec">By Product & Sub-Category</div>', unsafe_allow_html=True)

    # Tab view: Category summary vs detailed sub-category breakdown
    tab_cat, tab_sub = st.tabs(["📦 By Category", "🔍 By Sub-Category"])

    with tab_cat:
        prod_df = df.groupby("product").agg(
            Total     = ("review",      "count"),
            Positive  = ("sentiment",   lambda x: (x == "Positive").sum()),
            Negative  = ("sentiment",   lambda x: (x == "Negative").sum()),
            Neutral   = ("sentiment",   lambda x: (x == "Neutral").sum()),
            Fake      = ("fake_review", lambda x: (x == "Fake").sum()),
            Avg_Conf  = ("confidence",  lambda x: round(x.astype(float).mean() * 100, 1)),
        ).reset_index().rename(columns={"product": "Category", "Avg_Conf": "Avg Conf %"})
        prod_df = prod_df.sort_values("Total", ascending=False)

        st.dataframe(
            prod_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Avg Conf %": st.column_config.ProgressColumn(
                    "Avg Conf %",
                    help="Average model confidence (0–100%)",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                )
            },
        )

    with tab_sub:
        if "sub_category" in df.columns:
            sub_df = df.groupby(["product", "sub_category"]).agg(
                Total    = ("review",      "count"),
                Positive = ("sentiment",   lambda x: (x == "Positive").sum()),
                Negative = ("sentiment",   lambda x: (x == "Negative").sum()),
                Fake     = ("fake_review", lambda x: (x == "Fake").sum()),
                Avg_Conf = ("confidence",  lambda x: round(x.astype(float).mean() * 100, 1)),
            ).reset_index().rename(columns={
                "product": "Category", "sub_category": "Sub-Category", "Avg_Conf": "Avg Conf %"
            })
            sub_df = sub_df.sort_values("Total", ascending=False)
            st.dataframe(
                sub_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Avg Conf %": st.column_config.ProgressColumn(
                        "Avg Conf %",
                        min_value=0, max_value=100, format="%.1f%%",
                    )
                },
            )
        else:
            st.info("Sub-category data not available. Re-run batch analysis to populate.")

    # ── Duplicate Detection ───────────────────────────────────────────────────
    st.markdown('<div class="sec">Duplicate Detection</div>', unsafe_allow_html=True)
    if st.button("🔍 Run Duplicate Detection", key="run_dup"):
        with st.spinner("Analysing duplicates across 3 stages…"):
            try:
                dr = requests.get(f"{API_BASE}/duplicates", timeout=120)
                dr.raise_for_status()
                dup = dr.json()

                d1, d2, d3, d4 = st.columns(4)
                d1.markdown(f'<div class="kpi" style="--kc:#2563eb"><div class="kpi-lbl">Total</div><div class="kpi-val">{dup["total"]}</div><div class="kpi-sub">reviews checked</div></div>', unsafe_allow_html=True)
                d2.markdown(f'<div class="kpi" style="--kc:var(--teal)"><div class="kpi-lbl">Originals</div><div class="kpi-val">{dup["originals"]}</div><div class="kpi-sub">unique reviews</div></div>', unsafe_allow_html=True)
                d3.markdown(f'<div class="kpi" style="--kc:var(--rose)"><div class="kpi-lbl">Duplicates</div><div class="kpi-val">{dup["duplicates"]}</div><div class="kpi-sub">flagged</div></div>', unsafe_allow_html=True)
                d4.markdown(f'<div class="kpi" style="--kc:#b06000"><div class="kpi-lbl">Dedup Rate</div><div class="kpi-val">{dup["dedup_rate_pct"]}%</div><div class="kpi-sub">of dataset</div></div>', unsafe_allow_html=True)

                st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

                e_n = dup.get("exact_count", 0)
                n_n = dup.get("near_count", 0)
                s_n = dup.get("semantic_count", 0)
                st.markdown(f"""
                <div style="display:flex;gap:10px;margin-bottom:16px">
                  <div style="flex:1;background:var(--rose-pale);border:1px solid rgba(184,51,72,.2);border-radius:12px;padding:12px 16px;text-align:center">
                    <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--rose);margin-bottom:4px">Exact</div>
                    <div style="font-family:'DM Serif Display',serif;font-size:28px;color:var(--rose)">{e_n}</div>
                    <div style="font-size:11px;color:var(--muted)">byte-for-byte copies</div>
                  </div>
                  <div style="flex:1;background:#fff0e0;border:1px solid rgba(176,96,0,.2);border-radius:12px;padding:12px 16px;text-align:center">
                    <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#b06000;margin-bottom:4px">Near</div>
                    <div style="font-family:'DM Serif Display',serif;font-size:28px;color:#b06000">{n_n}</div>
                    <div style="font-size:11px;color:var(--muted)">minor edits / shuffled</div>
                  </div>
                  <div style="flex:1;background:var(--violet-pale);border:1px solid rgba(80,64,200,.2);border-radius:12px;padding:12px 16px;text-align:center">
                    <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--violet);margin-bottom:4px">Semantic</div>
                    <div style="font-family:'DM Serif Display',serif;font-size:28px;color:var(--violet)">{s_n}</div>
                    <div style="font-size:11px;color:var(--muted)">same meaning, diff words</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                dup_rows = [r for r in dup["results"] if r["dup_type"] != "original"]
                if dup_rows:
                    dup_df = pd.DataFrame(dup_rows)[["index","dup_type","similarity","review"]].rename(columns={
                        "index":"#","dup_type":"Type","similarity":"Similarity","review":"Review"
                    })
                    dup_df["Similarity"] = dup_df["Similarity"].apply(lambda x: f"{x:.2f}" if x is not None else "—")
                    st.dataframe(dup_df, use_container_width=True, hide_index=True)
                else:
                    st.success("✓ No duplicates found in this dataset.")

            except requests.exceptions.Timeout:
                st.error("Duplicate detection timed out — try a smaller dataset.")
            except Exception as e:
                st.error(f"Duplicate detection failed: {e}")
    else:
        st.markdown('<div style="font-size:12px;color:var(--muted);padding:8px 0">Click the button above to run 3-stage duplicate detection (Exact · Near · Semantic) across all reviews.</div>', unsafe_allow_html=True)

    # ── Results preview + download ────────────────────────────────────────────
    st.markdown('<div class="sec">Results preview — first 50 rows</div>', unsafe_allow_html=True)
    preview = df.head(50)[["review","sentiment","product","fake_review","confidence"]].copy()
    # multiply ×100 ONCE here for display only — source data stays 0–1
    preview["confidence"] = (preview["confidence"].astype(float) * 100).round(1).astype(str) + "%"
    st.dataframe(preview, use_container_width=True, hide_index=True)

    st.download_button(
        "↓ Download full results CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="sentiment_results.csv",
        mime="text/csv",
    )

# ══════════════════════════════════════════════════════════════════════════════
# PRODUCT DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Product Deep Dive":
    st.markdown('<div class="tag tag-violet">◎ Deep Dive</div>', unsafe_allow_html=True)
    st.markdown("## Product Deep Dive")
    st.markdown('<p style="font-size:13px;color:#6b6860;margin-bottom:22px">Select a product category to drill into sentiment breakdown, aspect performance, top complaints &amp; praise, fake rate, and sub-category comparison.</p>', unsafe_allow_html=True)

    if not backend_ok():
        st.error(f"Backend not reachable at `{API_BASE}`")
        st.stop()

    with st.spinner("Loading results…"):
        try:
            res  = requests.get(f"{API_BASE}/results", timeout=10)
            res.raise_for_status()
            data    = res.json()
            results = data.get("results", [])
        except Exception as e:
            st.error(f"Could not load results: {e}")
            st.stop()

    if not results:
        st.warning("No data yet. Run a Batch Analysis first.")
        st.stop()

    import plotly.graph_objects as go
    from collections import Counter
    import re as _re2

    df_dd = pd.DataFrame(results)
    if "sub_category" not in df_dd.columns:
        df_dd["sub_category"] = "General"

    # ── Product selector ──────────────────────────────────────────────────────
    all_cats  = sorted(df_dd["product"].dropna().unique().tolist())
    if not all_cats:
        st.warning("No product categories detected in results.")
        st.stop()

    sel_col, _ = st.columns([3, 5])
    with sel_col:
        sel_cat = st.selectbox(
            "Select a product category to drill into",
            all_cats,
            label_visibility="collapsed",
        )

    df_cat = df_dd[df_dd["product"] == sel_cat]
    n_cat  = len(df_cat)

    if n_cat == 0:
        st.info(f"No reviews found for **{sel_cat}**.")
        st.stop()

    # ── KPI strip ─────────────────────────────────────────────────────────────
    sc_cat   = df_cat["sentiment"].value_counts().to_dict()
    pos_cat  = sc_cat.get("Positive", 0)
    neg_cat  = sc_cat.get("Negative", 0)
    neu_cat  = sc_cat.get("Neutral",  0)
    fake_cat = int((df_cat["fake_review"] == "Fake").sum())
    avg_conf = round(df_cat["confidence"].astype(float).mean() * 100, 1)
    n_subs   = df_cat["sub_category"].nunique()

    prod_ico_dd = PROD_ICO.get(sel_cat, "📦")
    st.markdown(f'<div style="font-family:\'DM Serif Display\',serif;font-size:26px;margin:16px 0 4px">{prod_ico_dd} {sel_cat}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:12px;color:var(--muted);margin-bottom:20px;font-family:\'JetBrains Mono\',monospace">{n_cat:,} reviews  ·  {n_subs} sub-categories</div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    for col, (color, lbl, val, sub) in zip([k1,k2,k3,k4,k5], [
        ("#2563eb",        "Total",      f"{n_cat:,}",   "reviews"),
        ("var(--teal)",    "Positive",   str(pos_cat),   pct(pos_cat, n_cat)),
        ("var(--amber)",   "Neutral",    str(neu_cat),   pct(neu_cat, n_cat)),
        ("var(--rose)",    "Negative",   str(neg_cat),   pct(neg_cat, n_cat)),
        ("#b06000",        "Fake",       str(fake_cat),  pct(fake_cat, n_cat) + " flagged"),
    ]):
        col.markdown(f'<div class="kpi" style="--kc:{color}"><div class="kpi-lbl">{lbl}</div><div class="kpi-val">{val}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    # ── Row 1: Sentiment donut + Sub-category breakdown ───────────────────────
    st.markdown('<div class="sec">Sentiment &amp; Sub-Category Breakdown</div>', unsafe_allow_html=True)
    rc1, rc2 = st.columns([1, 2], gap="large")

    plo_dd = dict(margin=dict(t=30, b=10, l=0, r=0), height=240,
                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  font=dict(family="Syne, sans-serif", size=11, color="#6b6860"),
                  showlegend=True, legend=dict(font=dict(size=10), orientation="h", y=-0.18))

    with rc1:
        fig_sent = go.Figure(go.Pie(
            labels=["Positive","Neutral","Negative"],
            values=[pos_cat, neu_cat, neg_cat],
            hole=0.62,
            marker=dict(colors=["#0f7c6e","#c47b0a","#b83348"], line=dict(color="white", width=3)),
            textinfo="none", hoverinfo="label+percent+value",
        ))
        fig_sent.update_layout(title=dict(text="Sentiment split", font=dict(size=11), x=0.5), **plo_dd)
        st.plotly_chart(fig_sent, use_container_width=True, config={"displayModeBar": False})

    with rc2:
        sub_df = df_cat.groupby("sub_category").agg(
            Total    = ("review",    "count"),
            Positive = ("sentiment", lambda x: (x=="Positive").sum()),
            Negative = ("sentiment", lambda x: (x=="Negative").sum()),
            Neutral  = ("sentiment", lambda x: (x=="Neutral").sum()),
            Fake     = ("fake_review", lambda x: (x=="Fake").sum()),
        ).reset_index().sort_values("Total", ascending=False)

        # Horizontal stacked bar per sub-category
        subs  = sub_df["sub_category"].tolist()
        fig_sub = go.Figure()
        for lbl, color in [("Positive","#0f7c6e"),("Neutral","#c47b0a"),("Negative","#b83348")]:
            fig_sub.add_trace(go.Bar(
                name=lbl, y=subs, x=sub_df[lbl].tolist(),
                orientation="h",
                marker_color=color,
                hovertemplate=f"%{{y}}<br>{lbl}: %{{x}}<extra></extra>",
            ))
        fig_sub.update_layout(
            barmode="stack",
            title=dict(text="Reviews per sub-category", font=dict(size=11), x=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(tickfont=dict(size=10)),
            **plo_dd,
        )
        st.plotly_chart(fig_sub, use_container_width=True, config={"displayModeBar": False})

    # ── Row 2: Top complaints vs top praise ───────────────────────────────────
    st.markdown('<div class="sec">Top Complaints vs Top Praise</div>', unsafe_allow_html=True)

    _STOP_DD = {
        "the","and","for","are","but","not","you","all","can","was","one","our",
        "out","get","has","how","its","new","now","see","use","this","that","with",
        "from","have","very","well","like","just","been","more","when","than","they",
        "were","what","will","your","also","much","into","only","some","would",
        "could","really","great","good","product","item","order","it","is","my",
        "me","so","an","a","i","to","in","of","on","at","be","do","by","if",
    }

    def _top_kw_dd(texts, n=8):
        ctr = Counter()
        for t in texts:
            for w in _re2.findall(r"[a-z]{3,}", t.lower()):
                if w not in _STOP_DD:
                    ctr[w] += 1
        return ctr.most_common(n)

    neg_reviews = df_cat[df_cat["sentiment"] == "Negative"]["review"].tolist()
    pos_reviews = df_cat[df_cat["sentiment"] == "Positive"]["review"].tolist()
    neg_kws = _top_kw_dd(neg_reviews)
    pos_kws = _top_kw_dd(pos_reviews)

    kw1, kw2 = st.columns(2, gap="large")

    with kw1:
        st.markdown('<div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--rose);margin-bottom:10px">↓ Top complaint keywords</div>', unsafe_allow_html=True)
        if neg_kws:
            max_c = neg_kws[0][1] if neg_kws else 1
            for word, cnt in neg_kws:
                bar_w = int(cnt / max_c * 100)
                st.markdown(f'''
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:7px">
                  <div style="width:90px;font-size:12px;font-weight:600;color:var(--ink)">{word}</div>
                  <div style="flex:1;height:7px;background:rgba(12,12,14,.07);border-radius:100px;overflow:hidden">
                    <div style="height:100%;width:{bar_w}%;background:var(--rose);border-radius:100px"></div>
                  </div>
                  <div style="width:28px;text-align:right;font-size:11px;color:var(--muted);font-family:\'JetBrains Mono\',monospace">{cnt}</div>
                </div>''', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:12px;color:var(--muted)">No negative reviews for this category.</div>', unsafe_allow_html=True)

    with kw2:
        st.markdown('<div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--teal);margin-bottom:10px">↑ Top praise keywords</div>', unsafe_allow_html=True)
        if pos_kws:
            max_c = pos_kws[0][1] if pos_kws else 1
            for word, cnt in pos_kws:
                bar_w = int(cnt / max_c * 100)
                st.markdown(f'''
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:7px">
                  <div style="width:90px;font-size:12px;font-weight:600;color:var(--ink)">{word}</div>
                  <div style="flex:1;height:7px;background:rgba(12,12,14,.07);border-radius:100px;overflow:hidden">
                    <div style="height:100%;width:{bar_w}%;background:var(--teal);border-radius:100px"></div>
                  </div>
                  <div style="width:28px;text-align:right;font-size:11px;color:var(--muted);font-family:\'JetBrains Mono\',monospace">{cnt}</div>
                </div>''', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:12px;color:var(--muted)">No positive reviews for this category.</div>', unsafe_allow_html=True)

    # ── Row 3: Fake rate + confidence vs dataset average ──────────────────────
    st.markdown('<div class="sec">Fake Rate &amp; Confidence vs Dataset Average</div>', unsafe_allow_html=True)

    total_all    = len(df_dd)
    fake_all_pct = round((df_dd["fake_review"] == "Fake").sum() / total_all * 100, 1) if total_all else 0
    fake_cat_pct = round(fake_cat / n_cat * 100, 1) if n_cat else 0
    conf_all     = round(df_dd["confidence"].astype(float).mean() * 100, 1)

    cm1, cm2, cm3, cm4 = st.columns(4)
    for col, (color, lbl, v_cat, v_all, suffix) in zip([cm1, cm2, cm3, cm4], [
        ("var(--rose)",   "Fake rate — this category",  f"{fake_cat_pct}%",  f"Dataset avg: {fake_all_pct}%",   ""),
        ("#b06000",       "Fake count — this category", str(fake_cat),       f"of {n_cat} reviews",             ""),
        ("var(--teal)",   "Avg confidence — this cat",  f"{avg_conf}%",      f"Dataset avg: {conf_all}%",       ""),
        ("var(--violet)", "Sub-categories",             str(n_subs),         f"in {sel_cat}",                   ""),
    ]):
        col.markdown(f'<div class="kpi" style="--kc:{color}"><div class="kpi-lbl">{lbl}</div><div class="kpi-val">{v_cat}</div><div class="kpi-sub">{v_all}</div></div>', unsafe_allow_html=True)

    # ── Row 4: Sub-category detail table ──────────────────────────────────────
    st.markdown('<div class="sec">Sub-Category Detail</div>', unsafe_allow_html=True)

    sub_tbl = sub_df.copy()
    sub_tbl["Fake %"]   = (sub_tbl["Fake"] / sub_tbl["Total"] * 100).round(1)
    sub_tbl["Pos %"]    = (sub_tbl["Positive"] / sub_tbl["Total"] * 100).round(1)
    sub_tbl["Neg %"]    = (sub_tbl["Negative"] / sub_tbl["Total"] * 100).round(1)
    sub_tbl = sub_tbl.rename(columns={"sub_category": "Sub-Category"})
    sub_tbl = sub_tbl[["Sub-Category","Total","Positive","Neutral","Negative","Pos %","Neg %","Fake","Fake %"]]

    st.dataframe(
        sub_tbl,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Pos %":  st.column_config.ProgressColumn("Pos %",  min_value=0, max_value=100, format="%.1f%%"),
            "Neg %":  st.column_config.ProgressColumn("Neg %",  min_value=0, max_value=100, format="%.1f%%"),
            "Fake %": st.column_config.ProgressColumn("Fake %", min_value=0, max_value=100, format="%.1f%%"),
        },
    )

    # ── Row 5: Representative review snippets ─────────────────────────────────
    st.markdown('<div class="sec">Representative Reviews</div>', unsafe_allow_html=True)
    snip1, snip2 = st.columns(2, gap="large")

    def _render_snippets(col, reviews_subset, sentiment, color, icon):
        with col:
            st.markdown(f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:{color};margin-bottom:10px">{icon} {sentiment} samples</div>', unsafe_allow_html=True)
            sample = reviews_subset[:4]
            if not sample:
                st.markdown(f'<div style="font-size:12px;color:var(--muted)">No {sentiment.lower()} reviews.</div>', unsafe_allow_html=True)
                return
            for rv in sample:
                txt = rv.get("review","")
                txt = txt[:160] + "…" if len(txt) > 160 else txt
                txt = _sanitize_for_display(txt)
                conf_pct_s = round(float(rv.get("confidence",0))*100)
                fake_s     = rv.get("fake_review","Real")
                sub_s      = rv.get("sub_category","")
                fake_chip_s = f'<span style="font-size:9px;background:#fff0e0;color:#b06000;border:1px solid rgba(176,96,0,.2);padding:1px 7px;border-radius:100px;font-weight:700">⚠ Fake</span>' if fake_s == "Fake" else ""
                st.markdown(f'''
                <div style="background:var(--bg);border:1px solid var(--border);border-left:3px solid {color};border-radius:10px;padding:12px 14px;margin-bottom:8px">
                  <div style="font-size:12px;line-height:1.65;color:var(--ink);margin-bottom:8px">{txt}</div>
                  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
                    <span style="font-size:10px;color:var(--muted)">{sub_s}</span>
                    <span style="font-size:10px;color:var(--muted)">·</span>
                    <span style="font-size:10px;font-weight:700;color:{color}">{conf_pct_s}% conf</span>
                    {fake_chip_s}
                  </div>
                </div>''', unsafe_allow_html=True)

    neg_sample = df_cat[df_cat["sentiment"]=="Negative"].head(4).to_dict("records")
    pos_sample = df_cat[df_cat["sentiment"]=="Positive"].head(4).to_dict("records")
    _render_snippets(snip1, neg_sample, "Negative", "var(--rose)", "↓")
    _render_snippets(snip2, pos_sample, "Positive", "var(--teal)", "↑")