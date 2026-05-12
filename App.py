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
    idx_map  = {"Home":0,"Text Analysis":1,"Batch Upload":2,"Dashboard":3,"Product Deep Dive":4,"Customer View":5}

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
                          <div class="dleft">
                            <div class="dico" style="background:var(--amber-pale)">{eico(emotion)}</div>
                            <div><div class="dlbl">Dominant Emotion</div><div class="dval">{emotion}</div></div>
                          </div>
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
    st.markdown('<div class="sec">By Product</div>', unsafe_allow_html=True)
    prod_df = df.groupby("product").agg(
        Total     = ("review",      "count"),
        Positive  = ("sentiment",   lambda x: (x == "Positive").sum()),
        Negative  = ("sentiment",   lambda x: (x == "Negative").sum()),
        Neutral   = ("sentiment",   lambda x: (x == "Neutral").sum()),
        Fake      = ("fake_review", lambda x: (x == "Fake").sum()),
        Avg_Conf  = ("confidence",  lambda x: round(x.astype(float).mean() * 100, 1)),
    ).reset_index().rename(columns={"product": "Product", "Avg_Conf": "Avg Conf %"})
    prod_df = prod_df.sort_values("Total", ascending=False)

    st.dataframe(
        prod_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            # ProgressColumn expects values 0–100 — Avg_Conf is already ×100 above
            "Avg Conf %": st.column_config.ProgressColumn(
                "Avg Conf %",
                help="Average model confidence (0–100%)",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            )
        },
    )

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
    import plotly.graph_objects as go
    from product_detection import detect_product_full, TAXONOMY

    st.markdown('<div class="tag tag-violet">◎ Product Intelligence</div>', unsafe_allow_html=True)
    st.markdown("## Product Deep Dive")
    st.markdown('<p style="font-size:13px;color:#6b6860;margin-bottom:22px">Drill into any product category — sentiment breakdown, sub-category performance, and representative reviews.</p>', unsafe_allow_html=True)

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

    # ── Re-classify with sub-category (results only have top-level category) ──
    # We run detect_product_full on each review text so we get sub_category too.
    # Cache in session_state to avoid re-running on every interaction.
    cache_key = f"pdd_cache_{len(results)}"
    if cache_key not in st.session_state:
        enriched = []
        for r in results:
            pr = detect_product_full(r.get("review", ""))
            enriched.append({**r, "sub_category": pr.sub_category})
        st.session_state[cache_key] = enriched
    enriched = st.session_state[cache_key]

    df_pdd  = pd.DataFrame(enriched)
    total   = len(df_pdd)

    # ── Category selector ─────────────────────────────────────────────────────
    all_cats = sorted(df_pdd["product"].dropna().unique().tolist())
    if not all_cats:
        st.warning("No product categories found in results.")
        st.stop()

    plo_base = dict(
        margin=dict(t=36, b=16, l=0, r=0), height=220,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Syne, sans-serif", size=11, color="#6b6860"),
        showlegend=True,
        legend=dict(font=dict(size=10), orientation="h", y=-0.25),
    )

    # ── Overview strip: one KPI per category ─────────────────────────────────
    st.markdown('<div class="sec">Category Overview</div>', unsafe_allow_html=True)

    cat_summary = []
    for cat in all_cats:
        sub = df_pdd[df_pdd["product"] == cat]
        n   = len(sub)
        pos = int((sub["sentiment"] == "Positive").sum())
        neg = int((sub["sentiment"] == "Negative").sum())
        fk  = int((sub["fake_review"] == "Fake").sum())
        avg_conf = round(sub["confidence"].astype(float).mean() * 100, 1)
        cat_summary.append({"cat": cat, "n": n, "pos": pos, "neg": neg, "fake": fk, "conf": avg_conf})

    cat_summary.sort(key=lambda x: -x["n"])

    # Render as a scrollable overview grid
    grid_html = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin-bottom:24px">'
    for cs in cat_summary:
        ico = PROD_ICO.get(cs["cat"], "📦")
        pos_pct = round(cs["pos"] / cs["n"] * 100) if cs["n"] else 0
        neg_pct = round(cs["neg"] / cs["n"] * 100) if cs["n"] else 0
        bar_pos = f'<div style="height:4px;background:rgba(12,12,14,.08);border-radius:10px;margin-top:8px;overflow:hidden"><div style="width:{pos_pct}%;height:100%;background:linear-gradient(90deg,var(--teal),#1cbfac);border-radius:10px"></div></div>'
        grid_html += f'''<div class="kpi" style="--kc:var(--teal);cursor:pointer" onclick="void(0)">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div style="font-size:20px">{ico}</div>
            <div style="font-size:10px;font-weight:700;color:var(--muted);font-family:\'JetBrains Mono\',monospace">{cs["n"]} reviews</div>
          </div>
          <div style="font-size:12px;font-weight:700;margin:6px 0 2px">{cs["cat"]}</div>
          <div style="font-size:10px;color:var(--muted)">{pos_pct}% pos · {neg_pct}% neg · {cs["fake"]} fake</div>
          {bar_pos}
        </div>'''
    grid_html += '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)

    # ── Category deep-dive selector ───────────────────────────────────────────
    st.markdown('<div class="sec">Deep Dive — Select a Category</div>', unsafe_allow_html=True)
    sel_cat = st.selectbox("Product category", all_cats, label_visibility="collapsed")

    df_cat = df_pdd[df_pdd["product"] == sel_cat].copy()
    n_cat  = len(df_cat)

    if n_cat == 0:
        st.info(f"No reviews found for {sel_cat}.")
        st.stop()

    cat_pos  = int((df_cat["sentiment"] == "Positive").sum())
    cat_neg  = int((df_cat["sentiment"] == "Negative").sum())
    cat_neu  = int((df_cat["sentiment"] == "Neutral").sum())
    cat_fake = int((df_cat["fake_review"] == "Fake").sum())
    cat_conf = round(df_cat["confidence"].astype(float).mean() * 100, 1)

    # ── Category KPI row ──────────────────────────────────────────────────────
    ico_sel = PROD_ICO.get(sel_cat, "📦")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(f'<div class="kpi" style="--kc:#2563eb"><div class="kpi-em">{ico_sel}</div><div class="kpi-lbl">Reviews</div><div class="kpi-val">{n_cat:,}</div><div class="kpi-sub">in {sel_cat}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi" style="--kc:var(--teal)"><div class="kpi-em">😊</div><div class="kpi-lbl">Positive</div><div class="kpi-val">{cat_pos}</div><div class="kpi-sub">{pct(cat_pos,n_cat)}</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi" style="--kc:var(--amber)"><div class="kpi-em">😐</div><div class="kpi-lbl">Neutral</div><div class="kpi-val">{cat_neu}</div><div class="kpi-sub">{pct(cat_neu,n_cat)}</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi" style="--kc:var(--rose)"><div class="kpi-em">😞</div><div class="kpi-lbl">Negative</div><div class="kpi-val">{cat_neg}</div><div class="kpi-sub">{pct(cat_neg,n_cat)}</div></div>', unsafe_allow_html=True)
    k5.markdown(f'<div class="kpi" style="--kc:#b06000"><div class="kpi-em">🕵️</div><div class="kpi-lbl">Fake</div><div class="kpi-val">{cat_fake}</div><div class="kpi-sub">{pct(cat_fake,n_cat)} · {cat_conf}% conf</div></div>', unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────────
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)

    with ch1:
        fig = go.Figure(go.Pie(
            labels=["Positive","Neutral","Negative"],
            values=[cat_pos, cat_neu, cat_neg],
            hole=0.62,
            marker=dict(colors=["#0f7c6e","#c47b0a","#b83348"], line=dict(color="white", width=3)),
            textinfo="none", hoverinfo="label+percent+value",
        ))
        fig.update_layout(title=dict(text=f"Sentiment — {sel_cat}", font=dict(size=11), x=0.5), **plo_base)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with ch2:
        fig2 = go.Figure(go.Pie(
            labels=["Real","Fake"],
            values=[n_cat - cat_fake, cat_fake],
            hole=0.62,
            marker=dict(colors=["#0f7c6e","#b06000"], line=dict(color="white", width=3)),
            textinfo="none", hoverinfo="label+percent+value",
        ))
        fig2.update_layout(title=dict(text=f"Authenticity — {sel_cat}", font=dict(size=11), x=0.5), **plo_base)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ── Sub-category detail table ─────────────────────────────────────────────
    st.markdown('<div class="sec">Sub-Category Detail</div>', unsafe_allow_html=True)

    sub_rows = []
    for sub_name in df_cat["sub_category"].dropna().unique():
        sub_df = df_cat[df_cat["sub_category"] == sub_name]
        n_s    = len(sub_df)
        pos_s  = int((sub_df["sentiment"] == "Positive").sum())
        neu_s  = int((sub_df["sentiment"] == "Neutral").sum())
        neg_s  = int((sub_df["sentiment"] == "Negative").sum())
        fk_s   = int((sub_df["fake_review"] == "Fake").sum())
        conf_s = round(sub_df["confidence"].astype(float).mean() * 100, 1)
        sub_rows.append({
            "Sub-Category": sub_name,
            "Total":    n_s,
            "Positive": pos_s,
            "Neutral":  neu_s,
            "Negative": neg_s,
            "Pos %":    round(pos_s / n_s * 100, 1) if n_s else 0.0,
            "Neg %":    round(neg_s / n_s * 100, 1) if n_s else 0.0,
            "Fake":     fk_s,
            "Fake %":   round(fk_s / n_s * 100, 1) if n_s else 0.0,
            "Avg Conf %": conf_s,
        })

    if sub_rows:
        sub_df_display = pd.DataFrame(sub_rows).sort_values("Total", ascending=False)
        st.dataframe(
            sub_df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Pos %": st.column_config.ProgressColumn("Pos %", min_value=0, max_value=100, format="%.1f%%"),
                "Neg %": st.column_config.ProgressColumn("Neg %", min_value=0, max_value=100, format="%.1f%%"),
                "Fake %": st.column_config.ProgressColumn("Fake %", min_value=0, max_value=100, format="%.1f%%"),
                "Avg Conf %": st.column_config.ProgressColumn("Avg Conf %", min_value=0, max_value=100, format="%.1f%%"),
            },
        )
    else:
        st.info("No sub-category data available for this category.")

    # ── Sub-category selector for representative reviews ──────────────────────
    st.markdown('<div class="sec">Representative Reviews</div>', unsafe_allow_html=True)

    avail_subs = ["All Sub-Categories"] + sorted(df_cat["sub_category"].dropna().unique().tolist())
    sel_sub    = st.selectbox("Sub-category filter", avail_subs, label_visibility="collapsed")

    df_sub = df_cat if sel_sub == "All Sub-Categories" else df_cat[df_cat["sub_category"] == sel_sub]

    neg_samples  = df_sub[df_sub["sentiment"] == "Negative"]["review"].tolist()[:5]
    pos_samples  = df_sub[df_sub["sentiment"] == "Positive"]["review"].tolist()[:5]

    col_neg, col_pos = st.columns(2)

    with col_neg:
        st.markdown('<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:var(--rose);margin-bottom:10px">↓ Negative Samples</div>', unsafe_allow_html=True)
        if neg_samples:
            for rev in neg_samples:
                rev_display = _sanitize_for_display(rev[:200] + ("…" if len(rev) > 200 else ""))
                sub_tag = df_sub[df_sub["review"] == rev]["sub_category"].values
                sub_lbl = sub_tag[0] if len(sub_tag) else sel_cat
                conf_v  = df_sub[df_sub["review"] == rev]["confidence"].values
                conf_pct_v = round(float(conf_v[0]) * 100) if len(conf_v) else 0
                st.markdown(f'''<div style="background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--rose);border-radius:12px;padding:14px 16px;margin-bottom:8px">
                  <div style="font-size:13px;line-height:1.65;color:var(--ink);margin-bottom:8px">{rev_display}</div>
                  <div style="font-size:10px;color:var(--muted)">{sub_lbl} · <span style="color:var(--teal);font-weight:700">{conf_pct_v}% conf</span></div>
                </div>''', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:13px;color:var(--muted);padding:16px 0">No negative reviews.</div>', unsafe_allow_html=True)

    with col_pos:
        st.markdown('<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:var(--teal);margin-bottom:10px">↑ Positive Samples</div>', unsafe_allow_html=True)
        if pos_samples:
            for rev in pos_samples:
                rev_display = _sanitize_for_display(rev[:200] + ("…" if len(rev) > 200 else ""))
                sub_tag = df_sub[df_sub["review"] == rev]["sub_category"].values
                sub_lbl = sub_tag[0] if len(sub_tag) else sel_cat
                conf_v  = df_sub[df_sub["review"] == rev]["confidence"].values
                conf_pct_v = round(float(conf_v[0]) * 100) if len(conf_v) else 0
                st.markdown(f'''<div style="background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--teal);border-radius:12px;padding:14px 16px;margin-bottom:8px">
                  <div style="font-size:13px;line-height:1.65;color:var(--ink);margin-bottom:8px">{rev_display}</div>
                  <div style="font-size:10px;color:var(--muted)">{sub_lbl} · <span style="color:var(--teal);font-weight:700">{conf_pct_v}% conf</span></div>
                </div>''', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:13px;color:var(--muted);padding:16px 0">No positive reviews.</div>', unsafe_allow_html=True)

    # ── Download filtered ─────────────────────────────────────────────────────
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    dl_df = df_cat[["review","sentiment","sub_category","fake_review","confidence"]].copy()
    dl_df["confidence"] = (dl_df["confidence"].astype(float) * 100).round(1).astype(str) + "%"
    st.download_button(
        f"↓ Download {sel_cat} reviews CSV",
        data=dl_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{sel_cat.lower().replace(' ','_')}_reviews.csv",
        mime="text/csv",
    )

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOMER VIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Customer View":
    st.markdown('<div class="tag tag-teal">◎ Customer View</div>', unsafe_allow_html=True)
    st.markdown("## What customers are saying")
    st.markdown('<p style="font-size:13px;color:#6b6860;margin-bottom:22px">Browse reviews by product category, filter by sentiment & authenticity, and search across all fields without area restrictions.</p>', unsafe_allow_html=True)

    if not backend_ok():
        st.error(f"Backend not reachable at `{API_BASE}`")
        st.stop()

    with st.spinner("Loading reviews…"):
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

    df_cv = pd.DataFrame(results)
    total = len(df_cv)
    fname = data.get("file_name", "")

    # ── Initialize session state ──────────────────────────────────────────────────
    for k, v in [("cv_sent","All"),("cv_fake","All"),("cv_cat","All"),("cv_search",""),("cv_page",0),("cv_search_fields",["Review", "Product"])]:
        if k not in st.session_state:
            st.session_state[k] = v

    sc_   = df_cv["sentiment"].value_counts().to_dict()
    pos_c = sc_.get("Positive", 0)
    neg_c = sc_.get("Negative", 0)
    neu_c = sc_.get("Neutral",  0)
    fk_c  = int((df_cv["fake_review"] == "Fake").sum())

    st.markdown(f"""
    <div class="cv-stat-strip">
      <div class="cv-stat"><div class="cv-stat-lbl">Total Reviews</div><div class="cv-stat-val" style="color:#2563eb">{total:,}</div><div class="cv-stat-sub">{fname or "dataset"}</div></div>
      <div class="cv-stat"><div class="cv-stat-lbl">Positive</div><div class="cv-stat-val" style="color:var(--teal)">{pos_c}</div><div class="cv-stat-sub">{pct(pos_c,total)}</div></div>
      <div class="cv-stat"><div class="cv-stat-lbl">Negative</div><div class="cv-stat-val" style="color:var(--rose)">{neg_c}</div><div class="cv-stat-sub">{pct(neg_c,total)}</div></div>
      <div class="cv-stat"><div class="cv-stat-lbl">Flagged Fake</div><div class="cv-stat-val" style="color:#b06000">{fk_c}</div><div class="cv-stat-sub">{pct(fk_c,total)}</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Advanced Filters ──────────────────────────────────────────────────────────
    st.markdown('<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin:20px 0 12px">Filters & Search</div>', unsafe_allow_html=True)

    fc1, fc2, fc3, fc4 = st.columns([2.5, 1.8, 1.8, 1.8], gap="small")

    with fc1:
        search_q = st.text_input("🔍 Global search", value=st.session_state.cv_search,
                                  placeholder="Search across reviews, products, sentiment…", label_visibility="collapsed")
        if search_q != st.session_state.cv_search:
            st.session_state.cv_search = search_q
            st.session_state.cv_page   = 0

    with fc2:
        sent_opts = ["All Sentiments", "Positive", "Negative", "Neutral"]
        sent_sel  = st.selectbox("Sentiment", sent_opts,
                                  index=sent_opts.index(st.session_state.cv_sent) if st.session_state.cv_sent in sent_opts else 0,
                                  label_visibility="collapsed")
        if sent_sel != st.session_state.cv_sent:
            st.session_state.cv_sent = sent_sel
            st.session_state.cv_page = 0

    with fc3:
        fake_opts = ["All Reviews", "Real Only", "Fake Only"]
        fake_sel  = st.selectbox("Authenticity", fake_opts,
                                  index=fake_opts.index(st.session_state.cv_fake) if st.session_state.cv_fake in fake_opts else 0,
                                  label_visibility="collapsed")
        if fake_sel != st.session_state.cv_fake:
            st.session_state.cv_fake = fake_sel
            st.session_state.cv_page = 0

    with fc4:
        cat_vals = ["All Categories"] + sorted(df_cv["product"].dropna().unique().tolist())
        cat_sel  = st.selectbox("Product Category", cat_vals,
                                  index=cat_vals.index(st.session_state.cv_cat) if st.session_state.cv_cat in cat_vals else 0,
                                  label_visibility="collapsed")
        if cat_sel != st.session_state.cv_cat:
            st.session_state.cv_cat = cat_sel
            st.session_state.cv_page = 0

    # ── Apply all filters ─────────────────────────────────────────────────────────
    filtered = results[:]
    
    if sent_sel != "All Sentiments":
        filtered = [r for r in filtered if r.get("sentiment") == sent_sel]
    
    if fake_sel == "Real Only":
        filtered = [r for r in filtered if r.get("fake_review") == "Real"]
    elif fake_sel == "Fake Only":
        filtered = [r for r in filtered if r.get("fake_review") == "Fake"]
    
    if cat_sel != "All Categories":
        filtered = [r for r in filtered if r.get("product") == cat_sel]
    
    # ── Unlimited search across all fields (no area restriction) ────────────────
    if search_q.strip():
        q_lower = search_q.strip().lower()
        filtered = [
            r for r in filtered
            if (q_lower in r.get("review", "").lower() or
                q_lower in r.get("product", "").lower() or
                q_lower in r.get("sentiment", "").lower() or
                q_lower in r.get("emotion", "").lower() or
                q_lower in r.get("fake_review", "").lower())
        ]

    n_filtered = len(filtered)
    PAGE_SIZE  = 10
    n_pages    = max(1, (n_filtered + PAGE_SIZE - 1) // PAGE_SIZE)
    cur_page   = min(st.session_state.cv_page, n_pages - 1)
    page_start = cur_page * PAGE_SIZE
    page_end   = min(page_start + PAGE_SIZE, n_filtered)
    page_items = filtered[page_start:page_end]

    # ── Results summary ───────────────────────────────────────────────────────────
    summary_text = f'Showing {page_start+1}–{page_end} of {n_filtered:,} reviews'
    if n_filtered < total:
        summary_text += f'  ·  <strong>filtered</strong> from {total:,} total'
    
    st.markdown(
        f'<div style="font-size:11px;color:var(--muted);margin:16px 0 20px;font-family:\'JetBrains Mono\',monospace">{summary_text}</div>',
        unsafe_allow_html=True,
    )

    # ── Review cards with enhanced layout ──────────────────────────────────────
    if not page_items:
        st.markdown('<div class="cv-no-results">No reviews match the current filters.</div>', unsafe_allow_html=True)
    else:
        SENT_EMO = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}

        for idx, r in enumerate(page_items, start=page_start+1):
            review_txt = r.get("review", "")
            sentiment  = r.get("sentiment", "Neutral")
            confidence = float(r.get("confidence", 0))
            fake_lbl   = r.get("fake_review", "Real")
            fake_score = float(r.get("fake_score", 0))
            product    = r.get("product", "General")
            emotion    = r.get("emotion", "Neutral")
            aspects    = r.get("aspects", [])
            user_name  = r.get("user_name", "Customer")
            rating     = r.get("rating", "—")

            card_cls = {"Positive":"cv-card-pos","Negative":"cv-card-neg","Neutral":"cv-card-neu"}.get(sentiment, "cv-card-neu")
            conf_pct = round(confidence * 100)
            conf_bar = f'<div class="cv-conf-bar"><div class="cv-conf-fill" style="width:{conf_pct}%"></div></div>'
            prod_ico = PROD_ICO.get(product, "📦")
            sent_emo = SENT_EMO.get(sentiment, "😐")

            fake_chip = ""
            if fake_lbl == "Fake":
                fake_chip = f'<span class="cv-fake-warn">⚠ Possibly Fake · {round(fake_score*100)}%</span>'

            # ── Sanitize and highlight review text ────────────────────────────
            raw_text     = review_txt if len(review_txt) <= 280 else review_txt[:277] + "…"
            display_text = _sanitize_for_display(raw_text)

            if search_q.strip():
                escaped_q    = _html.escape(search_q.strip())
                display_text = _re.sub(
                    f'({_re.escape(escaped_q)})',
                    r'<mark style="background:#fef08a;border-radius:3px;padding:0 2px">\1</mark>',
                    display_text,
                    flags=_re.IGNORECASE,
                )

            # ── Render aspect chips ───────────────────────────────────────────
            asp_html = ""
            if aspects:
                chips = []
                for asp in aspects[:6]:
                    asp_cls = {
                        "Positive": "cv-asp-pos",
                        "Negative": "cv-asp-neg",
                        "Neutral":  "cv-asp-neu",
                    }.get(asp.get("sentiment", "Neutral"), "cv-asp-neu")
                    asp_ico = {"Positive": "↑", "Negative": "↓", "Neutral": "–"}.get(asp.get("sentiment"), "–")
                    chips.append(f'<span class="cv-asp {asp_cls}">{asp_ico} {asp["aspect"]}</span>')
                asp_html = f'<div class="cv-aspects">{"".join(chips)}</div>'

            st.markdown(f"""
            <div class="cv-card {card_cls}">
              <div class="cv-header">
                <div style="display:flex;align-items:center;gap:12px;flex:1">
                  <span style="font-size:24px">{sent_emo}</span>
                  <div style="flex:1">
                    <div style="font-size:12px;font-weight:700;color:var(--ink);margin-bottom:2px">{user_name}</div>
                    <div style="font-size:10px;color:var(--muted)">{prod_ico} {product}  ·  {emotion}</div>
                  </div>
                </div>
                <div class="cv-badges" style="flex-shrink:0">
                  {sent_badge(sentiment)}
                  {fake_badge(fake_lbl)}
                </div>
              </div>
              
              <div class="cv-text">{display_text}</div>
              
              {asp_html}
              
              <div class="cv-meta" style="margin-top:12px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px">
                <div style="display:flex;align-items:center;gap:16px;font-size:11px">
                  <div style="display:flex;align-items:center;gap:5px">
                    <span style="font-weight:700;color:var(--ink)">Rating:</span>
                    <span style="font-family:'DM Serif Display',serif;font-size:14px;color:var(--amber)">{rating}</span>
                  </div>
                  <div style="display:flex;align-items:center;gap:5px">
                    <span style="font-weight:700;color:var(--ink)">Confidence:</span>
                    <span style="font-weight:700;color:var(--teal)">{conf_pct}%</span>
                    {conf_bar}
                  </div>
                </div>
                <div style="font-size:10px;color:var(--muted);font-family:'JetBrains Mono',monospace">#{idx} of {n_filtered}</div>
              </div>
              
              {f'<div style="margin-top:10px;padding:10px 12px;background:rgba(176,96,0,.04);border-radius:8px;border:1px solid rgba(176,96,0,.12);font-size:11px;color:#b06000;font-weight:600">{fake_chip}</div>' if fake_chip else ''}
            </div>
            """, unsafe_allow_html=True)

    # ── Pagination ────────────────────────────────────────────────────────────
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    if n_pages > 1:
        pg1, pg2, pg3 = st.columns([2, 3, 2], gap="small")
        with pg1:
            if st.button("← Previous", disabled=(cur_page == 0), key="cv_prev"):
                st.session_state.cv_page = cur_page - 1
                st.rerun()
        with pg2:
            st.markdown(
                f'<div style="text-align:center;font-size:12px;color:var(--muted);padding-top:10px;font-family:\'JetBrains Mono\',monospace">'
                f'Page {cur_page+1} / {n_pages}  ·  {n_filtered:,} results</div>',
                unsafe_allow_html=True,
            )
        with pg3:
            if st.button("Next →", disabled=(cur_page >= n_pages - 1), key="cv_next"):
                st.session_state.cv_page = cur_page + 1
                st.rerun()