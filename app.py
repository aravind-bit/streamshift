# app.py
# Media Merger Analysis — "Who streams what after the deal."
# Minimal, explainable, hobby-friendly app with a few smart flourishes:
# - Rippleboard: plain-English notes + predicted status
# - IP Similarity Map: embeddings -> UMAP/PCA -> KMeans clusters (+ graceful fallback)
# - Originals from the target: heuristics so it never looks empty
# - Sources / Traceability: links if present; otherwise quick TMDB search
# - Headline Mood: tiny VADER sentiment over a small CSV of headlines

from __future__ import annotations

import os
import re
import json
import math
import textwrap
from typing import List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- Optional NLP/ML deps (guarded imports) ---
try:
    from sentence_transformers import SentenceTransformer
    _st_ok = True
except Exception:
    _st_ok = False

try:
    import umap
    _umap_ok = True
except Exception:
    _umap_ok = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader_ok = True
except Exception:
    _vader_ok = False


# =========================================
# Page config + tiny CSS for nicer headline
# =========================================
st.set_page_config(page_title="Media Merger Analysis", page_icon="🎬", layout="wide")

# Small CSS touches (keeps your “older” aesthetic)
st.markdown(
    """
    <style>
      .app-title { font-size: 40px; font-weight: 800; letter-spacing: 0.5px; text-align: center; margin-top: 0.25rem; }
      .app-subtitle { font-size: 16px; opacity: 0.85; text-align: center; margin-bottom: 1.25rem; }
      .section-note { opacity: 0.85; font-size: 13.5px; margin-top: -6px; margin-bottom: 8px; }
      .metric-chip { padding: 3px 8px; border-radius: 999px; background: rgba(148, 124, 255, .18); border: 1px solid rgba(148,124,255,.35); font-size: 12px; }
      .small-muted { font-size: 12.5px; opacity: 0.8; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">Media Merger Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Who streams what after the deal.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="app-subtitle" style="margin-top:-10px;">'
    "Visual sandbox for hypothetical media mergers — pick a buyer & target to see "
    "a Rippleboard of outcomes, an IP similarity map, and traceable sources. "
    "It’s a hobby project, not investment advice."
    "</p>",
    unsafe_allow_html=True,
)

# ==================
# Utility / Loaders
# ==================
DATA_FRANCHISES = "data/franchises.csv"
DATA_HEADLINES = "data/headlines.csv"

@st.cache_data(show_spinner=False)
def load_franchises(path: str = DATA_FRANCHISES) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=[
            "title", "current_platform", "tags", "notes",
            "predicted_status", "origin_label", "origin_brand", "source_urls"
        ])
    df = pd.read_csv(path)
    # normalize expected columns
    for c in ["title", "current_platform", "tags", "notes", "predicted_status",
              "origin_label", "origin_brand", "source_urls"]:
        if c not in df.columns:
            df[c] = np.nan
    # clean strings
    for c in ["title", "current_platform", "tags", "notes", "origin_label", "origin_brand", "source_urls"]:
        df[c] = df[c].astype(str).replace({"nan":"", "None":""})
    return df

@st.cache_data(show_spinner=False)
def load_headlines(path: str = DATA_HEADLINES) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["brand","headline","link","date"])
    df = pd.read_csv(path)
    for c in ["brand","headline","link","date"]:
        if c not in df.columns:
            df[c] = ""
    return df

def _list_unique_brands_from_platforms(df: pd.DataFrame) -> List[str]:
    """Extract rough brand candidates from current_platform strings."""
    if "current_platform" not in df.columns:
        return []
    brands = set()
    for s in df["current_platform"].dropna().astype(str).tolist():
        # split by comma to get first platforms
        parts = [p.strip() for p in s.split(",") if p.strip()]
        for p in parts:
            # simplify “Netflix Standard with Ads” -> “Netflix”
            base = re.split(r"\s+(Standard|with|Plus|TV|Channel|Premium)", p)[0].strip()
            if base:
                brands.add(base)
    # canonical additions seen in UI
    brands.update(["Amazon", "Netflix", "Max", "Paramount+", "Peacock", "Apple TV+", "Disney+"])
    return sorted(brands)

def _tmdb_link(title: str) -> str:
    q = re.sub(r"\s+", "%20", title.strip())
    return f"https://www.themoviedb.org/search?query={q}"

def _is_flagship(title: str) -> bool:
    """Very small demo heuristic (you can externalize as CSV)."""
    flagship = {
        "Stranger Things", "Game of Thrones", "Harry Potter", "The Lord of the Rings",
        "James Bond", "The Witcher", "The Last of Us", "House of the Dragon"
    }
    return title.strip().lower() in {t.lower() for t in flagship}

def _predict_status_and_note(row: pd.Series, buyer: str, target: str) -> Tuple[str, str]:
    """
    Simple, friendly rules:
      - If a title feels 'flagship' of the target -> Exclusive Distribution under buyer (if aligned) or 'stay' if conflict.
      - Else Licensed (stays put for now).
    It's intentionally explainable, not a black box.
    """
    title = (row.get("title") or "").strip()
    platform = (row.get("current_platform") or "").lower()
    buyer_l = (buyer or "").lower()
    target_l = (target or "").lower()

    # very small/regret-free logic:
    if _is_flagship(title):
        if buyer_l in platform:
            return "Exclusive Distribution", "High-value IP → likely exclusive under buyer."
        if target_l in platform or (target_l and target_l.split()[0] in platform):
            return "Exclusive Distribution", "Flagship of the target; exclusivity likely after integration."
        return "Exclusive Distribution", "Flagship title; consolidation would push exclusive window."
    # non-flagship
    return "Licensed", "Not a flagship—probably stays put for now."

def _split_urls(cell: str) -> List[str]:
    if not cell or cell.strip().lower() in {"nan","none"}:
        return []
    # support ';' or '|' or ',' separated
    parts = re.split(r"[;|,]", cell)
    return [p.strip() for p in parts if p.strip()]

# ==================================
# Session state for Buyer / Target
# ==================================
df = load_franchises()
brand_options = _list_unique_brands_from_platforms(df)

col_a, col_b, col_btn = st.columns([3,3,2])
with col_a:
    buyer = st.selectbox("Buyer", brand_options or ["Amazon", "Netflix", "Max", "Paramount+", "Peacock", "Apple TV+", "Disney+"], index=0)
with col_b:
    target = st.selectbox("Target", brand_options or ["Netflix", "Max", "Paramount+", "Peacock", "Amazon", "Apple TV+", "Disney+"], index=1)
with col_btn:
    st.write("")  # spacing
    run = st.button("Generate Analysis", use_container_width=True)

# stash names for other sections
st.session_state["buyer_label"] = buyer
st.session_state["target_label"] = target

# ============
# Rippleboard
# ============
st.markdown("### Rippleboard: The Future of Content")
st.markdown(
    '<div class="section-note">'
    "A quick ‘what happens next’ view for key IP: predicted status + a plain-English note."
    "</div>",
    unsafe_allow_html=True,
)

view_cols = ["title", "predicted_status", "notes", "current_platform", "tags"]
rows = []
for _, r in df.iterrows():
    status = (r.get("predicted_status") or "").strip()
    note = (r.get("notes") or "").strip()
    if not status:
        status, auto_note = _predict_status_and_note(r, buyer, target)
        if not note:
            note = auto_note
    rows.append({
        "title": r.get("title", ""),
        "predicted_status": status,
        "notes": note,
        "current_platform": r.get("current_platform", ""),
        "tags": r.get("tags", "")
    })
rb = pd.DataFrame(rows, columns=view_cols)

st.dataframe(
    rb,
    use_container_width=True,
    hide_index=True,
)

# =======================================
# Originals from the target (distribution)
# =======================================
st.markdown("### Originals from the target")
st.markdown(
    '<div class="section-note">'
    "Titles that look first-party to the target (brand/network/platform cues). "
    "If metadata is thin, we infer from platform/network so this section isn’t blank."
    "</div>",
    unsafe_allow_html=True,
)

def _looks_like_original(row: pd.Series, target_brand: str) -> bool:
    tg = (target_brand or "").lower()
    if not tg:
        return False
    nets = f"{row.get('origin_brand','')} {row.get('origin_label','')} {row.get('current_platform','')}".lower()
    return tg in nets

orig = df[df.apply(lambda r: _looks_like_original(r, target), axis=1)].copy()

originals_cols = ["title", "origin_brand", "origin_label", "current_platform", "source_urls"]
if len(orig) == 0:
    st.info("No clear originals discovered in this small demo—add a few and rerun enrichment.")
else:
    st.dataframe(
        orig[originals_cols],
        use_container_width=True,
        hide_index=True,
    )

# ========================================
# Sources / Traceability (for titles shown)
# ========================================
st.markdown("### Sources / Traceability (for titles shown)")
st.markdown(
    '<div class="section-note">'
    "Receipts for the titles in view. If a row has no source URL, we offer a quick TMDB search link."
    "</div>",
    unsafe_allow_html=True,
)

def _trace_rows(df_like: pd.DataFrame) -> pd.DataFrame:
    out = []
    seen = set()
    for _, r in df_like.iterrows():
        title = (r.get("title") or "").strip()
        if not title or title in seen:
            continue
        seen.add(title)
        urls = _split_urls(r.get("source_urls", ""))
        if not urls:
            urls = [_tmdb_link(title)]
        out.append({"title": title, "source": urls[0]})
    return pd.DataFrame(out)

trace_df = _trace_rows(df)
if len(trace_df) == 0:
    st.info("Add/refresh data to see TMDB/Wikidata links here (store in `source_urls`).")
else:
    st.dataframe(trace_df, use_container_width=True, hide_index=True)

# =========================
# IP Similarity Map (WOW v2)
# =========================
st.markdown("### IP Similarity Map")
st.markdown(
    '<div class="section-note">'
    "We turn title + tags + notes into vectors (story DNA), reduce to 2-D with UMAP/PCA, then color clusters with K-Means. "
    "Close dots ⇒ similar vibe/genre. Labels are editable."
    "</div>",
    unsafe_allow_html=True,
)

def _prep_text(df_in: pd.DataFrame) -> pd.Series:
    cols = [c for c in ["title", "tags", "notes"] if c in df_in.columns]
    if not cols:
        return df_in["title"].astype(str)
    return (
        df_in[cols].fillna("")
        .astype(str).agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

@st.cache_resource(show_spinner=False)
def load_st_model():
    if not _st_ok:
        return None
    # tiny + fast
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_data(show_spinner=False)
def embed_texts(texts: List[str]) -> np.ndarray:
    model = load_st_model()
    if model is not None:
        embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return embs
    # fallback TF-IDF -> PCA(100)
    tfidf = TfidfVectorizer(min_df=1, max_features=5000, ngram_range=(1, 2))
    X = tfidf.fit_transform(texts)
    # guard for tiny matrices
    n_comp = min(100, max(2, X.shape[1]-1))
    pca = PCA(n_components=n_comp, random_state=42)
    return pca.fit_transform(X.toarray())

@st.cache_data(show_spinner=False)
def reduce_2d(embs: np.ndarray) -> np.ndarray:
    if _umap_ok and embs.shape[0] >= 6:
        reducer = umap.UMAP(n_components=2, min_dist=0.15, n_neighbors=10, random_state=42)
        return reducer.fit_transform(embs)
    pca2 = PCA(n_components=2, random_state=42)
    return pca2.fit_transform(embs)

@st.cache_data(show_spinner=False)
def cluster_labels(xy: np.ndarray, k: int = 4) -> np.ndarray:
    k = max(2, min(k, len(xy)))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    return km.fit_predict(xy)

text_for_map = _prep_text(df)
if len(text_for_map) >= 2:
    embs = embed_texts(text_for_map.tolist())
    xy = reduce_2d(embs)
    labels = cluster_labels(xy, k=4)

    # Friendly cluster names — edit for taste
    DEFAULT_CLUSTER_NAMES = {
        0: "Teen Sci-Fi / Fantasy",
        1: "Crime / Spy / Thriller",
        2: "Comedy / Workplace",
        3: "Epic / Prestige Drama",
    }

    m = pd.DataFrame({
        "x": xy[:, 0],
        "y": xy[:, 1],
        "cluster": labels,
        "title": df["title"].astype(str),
        "platform": (df.get("current_platform") or pd.Series([""]*len(df))).astype(str),
        "tags": (df.get("tags") or pd.Series([""]*len(df))).astype(str),
        "notes": (df.get("notes") or pd.Series([""]*len(df))).astype(str),
    })
    m["cluster_name"] = m["cluster"].map(lambda c: DEFAULT_CLUSTER_NAMES.get(c, f"Group {c}"))
    m["hover"] = (
        "**" + m["title"] + "**" +
        "<br><b>Cluster:</b> " + m["cluster_name"] +
        "<br><b>Platform:</b> " + m["platform"].str.split(",").str[0].fillna("") +
        "<br><b>Tags:</b> " + m["tags"].fillna("") +
        "<br><b>Note:</b> " + m["notes"].fillna("")
    )

    fig = px.scatter(
        m, x="x", y="y",
        color="cluster_name",
        hover_name="title",
        hover_data={"x": False, "y": False, "cluster": False, "cluster_name": False, "platform": False, "tags": False, "notes": False},
        custom_data=["hover"],
        template="plotly_dark",
        opacity=0.9,
        height=420,
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=0)))
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Clusters are data-driven and lightly labeled. Edit names in `DEFAULT_CLUSTER_NAMES` to match your intuition "
        "(e.g., “Superhero”, “Family Animation”)."
    )
else:
    st.info("Need at least 2 titles to draw the map.")

# ==========================
# Headline Mood (quick NLP)
# ==========================
st.markdown("### Headline Mood (quick check)")
st.markdown(
    '<div class="section-note">'
    "Tiny NLP pulse for selected brands using VADER. It’s just a vibe check."
    "</div>",
    unsafe_allow_html=True,
)

def _score_texts(texts: List[str]) -> List[float]:
    if not _vader_ok:
        return [math.nan] * len(texts)
    analyzer = SentimentIntensityAnalyzer()
    return [analyzer.polarity_scores(t or "")["compound"] for t in texts]

hh = load_headlines()
focus = set()
if st.session_state.get("buyer_label"):
    focus.add(st.session_state["buyer_label"])
if st.session_state.get("target_label"):
    focus.add(st.session_state["target_label"])

if len(hh) == 0:
    st.info("Optional: add `data/headlines.csv` (brand, headline, link, date) to enable mood check.")
else:
    hh = hh.copy()
    if focus:
        hh = hh[hh["brand"].isin(focus)] if not hh.empty else hh
    hh["sentiment"] = _score_texts(hh["headline"].fillna(""))

    c1, c2 = st.columns((2,3))
    with c1:
        if _vader_ok and len(hh):
            s = hh.groupby("brand")["sentiment"].mean().reset_index()
            s["sentiment"] = s["sentiment"].round(3)
            st.dataframe(s, use_container_width=True, hide_index=True)
        else:
            st.info("Install `vaderSentiment` and add headlines to `data/headlines.csv`.")

    with c2:
        if _vader_ok and len(hh):
            chart_df = hh.groupby("brand")["sentiment"].mean().to_frame()
            chart_df.columns = ["Mood"]
            st.bar_chart(chart_df, height=200)
        else:
            st.empty()

    if len(hh):
        st.write("**Recent headlines**")
        for _, row in hh.sort_values("date", ascending=False).head(6).iterrows():
            brand = row.get("brand","")
            headline = row.get("headline","")
            link = row.get("link","")
            date = row.get("date","")
            st.write(f"- **{brand}** — [{headline}]({link})  \n  _{date}_")

# end of app