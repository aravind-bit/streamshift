# app.py — Media Merger Analysis (stable + WOW v2)
# ------------------------------------------------
# - Centered title/subtitle
# - Buyer/Target selectors (no "Generate" button)
# - Rippleboard (IP, Predicted Status, Notes, Platform, Tags)
# - Originals from the target (heuristic tagging if missing)
# - Sources / Traceability (links if present, TMDB search helper otherwise)
# - Headline Mood (quick VADER sentiment on data/headlines.csv)
# - IP Similarity Map WOW v2 (Sentence-Transformers -> UMAP/PCA -> KMeans)
#   with full fallbacks so the app never breaks.

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Plotting ----------
import plotly.express as px

# ---------- ML (with graceful fallbacks) ----------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

try:
    import umap  # type: ignore
    _UMAP_OK = True
except Exception:
    _UMAP_OK = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST_OK = True
except Exception:
    _ST_OK = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    _VADER_OK = True
except Exception:
    _VADER_OK = False


# ----------------------------
# Streamlit page configuration
# ----------------------------
st.set_page_config(
    page_title="Media Merger Analysis",
    page_icon="🎬",
    layout="wide",
)

# ----------------------------
# Utilities / IO
# ----------------------------
DATA_DIR = Path("data")
FRANCHISES_CSV = DATA_DIR / "franchises.csv"
HEADLINES_CSV = DATA_DIR / "headlines.csv"

FRIENDLY_BUYERS = [
    "Amazon",
    "Warner Bros. Discovery",
    "Paramount Global",
    "Comcast (NBCUniversal)",
    "Disney",
    "Netflix",
    "Apple",
    "Sony",
]

STREAMERS_BY_BRAND = {
    "Amazon": ["Prime Video", "Amazon Prime Video"],
    "Warner Bros. Discovery": ["Max", "HBO Max"],
    "Paramount Global": ["Paramount+", "Paramount Plus"],
    "Comcast (NBCUniversal)": ["Peacock"],
    "Disney": ["Disney+"],
    "Netflix": ["Netflix"],
    "Apple": ["Apple TV+"],
    "Sony": ["Crunchyroll", "SonyLIV"],
}


# ----------------------------
# Load / cache data
# ----------------------------
@st.cache_data(show_spinner=False)
def load_franchises(path: Path) -> pd.DataFrame:
    if not path.exists():
        # Create a tiny stub so app never crashes
        df = pd.DataFrame(
            [
                {
                    "title": "Stranger Things",
                    "predicted_status": "Exclusive Distribution",
                    "notes": "High-value IP → likely exclusive under Amazon.",
                    "current_platform": "Netflix, Netflix Standard with Ads",
                    "tags": "drama, sci-fi, teens",
                    "origin_label": "Exclusive Distribution",
                    "source_urls": "",
                },
                {
                    "title": "Game of Thrones",
                    "predicted_status": "Licensed",
                    "notes": "Not a flagship—probably stays put for now.",
                    "current_platform": "Max",
                    "tags": "fantasy, drama, epic",
                    "origin_label": "Licensed",
                    "source_urls": "",
                },
            ]
        )
        return df
    df = pd.read_csv(path)
    # Normalize column names we rely on
    want = [
        "title",
        "predicted_status",
        "status",
        "notes",
        "current_platform",
        "platform",
        "tags",
        "origin_label",
        "original_network",
        "distributor_list",
        "source_urls",
    ]
    for w in want:
        if w not in df.columns:
            df[w] = ""
    # pick one status column
    if "predicted_status" not in df.columns and "status" in df.columns:
        df["predicted_status"] = df["status"]
    elif "predicted_status" in df.columns and df["predicted_status"].isna().all() and "status" in df.columns:
        df["predicted_status"] = df["status"]
    # unify naming for platform
    if df["platform"].astype(str).str.strip().ne("").any() and df["current_platform"].astype(str).str.strip().eq("").all():
        df["current_platform"] = df["platform"]
    return df


@st.cache_data(show_spinner=False)
def load_headlines(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["brand", "headline", "link", "date"])
    df = pd.read_csv(path)
    for col in ["brand", "headline", "link", "date"]:
        if col not in df.columns:
            df[col] = ""
    return df


df = load_franchises(FRANCHISES_CSV)
headlines = load_headlines(HEADLINES_CSV)


# ----------------------------
# Centered Title / Subtitle
# ----------------------------
st.markdown(
    """
    <div style="text-align:center; margin-top: 8px;">
      <h1 style="margin-bottom: 0.25rem; font-size: 2.2rem;">Media Merger Analysis</h1>
      <div style="opacity:0.85; font-size:1.05rem;">
        Who streams what after the deal.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")  # small spacer


# ----------------------------
# Buyer / Target selectors
# ----------------------------
col_b, col_t = st.columns(2)
with col_b:
    st.markdown("#### Buyer")
    buyer = st.selectbox("", FRIENDLY_BUYERS, index=0, label_visibility="collapsed")
with col_t:
    st.markdown("#### Target")
    target = st.selectbox("", FRIENDLY_BUYERS, index=1, label_visibility="collapsed")

# keep in session for mood focus
st.session_state["buyer_label"] = buyer
st.session_state["target_label"] = target


# -------------------------------------------------------
# Heuristic: originals tagging if data is thin / inconsistent
# -------------------------------------------------------
def tag_originals_heuristic(frame: pd.DataFrame, brand: str) -> pd.Series:
    """
    Return a boolean Series "is_original" based on origin_label/platform/network
    matching the target brand's known services. Never crashes if columns are missing.
    """
    servs = STREAMERS_BY_BRAND.get(brand, [])
    plat = frame.get("current_platform", pd.Series([""] * len(frame))).astype(str).str.lower()
    netw = frame.get("original_network", pd.Series([""] * len(frame))).astype(str).str.lower()
    label = frame.get("origin_label", pd.Series([""] * len(frame))).astype(str).str.lower()

    # brand match list
    pats = [s.lower() for s in servs] + [brand.lower()]
    regex = "|".join([re.escape(p) for p in pats if p])

    condition = pd.Series([False] * len(frame))
    if regex:
        condition |= plat.str.contains(regex, regex=True, na=False)
        condition |= netw.str.contains(regex, regex=True, na=False)

    # strong label wins
    condition |= label.str.contains("original", na=False)
    condition |= label.str.contains("exclusive", na=False)

    return condition.fillna(False)


# ----------------------------
# Rippleboard table
# ----------------------------
st.markdown("### Rippleboard: The Future of Content")
st.caption(
    "A quick ‘what happens next’ table for marquee titles. "
    "Plain-English notes keep it explainable and hobby-friendly."
)

# Build skinny view
ripple_cols = [
    ("title", "IP / Franchise"),
    ("predicted_status", "Predicted Status"),
    ("notes", "Notes"),
    ("current_platform", "Current Platform"),
    ("tags", "Tags"),
]
ripple = pd.DataFrame({new: df.get(old, "") for old, new in ripple_cols})
st.dataframe(ripple, use_container_width=True, hide_index=True, height=380)


# ----------------------------
# IP Similarity Map (WOW v2)
# ----------------------------
st.markdown("### IP Similarity Map")
st.caption(
    "We turn titles & tags into vectors (story DNA), reduce to 2-D with UMAP/PCA, "
    "then color clusters with K-Means. Close dots ⇒ similar vibe/genre."
)

def _prep_text(_df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ["title", "tags", "notes"] if c in _df.columns]
    if not cols:
        return _df["title"].astype(str)
    return (
        _df[cols].fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

@st.cache_resource(show_spinner=False)
def _load_st_model():
    if not _ST_OK:
        return None
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_data(show_spinner=False)
def _embed_texts(texts: List[str]) -> np.ndarray:
    model = _load_st_model()
    if model is not None:
        embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return embs
    # fallback: TF-IDF → PCA(100)
    tfidf = TfidfVectorizer(min_df=1, max_features=5000, ngram_range=(1, 2))
    X = tfidf.fit_transform(texts)
    n_comp = max(2, min(100, X.shape[1] - 1))
    pca = PCA(n_components=n_comp, random_state=42)
    return pca.fit_transform(X.toarray())

@st.cache_data(show_spinner=False)
def _reduce_2d(embs: np.ndarray) -> np.ndarray:
    if _UMAP_OK and embs.shape[0] >= 6:
        reducer = umap.UMAP(n_components=2, min_dist=0.15, n_neighbors=10, random_state=42)
        return reducer.fit_transform(embs)
    pca2 = PCA(n_components=2, random_state=42)
    return pca2.fit_transform(embs)

@st.cache_data(show_spinner=False)
def _cluster(pts: np.ndarray, k: int = 4) -> np.ndarray:
    k = max(2, min(k, len(pts)))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    return km.fit_predict(pts)

_text = _prep_text(df)
_embs = _embed_texts(_text.tolist())
_xy = _reduce_2d(_embs)
_lbls = _cluster(_xy, k=4)

DEFAULT_CLUSTER_NAMES = {
    0: "Teen Sci-Fi / Fantasy",
    1: "Crime / Spy / Thriller",
    2: "Comedy / Workplace",
    3: "Epic / Prestige Drama",
}

df_map = pd.DataFrame(
    {
        "x": _xy[:, 0],
        "y": _xy[:, 1],
        "cluster": _lbls,
        "title": df["title"].astype(str),
        "platform": df.get("current_platform", pd.Series([""] * len(df))).astype(str),
        "tags": df.get("tags", pd.Series([""] * len(df))).astype(str),
        "notes": df.get("notes", pd.Series([""] * len(df))).astype(str),
    }
)
df_map["cluster_name"] = df_map["cluster"].map(lambda c: DEFAULT_CLUSTER_NAMES.get(c, f"Group {c}"))
df_map["hover"] = (
    "**" + df_map["title"] + "**"
    + "<br><b>Cluster:</b> " + df_map["cluster_name"]
    + "<br><b>Platform:</b> " + df_map["platform"].str.split(",").str[0].fillna("")
    + "<br><b>Tags:</b> " + df_map["tags"].fillna("")
    + "<br><b>Note:</b> " + df_map["notes"].fillna("")
)

fig = px.scatter(
    df_map,
    x="x",
    y="y",
    color="cluster_name",
    hover_name="title",
    hover_data={"x": False, "y": False, "cluster": False, "cluster_name": False, "platform": False, "tags": False, "notes": False},
    custom_data=["hover"],
    opacity=0.9,
    template="plotly_dark",
    height=420,
)
fig.update_traces(marker=dict(size=10, line=dict(width=0)))
fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Clusters are data-driven and lightly labeled. Edit names in `DEFAULT_CLUSTER_NAMES` "
    "to match your intuition (e.g., ‘Superhero’, ‘Family Animation’)."
)


# ----------------------------
# Originals from the target
# ----------------------------
st.markdown("### Originals from the target")
st.caption(
    "Target-brand originals inferred from platform/network/labels. "
    "If your CSV is thin, this heuristic helps avoid an empty table."
)

is_orig = tag_originals_heuristic(df, target)
orig_view = pd.DataFrame(
    {
        "title": df["title"],
        "original_network": df.get("original_network", ""),
        "producer_list": df.get("producer_list", ""),
        "platform": df.get("current_platform", df.get("platform", "")),
        "predicted_policy": df.get("predicted_status", df.get("status", "")),
    }
)[is_orig]

if len(orig_view):
    st.dataframe(orig_view, use_container_width=True, hide_index=True, height=260)
else:
    st.info("No clear originals detected for this small demo—add a few and rerun enrichment.")


# ----------------------------
# Sources / Traceability
# ----------------------------
st.markdown("### Sources / Traceability (for titles shown)")
st.caption(
    "Saved source URLs appear here. If missing, we show a quick TMDB search link so you can verify manually."
)

def _to_first_link(cell: str) -> str:
    if not cell or pd.isna(cell):
        return ""
    # accept pipe/comma separated sources
    parts = [p.strip() for p in str(cell).split("|") if p.strip()] or [p.strip() for p in str(cell).split(",") if p.strip()]
    return parts[0] if parts else ""

trace_df = pd.DataFrame(
    {
        "title": df["title"],
        "source": df["source_urls"].apply(_to_first_link),
    }
)

def tmdb_link(title: str) -> str:
    q = re.sub(r"\s+", "+", title.strip())
    return f"https://www.themoviedb.org/search?query={q}"

def linkify(row) -> str:
    if row["source"]:
        return f"[open]({row['source']})"
    return f"[search TMDB]({tmdb_link(row['title'])})"

trace_df["link"] = trace_df.apply(linkify, axis=1)
trace_view = trace_df[["title", "link"]]
st.dataframe(trace_view, use_container_width=True, hide_index=True, height=220)


# ----------------------------
# Headline Mood (quick check)
# ----------------------------
st.markdown("### Headline Mood (quick check)")
st.caption("Tiny NLP pulse based on recent headlines for the selected brands. Not investment advice—just a vibe check.")

def _score_texts(texts: List[str]) -> List[float]:
    if not _VADER_OK:
        return [np.nan] * len(texts)
    analyzer = SentimentIntensityAnalyzer()
    return [analyzer.polarity_scores(t)["compound"] for t in texts]

if len(headlines):
    focus = {buyer, target}
    hh = headlines.copy()
    hh = hh[hh["brand"].isin(focus)] if focus else hh
    hh = hh.copy()
    hh["sentiment"] = _score_texts(hh["headline"].fillna(""))

    c1, c2 = st.columns((2, 3))
    with c1:
        if _VADER_OK and len(hh):
            s = hh.groupby("brand")["sentiment"].mean().reset_index()
            s["sentiment"] = s["sentiment"].round(3)
            st.dataframe(s, use_container_width=True, hide_index=True)
        else:
            st.info("Add a few headlines to `data/headlines.csv` and ensure `vaderSentiment` is installed.")

    with c2:
        if _VADER_OK and len(hh):
            chart_df = hh.groupby("brand")["sentiment"].mean().to_frame()
            chart_df.columns = ["Mood"]
            st.bar_chart(chart_df, height=180)
        else:
            st.empty()

    st.write("**Recent headlines**")
    for _, row in hh.sort_values("date", ascending=False).head(6).iterrows():
        st.write(f"- **{row['brand']}** — [{row['headline']}]({row['link']})  \n  _{row['date']}_")
else:
    st.info("Optional: create `data/headlines.csv` with columns: brand, headline, link, date.")


# ----------------------------
# Footer (tiny)
# ----------------------------
st.write("")
st.caption("Built as a hobby project. Data may be incomplete—meant for discussion, not investment decisions.")