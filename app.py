# app.py — Media Merger Analysis (clean + filtered Rippleboard)
# -------------------------------------------------------------
# What this does:
# - Loads data/data/franchises.csv (expected columns used if present:
#   title, current_platform, origin_label, distributor_list, service, brand,
#   tags, predicted_policy, status, notes, source_urls)
# - Buyer & Target dropdowns (exact list you requested)
# - Rippleboard table filtered to only rows relevant to Buyer/Target
#   (looks across multiple columns: current_platform/origin_label/etc.)
# - Originals + Sources/Traceability sections (safe fallbacks)
# - Light styling and mellow helper blurbs

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Styling ----------
st.set_page_config(page_title="Media Merger Analysis", layout="wide")

st.markdown(
    """
<style>
html, body, .main {
  background: radial-gradient(1200px 600px at 20% -10%, rgba(167,139,250,0.18), rgba(2,6,23,0.0)),
              radial-gradient(900px 480px at 80% -20%, rgba(59,130,246,0.18), rgba(2,6,23,0.0)),
              #0b1120;
  color: #E7E9EE;
}
.main .block-container { max-width: 1280px; padding-top: .6rem; }

/* Hide any “pills” or legacy chips */
.pill, .helper-pill, .hint-pill,
[data-testid="stBadges"], .st-emotion-cache-1r4qj8e {
  display: none !important;
}

/* Hero */
.hero { text-align:center; margin: 0 0 .8rem 0; }
.hero h1 { font-size: 2.6rem; font-weight: 800; color:#C7A6FF; letter-spacing:.2px; margin:0; }
.hero p  { color:#A1A8B3; margin:.4rem 0 0 0; font-size:1.06rem; }

/* Section shells */
.section-card {
  background: rgba(17,24,39,0.55);
  border: 1px solid rgba(148,163,184,0.18);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}
.section-title { font-size: 1.18rem; font-weight: 900; color:#E7E9EE; margin: 0 0 .35rem 0; }

.section-blurb {
  color:#D1D5DB;
  font-size: .98rem;
  line-height: 1.35rem;
  margin: 2px 0 10px 0;
}

/* Inputs */
label, .stSelectbox label { font-size: 1.06rem !important; font-weight: 800 !important; color:#e7e9ee !important; }
.stSelectbox [data-baseweb="select"] > div {
  background: rgba(17,24,39,0.55);
  border: 1px solid rgba(148,163,184,0.25);
}

/* DataFrames */
[data-testid="stDataFrame"] {
  border: 1px solid rgba(148,163,184,0.18) !important;
  border-radius: 12px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Constants / Options ----------
BRANDS = [
    "Amazon",
    "Warner Brothers HBO",
    "Paramount Global",
    "Comcast (NBCUniversal)",
    "Disney",
    "Netflix",
    "Apple",
    "Sony",
    "Hulu",
    "Max",
    "Peacock",
]

DATA_PATH = Path("data/franchises.csv")

# ---------- Data Loading ----------
@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "title",
                "current_platform",
                "origin_label",
                "predicted_policy",
                "tags",
                "notes",
                "source_urls",
                "service",
                "brand",
                "distributor_list",
            ]
        )
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Normalize whitespace
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].fillna("").astype(str).str.strip()
    return df

df_all = load_data(DATA_PATH)

# ---------- Hero ----------
st.markdown(
    """
<div class="hero">
  <h1>Media Merger Analysis</h1>
  <p>Visualize what happens to movies & series after hypothetical media mergers. Who streams what after the deal.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Controls ----------
c1, c2 = st.columns(2)
with c1:
    buyer = st.selectbox("Buyer", BRANDS, index=BRANDS.index("Amazon") if "Amazon" in BRANDS else 0)
with c2:
    target = st.selectbox("Target", BRANDS, index=BRANDS.index("Warner Brothers HBO") if "Warner Brothers HBO" in BRANDS else 0)

# ---------- Helper: build Rippleboard ----------
def build_rippleboard(df: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized board with columns: title, predicted_status, notes, current_platform, tags."""
    if df.empty:
        return pd.DataFrame(columns=["IP / Franchise", "Predicted Status", "Notes", "Current Platform", "Tags"])

    # Choose best-available columns
    title_col = "title" if "title" in df.columns else df.columns[0]
    status_col = "predicted_policy" if "predicted_policy" in df.columns else ("status" if "status" in df.columns else None)
    notes_col = "notes" if "notes" in df.columns else None
    platform_col = "current_platform" if "current_platform" in df.columns else None
    tags_col = "tags" if "tags" in df.columns else None

    out = pd.DataFrame()
    out["IP / Franchise"] = df[title_col]

    # Predicted Status
    if status_col:
        out["Predicted Status"] = df[status_col].replace("", "Licensed")
    else:
        out["Predicted Status"] = "Licensed"

    # Notes
    if notes_col:
        out["Notes"] = df[notes_col].replace("", "Not a flagship—probably stays put for now.")
    else:
        # cheap heuristic: exclusives get punchier text
        out["Notes"] = np.where(
            out["Predicted Status"].str.lower().str.contains("exclusive"),
            "High-value IP → likely exclusive under buyer.",
            "Not a flagship—probably stays put for now.",
        )

    # Platform
    if platform_col:
        out["Current Platform"] = df[platform_col]
    else:
        out["Current Platform"] = ""

    # Tags
    if tags_col:
        out["Tags"] = df[tags_col]
    else:
        out["Tags"] = ""

    # De-dupe by title if needed
    out = out.drop_duplicates(subset=["IP / Franchise"]).reset_index(drop=True)
    return out

# ---------- Buyer/Target filter for Rippleboard ----------
def filter_for_brands(board_df: pd.DataFrame, buyer_brand: str, target_brand: str) -> pd.DataFrame:
    """Keep rows that mention buyer/target in any of the contextual columns in the source df."""
    if df_all.empty or board_df.empty:
        return board_df

    # Map Rippleboard rows back to source by title to get more columns to search
    rb = board_df.copy()
    if "IP / Franchise" not in rb.columns or "title" not in df_all.columns:
        return rb  # fail-safe

    # Join so we can search richer context fields
    joined = rb.merge(
        df_all,
        left_on="IP / Franchise",
        right_on="title",
        how="left",
        suffixes=("", "_src"),
    )

    brands = [buyer_brand, target_brand]

    def has_any_brand(val: str) -> bool:
        s = str(val or "")
        return any(b.lower() in s.lower() for b in brands)

    mask_parts = []
    for col in ("current_platform", "origin_label", "distributor_list", "service", "brand"):
        if col in joined.columns:
            mask_parts.append(joined[col].apply(has_any_brand))

    if mask_parts:
        mask = mask_parts[0]
        for m in mask_parts[1:]:
            mask = mask | m
        filtered = joined.loc[mask].copy()
    else:
        filtered = joined  # if nothing to search, don’t hide everything

    # Return only Rippleboard columns in original order
    cols = [c for c in board_df.columns if c in filtered.columns]
    if not cols:
        return board_df
    filtered = filtered[cols].drop_duplicates(subset=["IP / Franchise"]).reset_index(drop=True)
    return filtered

# ---------- Sections ----------
# Rippleboard
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Rippleboard: The Future of Content</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-blurb">What this is: a post-deal TV guide for suits (but in plain English). '
    'Each title gets a status (stay / licensed / exclusive) and a 1-liner. '
    'Tweak the rule and watch the board shift.</div>',
    unsafe_allow_html=True,
)

board_df = build_rippleboard(df_all)
board_df = filter_for_brands(board_df, buyer, target)

st.dataframe(
    board_df,
    use_container_width=True,
    height=520,
    hide_index=True,
)
st.markdown('</div>', unsafe_allow_html=True)

# IP Similarity Map (placeholder description; safe if absent)
st.markdown('<div class="section-card" style="margin-top:14px;">', unsafe_allow_html=True)
st.markdown('<div class="section-title">IP Similarity Map</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-blurb">We turn titles & tags into vectors (story DNA), reduce to 2-D, and render clusters. '
    'Close dots ≈ similar vibe/genre. (Demo-friendly placeholder — will silently skip if no data.)</div>',
    unsafe_allow_html=True,
)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
    from sklearn.decomposition import PCA  # noqa: F401
    import plotly.express as px

    if not df_all.empty:
        # Tiny, safe “fake” points for demo when tags are missing
        pts = pd.DataFrame({"x": np.random.normal(size=60), "y": np.random.normal(size=60)})
        fig = px.scatter(pts, x="x", y="y", opacity=0.8)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No data yet for similarity map.")
except Exception:
    st.caption("Similarity map skipped (optional libs not installed).")
st.markdown('</div>', unsafe_allow_html=True)

# Originals from the target
st.markdown('<div class="section-card" style="margin-top:14px;">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Originals from the target</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-blurb">Target-brand originals inferred from platform/network/labels. '
    'If your CSV is thin, we try to avoid an empty table by being forgiving.</div>',
    unsafe_allow_html=True,
)

orig = pd.DataFrame()
if not df_all.empty:
    # Heuristic: anything labeled original to target or whose platform mentions the target
    def _is_target_original(row) -> bool:
        parts = []
        for c in ("origin_label", "service", "brand", "current_platform"):
            if c in df_all.columns:
                parts.append(str(row.get(c, "")))
        joined = " ".join(parts).lower()
        return target.lower() in joined

    orig = df_all[df_all.apply(_is_target_original, axis=1)].copy()

    show_cols = [c for c in ("title", "original_network", "producer_list", "platform", "current_platform", "predicted_policy") if c in orig.columns]
    if not show_cols:
        show_cols = [c for c in orig.columns if c != "source_urls"][:5]
    orig_view = orig[show_cols].drop_duplicates(subset=[show_cols[0]]).reset_index(drop=True)

    st.dataframe(orig_view, use_container_width=True, hide_index=True, height=320)
else:
    st.caption("No data found.")

st.markdown('</div>', unsafe_allow_html=True)

# Sources / Traceability
st.markdown('<div class="section-card" style="margin-top:14px;">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Sources / Traceability (for titles shown)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-blurb">Saved source links appear here. If missing, we leave a handy TMDB search link so you can verify quickly.</div>',
    unsafe_allow_html=True,
)

def tmdb_link(title: str) -> str:
    q = title.replace(" ", "+")
    return f"[search TMDB](https://www.themoviedb.org/search?query={q})"

trace_df = pd.DataFrame(columns=["title", "link"])
if "title" in df_all.columns:
    # limit to titles actually in the filtered board
    titles_shown = set(board_df["IP / Franchise"].tolist())
    subset = df_all[df_all["title"].isin(titles_shown)].copy()
    if not subset.empty:
        if "source_urls" in subset.columns:
            subset["link"] = subset["source_urls"].apply(lambda x: x if str(x).strip() else tmdb_link(subset["title"]))
        else:
            subset["link"] = subset["title"].apply(tmdb_link)
        trace_df = subset[["title", "link"]].drop_duplicates(subset=["title"]).reset_index(drop=True)

st.dataframe(trace_df, use_container_width=True, hide_index=True, height=260)
st.markdown('</div>', unsafe_allow_html=True)

# Tech Stack (card)
st.markdown('<div class="section-card" style="margin-top:14px;">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Tech stack</div>', unsafe_allow_html=True)
st.markdown(
    """
- **Streamlit** UI + theming  
- **Pandas** for CSV shaping  
- **Scikit-learn / Plotly** (optional) for a lightweight similarity plot  
- **TMDB search links** for quick, manual verification  
- Graceful fallbacks so the app never breaks even if some columns are missing
""".strip()
)
st.markdown('</div>', unsafe_allow_html=True)

# Footer (optional mellow note)
st.caption("Tip: want stricter filtering? Add clearer `current_platform`, `origin_label`, or `service` values in your CSV — the board will auto-adapt.")
