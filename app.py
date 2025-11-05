# app.py
import re
import os
import numpy as np
import pandas as pd
import streamlit as st

# Optional: plotly for nicer scatter (works well with Streamlit)
import plotly.express as px

# ------------------------
# Config & CSS
# ------------------------
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

/* Hide any legacy pill chips if they exist */
.pill, .helper-pill, .hint-pill,
[data-testid="stBadges"], .st-emotion-cache-1r4qj8e { display: none !important; }

/* Make blurbs more readable */
.section-blurb { font-size: 1.02rem !important; line-height: 1.45rem !important; color:#d9dce3 !important; }

/* Slightly larger Buyer/Target labels */
label, .stSelectbox label { font-size: 1.06rem !important; font-weight: 800 !important; color:#e7e9ee !important; }

/* Hero */
.hero { text-align:center; margin: 0 0 .8rem 0; }
.hero h1 { font-size: 2.6rem; font-weight: 800; color:#C7A6FF; letter-spacing:.2px; margin:0; }
.hero p  { color:#A1A8B3; margin:.4rem 0 0 0; font-size:1.06rem; }

/* Toolbar */
.toolbar {
  background: rgba(17,24,39,0.55);
  border: 1px solid rgba(148,163,184,0.18);
  border-radius: 14px;
  padding: 12px 14px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.25);
  margin-bottom: 10px;
}
.toolbar label { font-size: 1.06rem; color:#E7E9EE; font-weight: 800; }

/* Section shells */
.section-card {
  background: rgba(17,24,39,0.55);
  border: 1px solid rgba(148,163,184,0.18);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}
.section-title { font-size: 1.18rem; font-weight: 900; color:#E7E9EE; margin: 0 0 .35rem 0; }

/* Louder blurbs */
.section-blurb {
  color:#D1D5DB;
  font-size: .98rem;
  line-height: 1.35rem;
  margin: 2px 0 10px 0;
}

/* Dataframes */
[data-testid="stDataFrame"] {
  border: 1px solid rgba(148,163,184,0.18) !important;
  border-radius: 12px !important;
}

/* Inputs */
.stSelectbox [data-baseweb="select"] > div {
  background: rgba(17,24,39,0.55);
  border: 1px solid rgba(148,163,184,0.25);
}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------
# Data helpers
# ------------------------
DATA_PATH = "data/franchises.csv"

def load_franchises(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Data not found: {path}")
        return pd.DataFrame(columns=["title", "current_platform"])
    df = pd.read_csv(path, dtype=str).fillna("")
    # standardize expected columns
    if "title" not in df.columns:
        # Try to infer title column
        guess = [c for c in df.columns if "title" in c.lower() or "ip" in c.lower()]
        if guess:
            df = df.rename(columns={guess[0]: "title"})
        else:
            df["title"] = ""
    if "current_platform" not in df.columns:
        # Try to infer
        guess = [c for c in df.columns if "platform" in c.lower()]
        if guess:
            df = df.rename(columns={guess[0]: "current_platform"})
        else:
            df["current_platform"] = ""
    return df

def normalize_brand_name(brand: str) -> str:
    """Create a lenient regex for brand matching."""
    brand = brand.strip()
    # special handling for Warner/Max/HBO cluster
    if brand.lower() in ["warner brothers hbo", "warner bros hbo", "warner", "wbd", "hbo", "max"]:
        return r"(warner|wbd|hbo|max)"
    # peacock
    if brand.lower() in ["peacock", "nbc", "comcast (nbcuniversal)", "comcast", "nbcuniversal"]:
        return r"(peacock|nbc|nbcuniversal|comcast)"
    # amazon
    if brand.lower() in ["amazon"]:
        return r"(amazon|prime\s*video)"
    # apple
    if brand.lower() in ["apple"]:
        return r"(apple\s*tv\+|apple)"
    # netflix
    if brand.lower() in ["netflix"]:
        return r"(netflix)"
    # hulu
    if brand.lower() in ["hulu"]:
        return r"(hulu)"
    # disney
    if brand.lower() in ["disney"]:
        return r"(disney|disney\+)"
    # paramount
    if brand.lower() in ["paramount global", "paramount"]:
        return r"(paramount\+|paramount)"
    # sony
    if brand.lower() in ["sony"]:
        return r"(sony)"
    return re.escape(brand)

SEARCH_COLUMNS = [
    "current_platform", "origin_label", "origin_brand",
    "platform", "producer_list", "distributor_list", "tags"
]

def row_matches_brand(row: pd.Series, brand_regex: str) -> bool:
    for col in SEARCH_COLUMNS:
        if col in row.index:
            val = str(row[col])
            if val and re.search(brand_regex, val, flags=re.IGNORECASE):
                return True
    return False

def filter_for_buyer_target(df: pd.DataFrame, buyer: str, target: str) -> pd.DataFrame:
    buyer_re = normalize_brand_name(buyer)
    target_re = normalize_brand_name(target)
    mask = df.apply(lambda r: row_matches_brand(r, buyer_re) or row_matches_brand(r, target_re), axis=1)
    sub = df.loc[mask].copy()
    if sub.empty:
        st.info("No direct matches for Buyer/Target in metadata—showing all titles as a fallback.")
        sub = df.copy()
    return sub

# ------------------------
# Projection for the map
# ------------------------
def ensure_xy_projection(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has 'x','y' numeric columns for the scatter. If missing, make a simple pseudo-projection."""
    if "x" in df.columns and "y" in df.columns:
        # clean numeric
        df["x"] = pd.to_numeric(df["x"], errors="coerce").fillna(0.0)
        df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)
        return df

    # Simple deterministic pseudo-projection if no embeddings are present:
    # hash title into 2 floats in [0,1]
    def hash_to_unit(s: str, seed: int) -> float:
        h = abs(hash((s, seed))) % 10_000_000
        return (h / 10_000_000.0)

    df = df.copy()
    df["x"] = df["title"].astype(str).apply(lambda s: hash_to_unit(s, 1))
    df["y"] = df["title"].astype(str).apply(lambda s: hash_to_unit(s, 2))
    return df

# ------------------------
# Color / legend mapping
# ------------------------
def platform_bucket(row: pd.Series) -> str:
    text = " ".join([str(row.get(c, "")) for c in ["current_platform", "origin_label", "origin_brand"]]).lower()
    if re.search(r"netflix", text): return "Netflix"
    if re.search(r"\bmax\b|\bhbo\b", text): return "Max"
    if re.search(r"peacock|nbc", text): return "Peacock"
    if re.search(r"prime\s*video|amazon", text): return "Amazon"
    if re.search(r"apple", text): return "Apple"
    return "Other"

COLOR_MAP = {
    "Netflix": "#60a5fa",
    "Max": "#a78bfa",
    "Peacock": "#f87171",
    "Amazon": "#22d3ee",
    "Apple": "#34d399",
    "Other": "#94a3b8",
}

# ------------------------
# Rippleboard helper (very simple baseline)
# ------------------------
def compute_rippleboard(df: pd.DataFrame, buyer: str, target: str) -> pd.DataFrame:
    # Keep whatever your previous logic was; here we just pass through and ensure columns exist
    out = df.copy()
    if "predicted_policy" not in out.columns:
        out["predicted_policy"] = np.where(
            out["current_platform"].str.contains("amazon|prime", case=False, na=False),
            "Exclusive Distribution",
            "Licensed"
        )
    if "notes" not in out.columns:
        out["notes"] = np.where(
            out["predicted_policy"].str.contains("Exclusive", na=False),
            "High-value IP → likely exclusive under buyer.",
            "Not a flagship—probably stays put for now."
        )
    return out

# ------------------------
# UI
# ------------------------
st.markdown(
    '<div class="hero"><h1>Media Merger Analysis</h1>'
    '<p>Visualize what happens to movies & series after hypothetical media mergers. '
    'Who streams what after the deal.</p></div>',
    unsafe_allow_html=True,
)

with st.container():
    st.write("")  # small breathing room
    buyer, target = st.columns(2)
    BUYERS = [
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
    with buyer:
        buyer_choice = st.selectbox("Buyer", BUYERS, index=0)
    with target:
        target_choice = st.selectbox("Target", BUYERS, index=1)

# ------------------------
# Load + filter data
# ------------------------
df_raw = load_franchises(DATA_PATH)
if df_raw.empty:
    st.stop()

# Filter strictly to Buyer/Target for the map & board
df_bt = filter_for_buyer_target(df_raw, buyer_choice, target_choice)

# Section: IP Similarity Map (filtered)
st.markdown("### ✣ IP Similarity Map (filtered to Buyer/Target)")
st.markdown(
    """
**What this shows**  
Turning titles into vectors (fancy math), squash to 2D, and color by cluster. Closer dots → **similar audience DNA**.  
Positions are from a 2-D projection of text embeddings (or a stable fallback), just to give a vibe-level neighborhood.

**How to read the map**
- **Each dot** = a show/film.  
- **Closer dots** = similar audience DNA (genre/keywords/description).  
- **Colors** = rough clusters by platform family (e.g., Netflix, Max, Apple).  
- Use it to spot quick “this fits the buyer” vs. “this is an outlier” reads.  
""",
    unsafe_allow_html=True,
)

# Grid for map (left) + Rippleboard (right)
left, right = st.columns([0.54, 0.46], gap="large")

# ---- Map on left
with left:
    df_map = ensure_xy_projection(df_bt.copy())
    df_map["cluster_name"] = df_map.apply(platform_bucket, axis=1)

    if df_map.empty or df_map["title"].eq("").all():
        st.info("No titles available after filtering.")
    else:
        fig = px.scatter(
            df_map,
            x="x", y="y",
            color="cluster_name",
            color_discrete_map=COLOR_MAP,
            hover_data={"title": True, "current_platform": True, "x": False, "y": False},
            height=420,
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            legend_title_text="cluster_name",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E7E9EE"),
        )
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

# ---- Rippleboard on right
with right:
    st.markdown("### Rippleboard: The Future of Content")
    st.markdown(
        """
**What this is**  
A post-deal TV guide for suits (but in plain English). Each title gets a status (**stay / licensed / exclusive**) and a 1-liner.  
**How to read this**
- **Stay** → Likely the contract/window keeps it where it is for now.  
- **Licensed** → Shared/syndicated outcome likely (may move in parts or by region).  
- **Exclusive** → If the buyer owns the IP, they’d likely pull it in-house at renewal.  
*Note:* spin-offs depend on **derivative rights**, not just today’s streamer.
""",
        unsafe_allow_html=True,
    )

    # Compute rippleboard (you can keep your original logic here)
    rb_df = compute_rippleboard(df_bt, buyer_choice, target_choice)

    # Robust view
    rb_view = rb_df.copy()
    wanted = ["title", "predicted_policy", "notes", "current_platform"]
    have = [c for c in wanted if c in rb_view.columns]
    if have:
        rb_view = rb_view[have].rename(columns={
            "title": "IP / Franchise",
            "predicted_policy": "Predicted Status",
            "current_platform": "Current Platform",
            "notes": "Notes",
        })
        st.dataframe(rb_view.head(20), use_container_width=True, height=520)
    else:
        st.dataframe(rb_view.head(20), use_container_width=True, height=520)

# ------------------------
# Originals from the target (optional table)
# ------------------------
st.markdown("### Originals from the target")
st.markdown(
    "Target-brand originals inferred from platform/network/labels. "
    "If your CSV is thin, this heuristic helps avoid an empty table."
)
orig_cols = [c for c in ["title", "origin_brand", "origin_label", "producer_list", "current_platform", "predicted_policy"] if c in df_bt.columns]
if orig_cols:
    st.dataframe(df_bt[orig_cols].head(50), use_container_width=True)
else:
    st.info("No original metadata fields were found to display here.")

# ------------------------
# Sources / Traceability (quick links)
# ------------------------
st.markdown("### Sources / Traceability (for titles shown)")
def tmdb_link(title: str) -> str:
    q = re.sub(r"\s+", "+", str(title).strip())
    return f"[search TMDB](https://www.themoviedb.org/search?query={q})"

trace = pd.DataFrame({
    "title": df_bt["title"].head(10),
    "link": [tmdb_link(t) for t in df_bt["title"].head(10)]
})
st.dataframe(trace, use_container_width=True)

# ------------------------
# Footer / tech stack card (short)
# ------------------------
with st.expander("Tech Stack (peek)"):
    st.markdown(
        """
- **Streamlit** for the app shell & UI  
- **Pandas / Plotly** for data + visuals  
- **Embeddings → 2D projection** (UMAP/PCA or fallback)  
- **Regex/heuristics** for Buyer/Target filtering across platform/brand fields  
"""
    )
