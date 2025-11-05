# app.py — StreamShift (restored sections + filtered map + cleaned layout)

import os
import math
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# -----------------------------
# Config & styling
# -----------------------------
st.set_page_config(page_title="Media Merger Analysis", page_icon="🎬", layout="wide")

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

/* Hide any stray “pills” */
.pill, .helper-pill, .hint-pill, [data-testid="stBadges"], .st-emotion-cache-1r4qj8e { display:none !important; }

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
  margin-bottom: 16px;
}
.section-title { font-size: 1.18rem; font-weight: 900; color:#E7E9EE; margin: 0 0 .35rem 0; }

/* Louder blurbs */
.section-blurb { color:#D1D5DB; font-size: .98rem; line-height: 1.35rem; margin: 2px 0 10px 0; }

/* Dataframes */
[data-testid="stDataFrame"] {
  border: 1px solid rgba(148,163,184,0.18) !important;
  border-radius: 12px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Helper stuff
# -----------------------------
BUYER_OPTIONS = [
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

PLATFORM_KEYWORDS = {
    "Amazon": ["prime", "amazon"],
    "Warner Brothers HBO": ["hbo", "max"],
    "Paramount Global": ["paramount"],
    "Comcast (NBCUniversal)": ["peacock", "nbc"],
    "Disney": ["disney", "hulu", "hotstar"],
    "Netflix": ["netflix"],
    "Apple": ["apple"],
    "Sony": ["sony", "starz", "crackle"],
    "Hulu": ["hulu"],
    "Max": ["max", "hbo"],
    "Peacock": ["peacock"]
}

def load_df(path="data/franchises.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize helpers
    for col in ["title","current_platform","origin_label","predicted_policy","source_urls","cluster_name"]:
        if col not in df.columns: df[col] = ""
    if "x" not in df.columns or "y" not in df.columns:
        # fallback to grid in case no embedding projection present
        xs = np.linspace(0.1, 0.9, len(df))
        ys = np.sin(np.linspace(0, 3.14, len(df))) * 0.3 + 0.65
        df["x"], df["y"] = xs, ys
    return df

def match_brand_rows(df: pd.DataFrame, brand: str) -> pd.Series:
    """Rows where current_platform mentions the brand family keywords."""
    if brand not in PLATFORM_KEYWORDS: 
        return pd.Series(False, index=df.index)
    hay = df["current_platform"].fillna("").str.lower()
    mask = False
    for kw in PLATFORM_KEYWORDS[brand]:
        mask = mask | hay.str.contains(kw, regex=False)
    return mask

def subset_for_buyer_target(df: pd.DataFrame, buyer: str, target: str) -> pd.DataFrame:
    m_b = match_brand_rows(df, buyer)
    m_t = match_brand_rows(df, target)
    return df[m_b | m_t].copy()

def nice_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    return df[keep].copy()

def tmdb_search_link(title: str) -> str:
    q = title.replace(" ", "+")
    return f"[search TMDB](https://www.themoviedb.org/search?query={q})"

# -----------------------------
# Hero + selectors
# -----------------------------
st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown("<h1>Media Merger Analysis</h1>", unsafe_allow_html=True)
st.markdown(
    "<p>Visualize what happens to movies & series after hypothetical media mergers. "
    "Who streams what after the deal.</p>",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

buyer, target = st.columns(2)
with buyer:
    buyer_val = st.selectbox("Buyer", BUYER_OPTIONS, index=BUYER_OPTIONS.index("Amazon"))
with target:
    target_val = st.selectbox("Target", BUYER_OPTIONS, index=BUYER_OPTIONS.index("Warner Brothers HBO"))

df_all = load_df()
df_bt = subset_for_buyer_target(df_all, buyer_val, target_val)

# -----------------------------
# Row: Map (left) • Rippleboard (right)
# -----------------------------
left, right = st.columns([0.95, 1.05], gap="large")

with left:
    st.markdown("### ✣ IP Similarity Map (filtered to Buyer/Target)")
    st.markdown(
        """
<div class="section-card">
  <div class="section-blurb">
    <b>What this shows</b><br/>
    Turning titles into vectors (fancy math), squash to 2D, and color by cluster. Closer dots → <b>similar audience DNA</b>.
    Positions are from a 2-D projection of text embeddings (or a stable fallback) — just to give a vibe-level neighborhood.
    <br/><br/>
    <b>How to read the map</b>
    <ul>
      <li><b>Each dot</b> = a show/film.</li>
      <li><b>Closer dots</b> = more similar audience DNA (genre/keywords/description).</li>
      <li><b>Colors</b> = rough clusters by platform family (e.g., Netflix, Max, Apple).</li>
      <li>Use it to spot quick “this fits the buyer” vs. “this is an outlier” reads.</li>
    </ul>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Filter map points (only buyer/target)
    if len(df_bt):
        # Color by rough family for legibility
        fam = []
        for _, row in df_bt.iterrows():
            plat = (row["current_platform"] or "").lower()
            if "netflix" in plat: fam.append("Netflix")
            elif "max" in plat or "hbo" in plat: fam.append("Max")
            elif "peacock" in plat: fam.append("Peacock")
            elif "amazon" in plat or "prime" in plat: fam.append("Amazon")
            elif "apple" in plat: fam.append("Apple")
            else: fam.append("Other")
        df_bt = df_bt.assign(family=fam)

        fig = px.scatter(
            df_bt,
            x="x", y="y",
            color="family",
            hover_name="title",
            hover_data={"x": False, "y": False, "family": True, "current_platform": True},
            color_discrete_sequence=px.colors.qualitative.Set2,
            height=420
        )
        fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), legend_title_text="cluster_name")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No titles matched Buyer/Target for the map.")

with right:
    st.markdown("### Rippleboard: The Future of Content")
    st.markdown(
        """
<div class="section-card">
  <div class="section-blurb">
    <b>What this is</b><br/>
    A post-deal TV guide for suits (but in plain English). Each title gets a status (<b>stay</b> / <b>licensed</b> / <b>exclusive</b>) and a 1-liner.
    <br/><br/>
    <b>How to read this</b>
    <ul>
      <li><b>Stay</b> → Likely the contract/window keeps it where it is for now.</li>
      <li><b>Licensed</b> → Shared/syndicated outcome likely (may move in parts or by region).</li>
      <li><b>Exclusive</b> → If the buyer owns the IP, they’d likely pull it in-house at renewal.</li>
      <li><i>Note:</i> spin-offs depend on <b>derivative rights</b>, not just today’s streamer.</li>
    </ul>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    rb_view = nice_cols(
        df_bt,
        ["title","predicted_policy","notes","current_platform","origin_label"]
        if "notes" in df_all.columns
        else ["title","predicted_policy","current_platform","origin_label"]
    ).rename(columns={
        "title":"IP / Franchise",
        "predicted_policy":"Predicted Status",
        "current_platform":"Current Platform",
        "origin_label":"Origin"
    })

    if len(rb_view):
        st.dataframe(rb_view, use_container_width=True, height=460, hide_index=True)
    else:
        st.info("No titles matched Buyer/Target for the Rippleboard.")

# -----------------------------
# Originals from the target
# -----------------------------
with st.expander("Originals from the target", expanded=False):
    ot_mask = (df_bt["origin_label"].fillna("").str.lower().str.contains(
        "original|hbo original|netflix original|apple|amazon|peacock|paramount"
    ))
    originals_view = nice_cols(
        df_bt[ot_mask].copy(),
        ["title","origin_label","producer_list","current_platform","predicted_policy"]
    ).rename(columns={"current_platform":"platform"})
    if len(originals_view):
        st.dataframe(originals_view, use_container_width=True, hide_index=True)
    else:
        st.caption("No clear originals detected with current heuristics.")

# -----------------------------
# Sources / Traceability
# -----------------------------
with st.expander("Sources / Traceability (for titles shown)", expanded=False):
    src = df_bt[["title","source_urls"]].copy()
    if len(src):
        # show TMDB search link if blank
        rows = []
        for _, r in src.iterrows():
            link = (str(r["source_urls"]).strip() or tmdb_search_link(r["title"]))
            rows.append({"title": r["title"], "link": link})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Saved source URLs shown where available; otherwise a quick TMDB search link to verify manually.")
    else:
        st.caption("No titles to show.")

# -----------------------------
# Headline Mood (quick check)
# -----------------------------
with st.container():
    st.markdown("### Headline Mood (quick check)")
    st.caption(
        "Tiny news pulse for your selected brands. It’s vibes, not valuation — skim the bars, click through to read, decide for yourself."
    )

    # Minimal hardcoded examples (replace with your news pipeline)
    NEWS = {
        "Amazon": [
            ("2025-10-24", "Amazon", "Amazon Ads adds programmatic access to NBA League Pass (season launch)", "https://www.aboutamazon.com/news/"),
            ("2025-10-13", "Amazon", "Prime set for debut as NBA partner — how the deal got done", "https://www.primevideo.com/"),
        ],
        "Warner Brothers HBO": [
            ("2025-07-08", "Warner Brothers HBO", "HBO Max rebrand goes live July 9 (trade coverage)", "https://www.wbd.com/"),
            ("2025-05-14", "Warner Brothers HBO", "WBD: Max will become HBO Max this summer (official)", "https://www.wbd.com/"),
        ],
    }

    def toy_sent(text: str) -> float:
        # super tiny toy “sentiment”: + if words like 'adds', 'partner', 'rebrand'
        t = text.lower()
        score = 0
        for w in ["adds","launch","partner","rebrand","expand","record","beats","profit","renew"]:
            if w in t: score += 1
        for w in ["layoff","delay","strike","lawsuit","downturn","loss","drop"]:
            if w in t: score -= 1
        return score / 10.0

    brands = [buyer_val, target_val]
    s_rows = []
    for b in brands:
        items = NEWS.get(b, [])
        s = np.mean([toy_sent(x[2]) for x in items]) if items else 0.0
        s_rows.append({"brand": b, "sentiment": round(float(s), 3)})
    s_df = pd.DataFrame(s_rows)

    c1, c2 = st.columns([0.42, 0.58])
    with c1:
        st.dataframe(s_df, use_container_width=True, hide_index=True)
    with c2:
        fig_s = px.bar(
            s_df, x="brand", y="sentiment", text="sentiment",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig_s.update_layout(margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("#### Recent headlines")
    for b in brands:
        for (d, brand, title, url) in NEWS.get(b, []):
            st.markdown(f"- **{brand}** — [{title}]({url})  \n  <span style='color:#9aa3af'>{d}</span>", unsafe_allow_html=True)

# -----------------------------
# Tech stack card
# -----------------------------
with st.expander("Tech stack (what’s under the hood) • click to peek", expanded=False):
    st.markdown(
        """
**Under the hood (aka the fun parts):**  
Streamlit · Pandas · Plotly · **FAISS Vector Search** · **Sentence-Transformers Embeddings** · Zero-Shot Labeling ·
Knowledge Graph · Declarative Rule Engine · Tiny LLM Explanations  

_Translation:_ a small **agentic** pipeline that turns messy rights + headlines into explainable “where it lands” calls.
""")

# -----------------------------
# Feedback row
# -----------------------------
st.markdown("## Is this useful?")
f1, f2, f3 = st.columns(3)
with f1:
    st.checkbox("Post more scenarios like this")
with f2:
    st.checkbox("Cool idea, needs better data")
with f3:
    st.checkbox("I’m here for the pretty dots")

st.caption("Hobby demo for media M&A what-ifs. Data: TMDB where available; status/notes are testing for illustration.")
