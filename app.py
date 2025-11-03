# Media Merger Analysis — backup look, plus HBO alias + "Warner Brothers HBO" branding
from __future__ import annotations
import re
from pathlib import Path

import pandas as pd
import streamlit as st

# Optional: lightweight clustering for the map (skips gracefully if missing)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA
    HAVE_SK = True
except Exception:
    HAVE_SK = False

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# ---------------- helpers ----------------
def read_csv_safe(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252", "mac_roman"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

def ensure_cols(df: pd.DataFrame, template: dict) -> pd.DataFrame:
    for k, v in template.items():
        if k not in df.columns:
            df[k] = v
        df[k] = df[k].fillna("")
    return df

# Display aliases (UI only)
PLATFORM_ALIASES = {
    "HBO Max": "HBO",
    "Max": "HBO",
}
def apply_platform_aliases(text: str) -> str:
    if not isinstance(text, str):
        return text
    out = text
    for k, v in PLATFORM_ALIASES.items():
        out = re.sub(rf"\b{k}\b", v, out)
    return out

# The order of default brands to show in the selectors
DEFAULT_BRANDS = [
    "Amazon",
    "Warner Brothers HBO",  # formerly shown as "Max/HBO Max"
    "Paramount Global",
    "Comcast (NBCUniversal)",
    "Disney",
    "Netflix",
    "Apple",
    "Sony",
    "Hulu",
]

# Map brand -> platforms we consider "owned" (used for heuristics/originals)
STREAMERS_BY_BRAND = {
    "Amazon": ["Prime Video", "Amazon Prime Video"],
    "Warner Brothers HBO": ["HBO", "HBO Max", "Max"],
    "Paramount Global": ["Paramount+", "Paramount Plus"],
    "Comcast (NBCUniversal)": ["Peacock"],
    "Disney": ["Disney+"],
    "Netflix": ["Netflix"],
    "Apple": ["Apple TV+"],
    "Sony": ["Crunchyroll", "SonyLIV"],
    "Hulu": ["Hulu"],
}

def infer_brand_text(txt: str) -> str:
    """Infer a brand from text such as current_platform/original_network strings."""
    t = f" {str(txt or '').lower()} "
    if " netflix " in t: return "Netflix"
    # collapse any HBO/Max variants to the single brand label
    if " hbo " in t or " hbomax " in t or " hbo max " in t or " max " in t:
        return "Warner Brothers HBO"
    if " prime video " in t or " amazon " in t:
        return "Amazon"
    if " paramount+ " in t or " paramount plus " in t or " cbs " in t:
        return "Paramount Global"
    if " peacock " in t or " nbc " in t or " universal television " in t:
        return "Comcast (NBCUniversal)"
    if " apple tv+ " in t:
        return "Apple"
    if " disney+ " in t:
        return "Disney"
    if " hulu " in t:
        return "Hulu"
    return ""

def buyer_target_options(fr: pd.DataFrame) -> list[str]:
    """Combine our default brand list with any discovered `original_brand` values."""
    discovered = sorted(set(x for x in fr.get("original_brand", []).tolist() if str(x).strip()))
    out: list[str] = []
    for b in DEFAULT_BRANDS + discovered:
        if b and b not in out:
            out.append(b)
    return out or DEFAULT_BRANDS

def tmdb_links(title: str, source_urls: str) -> list[str]:
    """Prefer explicit URLs in CSV; otherwise provide a TMDB search to verify quickly."""
    urls = [u.strip() for u in str(source_urls or "").split(";") if u.strip()]
    if urls:
        return urls
    if title:
        q = re.sub(r"\s+", "+", title.strip())
        return [f"https://www.themoviedb.org/search?query={q}"]
    return []

def synth_note(status: str, buyer: str, target: str) -> str:
    s = (status or "").lower()
    if "exclusive" in s: return f"High-value IP → likely exclusive under {buyer}."
    if "window" in s: return f"Window under {buyer}; long-tail remains with {target}."
    if "shared" in s: return "Lower strategic value → shared licensing continues."
    return "Not a flagship—probably stays put for now."

def compact_table(df: pd.DataFrame, keep_order: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for c in df.columns:
        df[c] = df[c].replace("", pd.NA)
    usable = [c for c in df.columns if df[c].notna().any()]
    df = df[usable]
    ordered = [c for c in keep_order if c in df.columns] + [c for c in df.columns if c not in keep_order]
    return df[ordered].fillna("—")

# ---------------- load data ----------------
st.set_page_config(page_title="Media Merger Analysis", layout="wide")

fr = read_csv_safe(DATA / "franchises.csv")
if fr.empty:
    st.error("Could not read data/franchises.csv. Add your data and rerun.")
    st.stop()

fr = ensure_cols(
    fr,
    {
        "title": "",
        "origin_label": "",          # your status label/rule (e.g., Exclusive/Stay/Windowed)
        "predicted_policy": "",      # free-text rationale if you saved one
        "original_flag": "",         # Y/N
        "original_brand": "",
        "original_network": "",
        "producer_list": "",
        "current_platform": "",
        "source_urls": "",
        "genre_tags": "",
    },
)

# ---------------- CSS (visual parity with your backup) ----------------
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

# ---------------- Hero ----------------
st.markdown(
    "<div class='hero'><h1>Media Merger Analysis</h1>"
    "<p>Visualize what happens to movies & series after hypothetical media mergers. "
    "Who streams what after the deal.</p></div>",
    unsafe_allow_html=True,
)

# ---------------- Toolbar: Buyer / Target ----------------
st.markdown("<div class='toolbar'>", unsafe_allow_html=True)
c1, c2 = st.columns([1, 1])
opts = buyer_target_options(fr)
with c1:
    buyer = st.selectbox("Buyer", opts, index=opts.index("Amazon") if "Amazon" in opts else 0)
with c2:
    target_default = "Warner Brothers HBO" if "Warner Brothers HBO" in opts else (opts[1] if len(opts) > 1 else opts[0])
    target = st.selectbox("Target", opts, index=opts.index(target_default))
st.markdown("</div>", unsafe_allow_html=True)

# keep labels for optional sentiment section
st.session_state["buyer_label"] = buyer
st.session_state["target_label"] = target

# ---------------- Main: map + rippleboard ----------------
L, R = st.columns([1.05, 1.6])

with L:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>✣ IP Similarity Map</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-blurb'>"
        "<b>What this shows:</b> we turn title/genre text into vectors (TF-IDF) and plot with PCA. "
        "Closer dots ≈ similar audience DNA or adjacent viewing clusters. "
        "<b>How to read:</b> look for groups near the buyer’s core slate to spot fast-track cross-sell."
        "</div>",
        unsafe_allow_html=True,
    )
    try:
        if HAVE_SK and len(fr) >= 3:
            corpus = (fr["title"].astype(str) + " " + fr["genre_tags"].astype(str)).tolist()
            X = TfidfVectorizer(min_df=1, max_features=1000).fit_transform(corpus)
            coords = PCA(n_components=2, random_state=42).fit_transform(X.toarray())
            dfp = pd.DataFrame(
                {
                    "x": coords[:, 0],
                    "y": coords[:, 1],
                    "brand": fr.apply(
                        lambda r: (r.get("original_brand")
                                   or infer_brand_text(r.get("current_platform"))
                                   or "Other"),
                        axis=1,
                    ),
                }
            )
            import plotly.express as px

            fig = px.scatter(dfp, x="x", y="y", color="brand", height=360)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                legend_title_text="",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Add more rows or install scikit-learn to see the map.")
    except Exception as e:
        st.caption(f"Map unavailable: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

with R:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Rippleboard: The Future of Content</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-blurb'>"
        "<b>What this shows:</b> where marquee IP likely lands post-deal. "
        "<b>Status</b> comes from your CSV rules; <b>Notes</b> are the saved rationale or an auto-explanation."
        "</div>",
        unsafe_allow_html=True,
    )

    rb = fr.copy()
    # If you use origin_label as your status, reuse it; otherwise it falls back to "Stay"
    rb["Predicted Status"] = rb["origin_label"].replace("", "Stay")

    # Alias visible platform strings to "HBO"
    rb["current_platform"] = rb["current_platform"].astype(str).apply(apply_platform_aliases)

    rb["Notes"] = [
        (str(row.get("predicted_policy", "")).strip()
         or synth_note(str(row.get("Predicted Status", "")), buyer, target))
        for _, row in rb.iterrows()
    ]
    rb_view = (
        rb[["title", "Predicted Status", "Notes", "current_platform"]]
        .rename(columns={"title": "IP / Franchise", "current_platform": "Current Platform"})
    )
    st.dataframe(rb_view.head(20), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Originals (expander) ----------------
with st.expander("Originals from the target"):
    st.markdown(
        "<div class='section-blurb'>"
        "<b>Purpose:</b> quick scan of titles born at the target. "
        "If explicit original tags are missing, we infer from platform/network so it doesn’t look blank."
        "</div>",
        unsafe_allow_html=True,
    )
    # strict originals first
    orig = fr[
        (fr.get("original_flag", "").astype(str).str.upper() == "Y")
        & (fr.get("original_brand", "").astype(str).str.lower() == target.lower())
    ][["title", "original_network", "producer_list", "current_platform", "source_urls"]].copy()

    # if empty, infer by platform/network brand
    if orig.empty:
        mask = fr.apply(
            lambda r: infer_brand_text(r.get("current_platform", "")) == target
            or infer_brand_text(r.get("original_network", "")) == target,
            axis=1,
        )
        orig = fr.loc[
            mask, ["title", "original_network", "producer_list", "current_platform", "source_urls"]
        ].head(20).copy()

    # UI alias for platforms
    orig["current_platform"] = orig["current_platform"].astype(str).apply(apply_platform_aliases)

    orig = compact_table(
        orig, ["title", "original_network", "producer_list", "current_platform", "source_urls"]
    )
    if not orig.empty:
        st.dataframe(
            orig.rename(columns={"current_platform": "platform"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No clear originals visible yet—add a few and rerun enrichment.")

# ---------------- Sources / Traceability ----------------
with st.expander("Sources / Traceability (for titles shown)"):
    st.markdown(
        "<div class='section-blurb'>"
        "<b>Why this exists:</b> so you can answer ‘where did you get that?’ publicly. "
        "We show explicit links when present; otherwise a TMDB search link for quick verification."
        "</div>",
        unsafe_allow_html=True,
    )
    shown = pd.concat(
        [
            rb_view.head(20)[["IP / Franchise"]].rename(columns={"IP / Franchise": "title"}),
            orig[["title"]] if 'orig' in locals() and not orig.empty else pd.DataFrame(columns=["title"]),
        ],
        ignore_index=True,
    ).drop_duplicates()

    if shown.empty:
        st.caption("No titles in view yet.")
    else:
        joined = shown.merge(fr[["title", "source_urls"]], on="title", how="left")
        for _, r in joined.iterrows():
            t = str(r["title"]).strip()
            links = tmdb_links(t, r.get("source_urls", ""))
            st.markdown(f"**{t}**")
            for u in links:
                st.write(f"- {u}")

# ---------- Headline Mood (quick check) ----------
st.markdown("### Headline Mood (quick check)")
st.caption(
    "Tiny NLP pulse based on recent headlines for the selected brands. Not investment advice—just a vibe check."
)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader_ok = True
except Exception:
    _vader_ok = False

@st.cache_data(show_spinner=False)
def load_headlines(path="data/headlines.csv"):
    df = pd.read_csv(path)
    for col in ["brand", "headline", "link", "date"]:
        if col not in df.columns:
            raise ValueError("headlines.csv must have columns: brand, headline, link, date")
    return df

def _score_texts(texts):
    if not _vader_ok:
        return [float("nan")] * len(texts)
    analyzer = SentimentIntensityAnalyzer()
    return [analyzer.polarity_scores(t)["compound"] for t in texts]

try:
    h = load_headlines()
    focus = {buyer, target}
    hh = h[h["brand"].isin(focus)] if len(h) else h

    if _vader_ok and len(hh):
        hh = hh.copy()
        hh["sentiment"] = _score_texts(hh["headline"].fillna(""))

        col1, col2 = st.columns((2, 3))
        with col1:
            s = hh.groupby("brand")["sentiment"].mean().reset_index()
            s["sentiment"] = s["sentiment"].round(3)
            st.dataframe(s, use_container_width=True, hide_index=True)
        with col2:
            chart_df = hh.groupby("brand")["sentiment"].mean().to_frame()
            chart_df.columns = ["Mood"]
            st.bar_chart(chart_df, height=180)

        st.write("**Recent headlines**")
        for _, row in hh.sort_values("date", ascending=False).head(6).iterrows():
            st.write(f"- **{row['brand']}** — [{row['headline']}]({row['link']})  \n  _{row['date']}_")
    else:
        st.info(
            "Optional: add `data/headlines.csv` and install `vaderSentiment` "
            "for a tiny mood check (columns: brand, headline, link, date)."
        )
except FileNotFoundError:
    st.info("Optional: add `data/headlines.csv` for mood check (columns: brand, headline, link, date).")

# ---------------- footer ----------------
st.caption(
    "Hobby demo for media M&A what-ifs. Data: TMDB where available; status/notes are heuristic for illustration."
)
