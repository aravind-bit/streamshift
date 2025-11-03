# Media Merger Analysis — backup look, plus HBO alias + "Warner Brothers HBO" branding
from __future__ import annotations
import re
from pathlib import Path

import pandas as pd
import streamlit as st

# --- GA4 Measurement Protocol helper ---
import uuid, requests

def send_ga_event(event_name: str, params: dict):
    """Fire-and-forget GA4 event. No-op if secrets not configured."""
    ga = st.secrets.get("ga4", {})
    mid = ga.get("measurement_id")
    sec = ga.get("api_secret")
    if not (mid and sec):
        return  # GA not configured

    payload = {
        "client_id": str(uuid.uuid4()),  # anonymous session id
        "events": [{
            "name": event_name,
            "params": params
        }]
    }
    try:
        requests.post(
            f"https://www.google-analytics.com/mp/collect"
            f"?measurement_id={mid}&api_secret={sec}",
            json=payload,
            timeout=4
        )
    except Exception:
        # never break the app if GA is down
        pass


# Optional: lightweight clustering for other places (kept from your backup;
# we will handle TF-IDF import again inside the map block to be robust)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA
    HAVE_SK = True
except Exception:
    HAVE_SK = False

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# ---------------- embeddings + projection helpers (NEW) ----------------
def _load_embeddings():
    """
    Return (index, meta_df) if FAISS artifacts exist; else (None, None).
    We still re-encode text with Sentence-Transformers to keep logic simple and portable.
    """
    idx_path = DATA / "embeddings.index"
    meta_path = DATA / "embeddings_meta.parquet"
    try:
        if idx_path.exists() and meta_path.exists():
            import faiss  # type: ignore
            meta = pd.read_parquet(meta_path)
            index = faiss.read_index(str(idx_path))
            if index.ntotal == len(meta):
                return index, meta
    except Exception:
        pass
    return None, None

def project_points(X):
    """Project high-dimensional vectors to 2D with UMAP (preferred) or PCA fallback."""
    import numpy as np
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=10, min_dist=0.12)
        pts = reducer.fit_transform(X)
    except Exception:
        # PCA fallback (works for dense arrays)
        try:
            from sklearn.decomposition import PCA  # type: ignore
            pts = PCA(n_components=2, random_state=42).fit_transform(X)
        except Exception:
            # final fallback: make a tiny dummy projection if everything is missing
            n = X.shape[0]
            pts = __import__("numpy").random.RandomState(42).randn(n, 2)

    # Normalize to [0,1] for stable plotting bounds
    x = (pts[:, 0] - pts[:, 0].min()) / (pts[:, 0].ptp() + 1e-9)
    y = (pts[:, 1] - pts[:, 1].min()) / (pts[:, 1].ptp() + 1e-9)
    return x, y

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

# Map brand -> platforms consider "owned" (used for heuristics/originals)
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
/* ===== Global shell ===== */
html, body, .main {
  background: radial-gradient(1200px 600px at 20% -10%, rgba(167,139,250,0.18), rgba(2,6,23,0.0)),
              radial-gradient(900px 480px at 80% -20%, rgba(59,130,246,0.18), rgba(2,6,23,0.0)),
              #0b1120;
  color: #E7E9EE;
}
.main .block-container { max-width: 1280px; padding-top: .6rem; }

/* ===== HARD HIDE ANY CHIP/BADGE/PILL VARIANTS (stronger than before) ===== */
[data-baseweb="tag"],
[data-baseweb="badge"] { display: none !important; }

[data-testid="stBadge"],
.st-badge, .stBadge, .st-badge-container { display: none !important; }

/* Any element whose class suggests a pill/chip */
[class*="pill"], [class*="Pill"], [class*="chip"], [class*="Chip"] { display: none !important; }

/* Stray helper spans that render like pills */
div[role="note"], div[role="status"] span, .st-emotion-cache-badge { display: none !important; }

/* If a horizontal block injected BaseWeb tags, kill those rows */
section [data-testid="stHorizontalBlock"] > div > div:has([data-baseweb="tag"]),
section [data-testid="stHorizontalBlock"] > div > [data-baseweb="tag"] { display:none !important; }

/* ===== Typography tweaks ===== */
.section-blurb { font-size: 1.02rem !important; line-height: 1.45rem !important; color:#d9dce3 !important; }
label, .stSelectbox label { font-size: 1.06rem !important; font-weight: 800 !important; color:#e7e9ee !important; }

/* ===== Hero ===== */
.hero { text-align:center; margin: 0 0 .8rem 0; }
.hero h1 { font-size: 2.6rem; font-weight: 800; color:#C7A6FF; letter-spacing:.2px; margin:0; }
.hero p  { color:#A1A8B3; margin:.4rem 0 0 0; font-size:1.06rem; }

/* ===== Toolbar ===== */
.toolbar {
  background: rgba(17,24,39,0.55);
  border: 1px solid rgba(148,163,184,0.18);
  border-radius: 14px;
  padding: 12px 14px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.25);
  margin-bottom: 10px;
}
.toolbar label { font-size: 1.06rem; color:#E7E9EE; font-weight: 800; }

/* ===== Section shells ===== */
.section-card {
  background: rgba(17,24,39,0.55);
  border: 1px solid rgba(148,163,184,0.18);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}
.section-title { font-size: 1.18rem; font-weight: 900; color:#E7E9EE; margin: 0 0 .35rem 0; }

/* ===== Louder blurbs ===== */
.section-blurb {
  color:#D1D5DB;
  font-size: .98rem;
  line-height: 1.35rem;
  margin: 2px 0 10px 0;
}

/* ===== Dataframes ===== */
[data-testid="stDataFrame"] {
  border: 1px solid rgba(148,163,184,0.18) !important;
  border-radius: 12px !important;
}

/* ===== Inputs ===== */
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
    "<p>Not a ‘where to stream now’ tool — this is a quick <b>after-the-deal</b> view: "
    "who keeps what, and why. Transparent, source-linked.</p></div>",
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
        "Turning titles into vectors (fancy math), squash to 2D, and color by cluster. "
        "<b>Closer dots → similar audience DNA.</b> Use it like a cross-sell radar: "
        "‘fans of this might follow that’."
        "</div>",
        unsafe_allow_html=True,
    )

    # --------- NEW MAP ENGINE: Embeddings → TF-IDF fallback ----------
    try:
        import numpy as np
        # corpus: title + tags
        if "genre_tags" not in fr.columns:
            fr["genre_tags"] = ""
        corpus_full = (fr["title"].astype(str).str.strip() + " " + fr["genre_tags"].astype(str).str.strip()).str.strip()
        mask = corpus_full.str.len() > 0
        df_map = fr.loc[mask].reset_index(drop=True).copy()

        use_embeddings = False
        X = None

        # 1) Try embeddings via Sentence-Transformers (preferable locally)
        index, meta = _load_embeddings()
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            vecs = model.encode(
                df_map["title"].astype(str).str.cat(df_map["genre_tags"].astype(str), sep=" ").tolist(),
                batch_size=64, show_progress_bar=False, normalize_embeddings=True
            ).astype("float32")
            X = vecs
            use_embeddings = True
        except Exception:
            # 2) Fallback to TF-IDF if ST missing (e.g., Streamlit Cloud)
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
                tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=1)
                X = tfidf.fit_transform(corpus_full.loc[mask].tolist()).toarray().astype("float32")
                use_embeddings = False
            except Exception as e:
                st.caption(f"Map unavailable (no embeddings, no scikit-learn): {e}")
                X = None

        engine_label = "Embeddings • FAISS/ST" if use_embeddings else "Classic • TF-IDF"
        st.caption(f"Similarity engine: {engine_label}")

        if X is not None and len(df_map) >= 3:
            mx, my = project_points(X)
            dfp = pd.DataFrame({
                "x": mx,
                "y": my,
                "brand": df_map.apply(
                    lambda r: (r.get("original_brand")
                               or infer_brand_text(r.get("current_platform"))
                               or "Other"),
                    axis=1,
                ),
            })
            import plotly.express as px
            fig = px.scatter(dfp, x="x", y="y", color="brand", height=360)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                legend_title_text="",
            )
            st.plotly_chart(fig, use_container_width=True)
        elif X is None:
            pass  # already messaged
        else:
            st.caption("Add more rows to see the map.")
    except Exception as e:
        st.caption(f"Map unavailable: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

with R:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Rippleboard: The Future of Content</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-blurb'>"
        "<b>What this is:</b> a post-deal TV guide for suits (but in plain English). "
        "Each title gets a status (stay / licensed / exclusive) and a 1-liner. "
        "Tweak the rule and watch the board shift."
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
        "First-party stuff that defines the brand. If this table is loud, "
        "the brand will fight to keep these <b>home</b>."
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
        "Predictions need receipts. official pages; if missing, a TMDB search link. "
        "<i>Not gospel — just a transparent starting point.</i>"
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
st.markdown(
    "<div class='section-blurb'>"
    "Tiny news pulse for your selected brands. It’s vibes, not valuation — "
    "skim the bars, click through to read, decide for yourself."
    "</div>",
    unsafe_allow_html=True,
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

# --- Tech Stack card (spicy but honest)
with st.expander("Tech stack (what’s under the hood) • click to peek", expanded=True):
    content = None
    try:
        content = Path("docs/tech_stack_card.md").read_text(encoding="utf-8")
    except Exception:
        content = (
            "**Under the hood (aka the fun parts):**  \n"
            "Streamlit • Pandas • Plotly • **FAISS Vector Search** • **Sentence-Transformers embeddings** • "
            "Zero-Shot labeling • Knowledge Graph • Declarative Rule Engine • Tiny LLM explanations\n\n"
            "*Translation:* a small **agentic** pipeline that turns messy rights + headlines into explainable "
            "“where it lands” calls."
        )
    st.markdown(content)

# --- Quick feedback (lightweight, local only)

st.markdown("### Is this useful?")
c1, c2, c3 = st.columns(3)
with c1:
    f1 = st.checkbox("Post more scenarios like this", value=False)
with c2:
    f2 = st.checkbox("Cool idea, needs better data", value=False)
with c3:
    f3 = st.checkbox("I’m here for the pretty dots", value=False)
if any([f1, f2, f3]):
    st.success("Thanks for the signal — noted!")

    # Send one GA4 event with useful context
    send_ga_event("feedback_check", {
        "buyer": buyer,
        "target": target,
        "more_scenarios": int(bool(f1)),
        "better_data":   int(bool(f2)),
        "pretty_dots":   int(bool(f3)),
    })

st.markdown("""
<style>
/* Late kill-switch for any stragglers created after first render */
[data-baseweb="tag"],
[data-baseweb="badge"],
[data-testid="stBadge"],
.st-badge, .stBadge, .st-badge-container,
[class*="pill"], [class*="Pill"], [class*="chip"], [class*="Chip"],
div[role="note"], div[role="status"] span, .st-emotion-cache-badge {
  display: none !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- footer ----------------
st.caption(
    "Hobby demo for media M&A what-ifs. Data: TMDB where available; status/notes are testing for illustration."
)
