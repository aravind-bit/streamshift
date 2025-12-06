# Media Merger Analysis — backup look, plus HBO alias + "Warner Brothers HBO" branding
from __future__ import annotations
import re
from pathlib import Path
from openai import OpenAI
import json
#client = OpenAI()
# --- Optional OpenAI client (for AI Deal Analyst) ---
try:
    from openai import OpenAI
    _openai_ok = True
    client = OpenAI()
except Exception:
    _openai_ok = False
    client = None


import numpy as np
import pandas as pd
import streamlit as st
from logging_utils import log_usage, LOG_PATH
LAST_UPDATED = "2025-12-06"  # date of last refresh of deal details

import base64

import uuid
import pandas as pd
from datetime import datetime
import os

SESSION_ID = str(uuid.uuid4())[:8]  # lightweight anonymous session tracking

ANALYTICS_DIR = "analytics"
os.makedirs(ANALYTICS_DIR, exist_ok=True)

USAGE_PATH = os.path.join(ANALYTICS_DIR, "usage_log.csv")

def log_usage(event_type: str):
    """Append a lightweight usage event."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    row = pd.DataFrame([{
        "timestamp_utc": now,
        "session_id": SESSION_ID,
        "event": event_type
    }])

    if not os.path.exists(USAGE_PATH):
        row.to_csv(USAGE_PATH, index=False)
    else:
        row.to_csv(USAGE_PATH, mode="a", header=False, index=False)

# Log page visit
log_usage("page_loaded")

AI_QA_PATH = os.path.join(ANALYTICS_DIR, "ai_questions.csv")

def log_ai_question(question: str, scenario: str):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    row = pd.DataFrame([{
        "timestamp_utc": now,
        "session_id": SESSION_ID,
        "scenario": scenario,
        "question": question
    }])

    if not os.path.exists(AI_QA_PATH):
        row.to_csv(AI_QA_PATH, index=False)
    else:
        row.to_csv(AI_QA_PATH, mode="a", header=False, index=False)


TITLES_ENRICHED_PATH = Path("data") / "titles_enriched.csv"


@st.cache_data
def load_enriched_titles():
    if not TITLES_ENRICHED_PATH.exists():
        return None
    df = pd.read_csv(TITLES_ENRICHED_PATH)
    # Safety: ensure value_score_norm exists
    if "value_score_norm" not in df.columns:
        df["value_score_norm"] = 0.0
    return df


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

    x = (pts[:, 0] - pts[:, 0].min()) / (np.ptp(pts[:, 0]) + 1e-9)
    y = (pts[:, 1] - pts[:, 1].min()) / (np.ptp(pts[:, 1]) + 1e-9)

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
def apply_platform_aliases(txt: str) -> str:
    """
    Normalize platform names for display consistency.
    Example: Max/HBO -> HBO; Apple TV variants -> Apple TV+; remove dupes, trim.
    """
    if not txt:
        return txt
    s = str(txt)

    # HBO / Max
    s = s.replace("HBO Max", "HBO")
    s = s.replace("Max", "HBO")  # show as HBO for simplicity

    # Apple TV+ variants
    s = s.replace("Apple TV Plus", "Apple TV+")
    s = s.replace("AppleTV+", "Apple TV+")
    s = s.replace("Apple TV +", "Apple TV+")
    s = s.replace("Apple Tv+", "Apple TV+")

    # Paramount+ variants
    s = s.replace("Paramount Plus", "Paramount+")
    s = s.replace("Paramount Plus Apple TV Channel", "Paramount+ Apple TV Channel ")

    # Remove accidental double spaces
    s = " ".join(s.split())

    # De-dupe comma-separated platforms while preserving order
    parts = [p.strip() for p in s.split(",")]
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return ", ".join(out)


# The order of default brands to show in the selectors
DEFAULT_BRANDS = [
    "Amazon",
    "Warner Brothers HBO",   # canonical brand for HBO/HBO Max/Max
    "Paramount Global",
    "Comcast (NBCUniversal)",
    "Disney",
    "Netflix",
    "Apple",
    "Sony",
    "Hulu",
    "Max",                   # UI option (normalized to Warner Brothers HBO)
    "Peacock",               # UI option (normalized to Comcast/NBCU below)
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

def infer_brand_text(txt: str | None) -> str | None:
    """
    Best-effort brand inference from an arbitrary platform/network string.
    Returns a canonical brand name or None.
    """
    if not txt:
        return None
    s = str(txt).lower()

    # Apple — match first to avoid accidental hits
    if "apple tv" in s or "appletv" in s or "apple tv+" in s or "apple tv plus" in s:
        return "Apple"

    # Amazon
    if "prime video" in s or "amazon" in s or "freevee" in s:
        return "Amazon"

    # HBO/Max
    if "hbo" in s or "max" in s or "warner bros" in s or "wbd" in s:
        return "Warner Brothers HBO"

    # Peacock / NBCU
    if "peacock" in s or "nbc" in s or "comcast" in s:
        return "Comcast (NBCUniversal)"

    # Paramount
    if "paramount" in s or "showtime" in s:
        return "Paramount Global"

    # Disney / Hulu
    if "disney" in s or "hulu" in s or "star+" in s:
        return "Disney"

    # Netflix
    if "netflix" in s:
        return "Netflix"

    # Sony (rare in streaming labels, but keep)
    if "sony" in s:
        return "Sony"

    return None

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
/* Netflix */
.netflix-text {
    color: #E50914;  /* Netflix red */
    font-weight: 800;
}

/* Warner Bros */
.wb-text {
    color: #EFBF04;  /* brighter WB-style blue, pops on dark bg */
    font-weight: 800;
    text-shadow: 0 0 4px rgba(0, 0, 0, 0.6);
}

/* Hero title */
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    text-align: center;
}

/* Subtitle */
.hero-subtitle {
    text-align: center;
    font-size: 1.25rem;
    color: #1E88FF;
    margin-top: 0.5rem;
    opacity: 0.95;
}

/* Tagline */
.hero-tagline {
    text-align: center;
    margin-top: 0.8rem;
    font-size: 1.05rem;
    opacity: 0.88;
}

/* Jump link */
.jump-link-wrapper {
    text-align: center;
    margin-top: 2rem;
    margin-bottom: 1.5rem;
}

.jump-link-wrapper a {
    font-size: 1rem;
    font-weight: 500;
    color: #1E88FF;
    text-decoration: underline;
}

.header-band {
  background: linear-gradient(90deg, #111827 0%, #020617 40%, #111827 100%);
  padding: 2.5rem 1.5rem 1.5rem 1.5rem;
  border-radius: 1.5rem;
  border: 1px solid rgba(148, 163, 184, 0.25);
  margin-bottom: 1.5rem;
}
.appview-container .main .block-container {
  background-image: url("assets/bg_hero.png");
  background-size: cover;
  background-repeat: no-repeat;
  background-position: top center;
}
.jump-link-wrapper a {
  font-size: 1rem;
  font-weight: 500;
  color: #1E88FF;
  text-decoration: underline;
  animation: pulse 2.5s infinite;
}

@keyframes pulse {
  0%   { opacity: 0.9; }
  50%  { opacity: 0.5; }
  100% { opacity: 0.9; }
}


.jump-link-wrapper a:hover {
    opacity: 0.75;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
/* App-wide background image with a dark overlay for readability */
.stApp {
    background: 
        linear-gradient(rgba(3, 7, 18, 0.90), rgba(3, 7, 18, 0.96)),
        url("assets/streamshift_bg.png");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: top center;
}

/* Make the main content slightly transparent so the bg peeks through subtly */
.block-container {
    background-color: rgba(3, 7, 18, 0.92);
}
</style>
""",
    unsafe_allow_html=True,
)




#st.image("assets/bg_hero.png", use_column_width=True)

st.markdown(
    """
<div class="hero-wrapper">
  <div class="hero-title">
  <span class="netflix-text">Netflix</span> -
  <span class="wb-text">Warner Bros</span> Deal
  </div>
  <div class="hero-subtitle">
    Media Merger Analysis for the Biggest Streaming Shift Yet
  </div>
  <div class="hero-tagline">
    A data-backed, first-pass framework for <strong>“who keeps what”</strong>
    after the Netflix–Warner Bros Discovery announcement.
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
<div style='margin-top:10px; padding:10px; border-left: 3px solid #E50914;'>
<strong>AI Deal Analyst available:</strong>  
You can ask questions about the Netflix × Warner Bros deal — scenario impacts, platform exposure, 
and title-level dynamics. Responses are generated using the dashboard’s real data and selected assumptions.
</div>
""",
    unsafe_allow_html=True,
)


st.markdown(
    "<div class='hero-link'><a href='#sources-section'>Jump to deal coverage & sources →</a></div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='hero-link'><a href='#target-audiance'> THIS DASHBOARD IS BUILT FOR →</a></div>",
    unsafe_allow_html=True,
)

with st.container():
    st.markdown("### Deal summary — what’s actually been announced")

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown(
            f"""
**Headline**

Netflix has entered a **definitive agreement** to acquire the **studio and streaming assets of Warner Bros. Discovery** in a cash-and-stock deal valuing the business at **~$82.7B enterprise value (~$72B equity)**, pending regulatory approval.

**Structure**

- **Acquirer:** Netflix, Inc.  
- **Seller:** Warner Bros. Discovery (post spin-off of its linear-TV unit *Discovery Global*)  
- **Included:**  
  - Warner Bros. film & TV studios  
  - HBO / HBO Max streaming business & libraries  
  - DC Entertainment / DC Studios  
  - Distribution & licensing units and related games/content IP  
- **Excluded:**  
  - Linear cable networks (CNN, Discovery Channel, etc.), which are expected to sit in a separate “Discovery Global” company

**Timing & status**

- Deal **announced:** 5 Dec 2025  
- Expected close: **after the Discovery Global spin-off**, targeted for **Q3 2026**  
- Status: **Pending** regulatory and shareholder approvals
"""
        )

    with col2:
        st.markdown(
            f"""
**Why this matters for this dashboard**

- Puts **Netflix in control of HBO / WB / DC IP**, not just licensed windows  
- Raises questions about **where flagship series will live** (Netflix vs. HBO Max vs. third-party platforms)  
- Creates one of the **largest streaming content libraries** under a single decision-maker

**Regulatory & political backdrop**

- U.S. and EU regulators are expected to probe the deal for **market power / competition** concerns  
- Industry groups and politicians have already signalled **pushback on consolidation and pricing power**

**Last updated:** {LAST_UPDATED}

> This summary reflects **public reporting** from outlets such as Netflix IR, AP, Reuters, FT, Variety, and others as of the date above.  
> It is **context**, not investment or legal advice.
"""
        )


with st.expander("Methodology & Data Sources"):
    st.markdown("""
**What this tool uses**

- **TMDb (The Movie Database)** for title-level metadata  
  - Popularity index, vote counts, ratings  
- **Curated Warner Bros Discovery IP list**  
  - Flagship franchises (Harry Potter, DC, LOTR, HBO dramas, etc.)  
- **Current platform mapping**  
  - Where titles are currently streaming (Max, Netflix, Netflix/Max, etc.)  
- **Simple value score**  
  - Combines popularity and vote count, normalized to 0–100  
- **Platform exposure logic**  
  - Owned by WBD (Max) → lower incremental risk  
  - Already partly on Netflix → medium risk  
  - On third-party platforms → higher exposure if Netflix internalizes WB IP  

**What this is _not_**

- It is **not** a view into private contract details  
- It does **not** forecast exact legal outcomes or timing  
- It is **not** investment advice  

Instead, it gives a **transparent, consistent framework** for thinking about 
how high-value WB titles might move *if* Netflix consolidates content over time.
""")

st.markdown(
    "<div class='section-heading'>RIGHTS & LICENSING REALITY CHECK</div>",
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.4, 1])

with left_col:
    st.markdown(
        """
This dashboard is about **content impact**, not contract forensics.

To keep the model honest, here’s how to read the output:

**What we treat as “owned” vs “licensed”**

- If a title is produced by a **Warner Bros / HBO / DC studio**, we treat it as **WB-controlled IP**.  
- If a title mainly streams on **Max** today, we assume WB (and post-deal Netflix) has **strong control** over future placement.  
- If a title streams on **Netflix and a third-party platform**, we treat it as **licensed / shared**.  
- If a title lives only on **other platforms** (Prime, Hulu, etc.) but is WB-produced, we tag it as **exposed** if Netflix consolidates WB content.

**What this model approximates**

- Relative **value** of WB titles using TMDb popularity + vote data  
- How much **“leverage”** major WB IP gives Netflix if pulled in-house  
- Which **platforms are most exposed** to a Netflix–WB consolidation of content  
- High-level **“stay / licensed / exclusive”** outcomes based on ownership + current home

**What this model does *not* claim**

- It does **not** know the fine print of every output deal or carve-out.  
- It does **not** predict exact timing of when a title will move.  
- It does **not** override regulatory decisions, guild/union agreements, or talent renegotiations.  

Think of this as a **first-pass deal prep lens**:  
*“If Netflix ends up with full WB control, where are the obvious content and platform pressure points?”*  
Use it to frame questions and sanity-check narratives, not as a binding rights database.
        """
    )

with right_col:
    st.markdown(
        """<div class='section-heading' style='margin-top:0;'>WHAT THIS DASHBOARD HELPS YOU ANSWER</div>

**For analysts, reporters & strategists**

- **Which WB titles matter most to Netflix?**  
  → Check the **Deal Deck metrics** and **Spotlight IPs** ranked by value score.

- **Which platforms lose leverage if Netflix pulls WB content in-house?**  
  → See the **Platform Exposure Risk** table.

- **How do WB franchises cluster by audience DNA?**  
  → Use the **IP similarity map** (UMAP/PCA scatter).

- **How big is the “value pool” Netflix consolidates in aggressive scenarios?**  
  → Read the **deal-impact snapshot** headline above the risk table.

Use this page as a **one-stop prep sheet** before building slides, writing notes, or doing deeper Excel modeling.
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='section-heading'>KEY QUESTIONS AROUND THIS DEAL</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
Use this dashboard as a **first-pass lens** on questions like:

- **Which WB franchises become true differentiators for Netflix?**  
  → e.g., *Game of Thrones, Harry Potter, DC, LOTR*, and which long-tail titles quietly matter.

- **Which platforms are most exposed if Netflix pulls WB content closer over time?**  
  → Which Max / third-party deals look most vulnerable under aggressive consolidation.

- **How much “value pool” is at stake in different scenarios?**  
  → How much of WB’s modeled content value currently touches other platforms vs. Netflix.

- **Where will fans feel the impact first?**  
  → Which shows are likely to move, disappear, or become harder to binge outside Netflix.

- **What story do investors / reporters tend to miss?**  
  → That it’s not just about one or two flagships, but about clusters of similar IP and the 
    **aggregate leverage** Netflix gains over time.
"""
)


with st.expander(" New to streaming deals? Start here"):
    st.markdown("""
**Key ideas in plain English**

- A show has an **owner** (who owns the IP) and a **home** (where it streams).  
- Sometimes the owner and the streamer are the same company; sometimes they are not.  
- When a big merger happens, the new parent company can **pull shows closer** over time:
  - Keep them where they are until contracts expire  
  - Move them to their own platform when it makes strategic sense  
- High-value titles (huge fandom, lots of viewing) have **outsized impact**:
  - They drive subscriptions, reduce churn, and anchor bundles  
- This tool:
  - Lists key WB franchises  
  - Uses audience data (via TMDb) to estimate **relative importance**  
  - Shows where those titles live today  
  - Highlights where the **biggest competitive shocks** could be if Netflix consolidates them.
""")


# --- Brand matching helpers (place near imports / after reading CSV) ---
BRAND_PATTERNS = {
    "Amazon": ["Amazon", "Prime Video"],
    "Warner Brothers HBO": ["HBO", r"\bMax\b", "Warner", "WBD", "Warner Bros"],
    "Paramount Global": ["Paramount", "Showtime"],
    "Comcast (NBCUniversal)": ["Peacock", "NBC", "NBCUniversal", "Comcast"],
    "Disney": ["Disney", "Hotstar", r"Star\+"],
    "Netflix": ["Netflix"],
    "Apple": [r"Apple TV\+"],
    "Sony": ["Sony", "Crunchyroll"],   # adjust if you don’t want Crunchyroll here
    "Hulu": ["Hulu"],
    "Max":  [r"\bMax\b", "HBO"],
    "Peacock": ["Peacock"],
}

def brand_regex(brand: str) -> str:
    pats = BRAND_PATTERNS.get(brand, [brand])
    return r"(?i)(" + "|".join(pats) + r")"

def filter_for_buyer_target(fr: pd.DataFrame, buyer: str, target: str) -> pd.DataFrame:
    """
    Return rows relevant to buyer or target based on platform or origin/brand hints.
    - Looks in current_platform and origin_label/original_network
    - Uses brand-specific regex patterns (expanded with Apple TV+ variants)
    """
    BRAND_PATTERNS = {
        "Amazon": r"amazon|prime video|freevee",
        "Warner Brothers HBO": r"hbo|max|warner bros|wbd",
        "Comcast (NBCUniversal)": r"peacock|nbc|comcast|universal",
        "Paramount Global": r"paramount\+|paramount plus|showtime",
        "Disney": r"disney\+|hulu|star\+",
        "Netflix": r"netflix",
        "Apple": r"apple tv\+|apple tv plus|appletv\+|apple tv",
        "Sony": r"\bsony\b",
        "Hulu": r"\bhulu\b",
        "Max": r"\bmax\b|hbo",
        "Peacock": r"\bpeacock\b",
    }

    buy_pat = BRAND_PATTERNS.get(buyer, buyer.lower())
    tgt_pat = BRAND_PATTERNS.get(target, target.lower())

    def _text_cols(row) -> str:
        return " | ".join([
            str(row.get("current_platform", "")),
            str(row.get("origin_label", "")),
            str(row.get("original_network", "")),
            str(row.get("original_brand", "")),
        ]).lower()

    df2 = fr.copy()
    hay = df2.apply(_text_cols, axis=1)

    buy_mask = hay.str.contains(buy_pat, regex=True, na=False)
    tgt_mask = hay.str.contains(tgt_pat, regex=True, na=False)
    out = df2.loc[buy_mask | tgt_mask].copy()

    # As a last resort, infer from platform text if origin is missing
    if "origin_label" in out.columns:
        empty_origin = out["origin_label"].fillna("").eq("")
        out.loc[empty_origin, "origin_label"] = out.loc[empty_origin, "current_platform"].apply(infer_brand_text)

    return out.reset_index(drop=True)


# ---------------- Fixed Scenario: Netflix × Warner Bros Discovery ----------------
buyer = "Netflix"
target = "Warner Brothers HBO"

st.markdown("""
<div class='toolbar'>
  <p><strong>Scenario:</strong> This special edition focuses on <strong>Netflix</strong> acquiring 
  <strong>Warner Bros Discovery / HBO</strong>. Other combinations are out of scope by design.</p>
</div>
""", unsafe_allow_html=True)

log_usage(f"scenario_fixed_{buyer}_{target}_netflix_wbd")

st.markdown(
    "<div class='section-heading'>SCENARIO ASSUMPTIONS</div>",
    unsafe_allow_html=True,
)

scenario = st.radio(
    "How aggressively do you think Netflix will pull WB content in-house?",
    options=["Conservative", "Base case", "Aggressive"],
    index=1,
    horizontal=True,
    help=(
        "Conservative: only the most exposed titles move.\n"
        "Base case: matches current model.\n"
        "Aggressive: assume Netflix eventually consolidates most WB tentpoles."
    ),
)


# ---------------- Business-Impact Mode: Netflix × Warner Bros ----------------
if buyer == "Netflix" and "Warner" in str(target):
    titles_df = load_enriched_titles()

    if titles_df is not None and not titles_df.empty:
        # Sort once, reuse everywhere inside this block
        titles_sorted = titles_df.sort_values("value_score_norm", ascending=False)

        st.markdown("### Deal Deck – Netflix × Warner Bros IP Snapshot")

        # Basic metrics
        total_titles = len(titles_sorted)
        high_value_cutoff = 70.0
        high_value_count = int((titles_sorted["value_score_norm"] >= high_value_cutoff).sum())
        top_franchises = (
            titles_sorted["franchise_group"]
            .value_counts()
            .head(3)
            .index.tolist()
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total WB Titles in Focus", f"{total_titles}")
        with c2:
            st.metric("High-Value IPs (score ≥ 70)", f"{high_value_count}")
        with c3:
            st.metric("Top Franchise Clusters", ", ".join(top_franchises) if top_franchises else "–")

        top5 = titles_sorted.head(5)[["title", "franchise_group", "value_score_norm"]]
        st.markdown("#### Top IPs by Attention & Popularity (TMDb-based)")
        st.dataframe(
            top5.rename(
                columns={
                    "title": "Title",
                    "franchise_group": "Franchise",
                    "value_score_norm": "Value Score (0–100)",
                }
            ),
            use_container_width=True,
        )

        st.caption(
            "Value scores are derived from TMDb popularity and vote counts, normalized to 0–100. "
            "Data from TMDb; this view is illustrative, not investment advice."
        )

        # ---- Poster Spotlight ----
        st.markdown(
            """
        <span style='padding:2px 8px; border-radius:999px; background-color:#E50914; font-size:0.7rem; margin-right:4px;'>Netflix</span>
        <span style='padding:2px 8px; border-radius:999px; background-color:#1E88FF; font-size:0.7rem;'>Warner Bros</span>

        ### Spotlight IPs – What Makes WB Valuable to Netflix
        """,
            unsafe_allow_html=True,
        )

        spotlight = titles_sorted.head(4)
        cols = st.columns(4)
        for i, (_, row) in enumerate(spotlight.iterrows()):
            with cols[i]:
                poster_url = (
                    f"https://image.tmdb.org/t/p/w300{row['tmdb_poster_path']}"
                    if pd.notna(row["tmdb_poster_path"]) and row["tmdb_poster_path"]
                    else None
                )
                if poster_url:
                    st.image(poster_url, use_column_width=True)

                st.markdown(
                    f"**{row['title']}**  \n"
                    f"{row['franchise_group']} • *Value Score:* **{row['value_score_norm']:.1f}**"
                )

        # ---- Platform Exposure Risk ----

        st.markdown(
            "<div class='section-heading'>SCENARIO ASSUMPTIONS</div>",
            unsafe_allow_html=True,
        )

        scenario = st.radio(
            "How aggressively do you think Netflix will pull WB content in-house?",
            options=["Conservative", "Base case", "Aggressive"],
            index=1,
            horizontal=True,
            help=(
                "Conservative: only the most exposed titles move.\n"
                "Base case: matches current model.\n"
                "Aggressive: assume Netflix eventually consolidates most WB tentpoles."
            ),
            key="scenario_assumptions_radio",  # <-- unique key
        )


        scenario_multiplier = {
            "Conservative": 0.6,
            "Base case": 1.0,
            "Aggressive": 1.4,
        }[scenario]

        st.markdown(
            """
        <span style='padding:2px 8px; border-radius:999px; background-color:#E50914; font-size:0.7rem; margin-right:4px;'>Netflix</span>
        <span style='padding:2px 8px; border-radius:999px; background-color:#1E88FF; font-size:0.7rem;'>Warner Bros</span>

        ### Platform Exposure Risk  
        <small>Which WB titles create the most disruption outside Netflix if brought in-house.</small>
        """,
            unsafe_allow_html=True,
        )


        def compute_risk_score(row):
            platform = str(row["current_platform"])
            value = row["value_score_norm"]

            if "Max" in platform:
                return value * 0.2  # low disruption / exposure (owned by WBD)
            if "Netflix" in platform:
                return value * 0.4  # medium exposure (already partly on Netflix)
            return value * 1.0      # highest exposure for third-party platforms


        # Base risk dataframe
        risk_df = titles_sorted.copy()
        risk_df["risk_score"] = risk_df.apply(compute_risk_score, axis=1)

        # ---- Scenario-adjusted risk dataframe ----
        # We assume titles_sorted has: title, franchise_group, current_platform, value_score_norm
        risk_df_scenario = risk_df.copy()
        risk_df_scenario["Risk Score"] = risk_df_scenario["risk_score"] * scenario_multiplier

        # Rename internal columns to display names once
        risk_display = risk_df_scenario.rename(
            columns={
                "title": "Title",
                "franchise_group": "Franchise",
                "current_platform": "Currently On",
            }
        )

        # Build the "who is most exposed" table
        display_cols = ["Title", "Franchise", "Currently On", "Risk Score"]

        risk_top = (
            risk_display
            .sort_values("Risk Score", ascending=False)
            [display_cols]
            .head(6)
        )

        st.dataframe(risk_top, use_container_width=True, hide_index=True)
        st.caption("Risk score = value score × non-Netflix exposure, scaled by the scenario above.")

        # ---- Evidence-based headline summary ----
        total_value = titles_sorted["value_score_norm"].sum()
        top_risk_value = risk_top["Risk Score"].sum() if not risk_top.empty else 0.0
        share_pct = (top_risk_value / total_value * 100) if total_value > 0 else 0.0

        scenario_label = {
            "Conservative": "only the most exposed titles move",
            "Base case": "a gradual consolidation of key WB series",
            "Aggressive": "an aggressive pull-in of most WB tentpoles",
        }[scenario]

        headline = (
            f"Under a **{scenario}** view ({scenario_label}), the top {len(risk_top)} exposed WB titles "
            f"represent roughly **{share_pct:.1f}%** of the modeled WB value pool that currently touches non-Netflix platforms."
        )

        st.markdown(f"#### Deal-impact snapshot\n{headline}")

        # ---- AI-style explainer for a selected WB title ----
        st.markdown(
            "<div class='section-heading'>EXPLAIN THIS TITLE</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "Pick a WB title and get a short, scenario-aware explanation of why it matters in this deal."
        )

        # Build a simple list of titles sorted by risk under the current scenario
        explainer_df = (
            risk_display.sort_values("Risk Score", ascending=False)
            [["Title", "Franchise", "Currently On", "Risk Score", "value_score_norm"]]
            .drop_duplicates(subset=["Title"])
        )

        if not explainer_df.empty:
            default_title = explainer_df["Title"].iloc[0]
            selected_title = st.selectbox(
                "Choose a title to explain",
                explainer_df["Title"].tolist(),
                index=0,
                key="explain_title_select",
            )
            log_usage(f"explain_title_select{selected_title}")

            row = explainer_df[explainer_df["Title"] == selected_title].iloc[0]
            franchise = row["Franchise"]
            platform = row["Currently On"]
            risk_score = row["Risk Score"]
            value_score = row["value_score_norm"]

            # Very simple interpretation of platform situation
            platform_note = "currently anchored on Max."
            if "Netflix" in platform and "Max" in platform:
                platform_note = "shared between Netflix and Max right now."
            elif "Netflix" in platform and "Max" not in platform:
                platform_note = "already strongly associated with Netflix."
            elif "Netflix" not in platform and "Max" not in platform:
                platform_note = f"primarily living on {platform}, outside Netflix and Max."

            # Scenario sentence
            scenario_label = {
                "Conservative": "only modest movement from existing windows",
                "Base case": "a gradual pull of key WB hits into Netflix over time",
                "Aggressive": "an aggressive consolidation of most WB tentpoles under Netflix",
            }[scenario]

            st.markdown(
                f"""
        **Title:** `{selected_title}`  
        **Franchise:** `{franchise}`  
        **Current home:** `{platform}`  

        > **Why this matters under a {scenario} view**

        - This title sits in the **top tier of WB content** by modeled value (normalized score ≈ `{value_score:.2f}`).
        - Its **platform exposure** risk is relatively high (scenario-adjusted risk score ≈ `{risk_score:.2f}`) because it is {platform_note}
        - Under the current scenario, we assume **{scenario_label}**, which makes this title a **useful signal** for:
        - How far Netflix is willing to go on **exclusivity** vs. shared licensing.
        - How much **churn / leverage** Max and other platforms could lose if Netflix pulls WB content closer.
        - How fast fans would *feel* the deal in their day-to-day app usage (search, recommendations, binge behavior).

        **How an analyst might use this**

        - In a deck or note, this becomes an example of *“high-value WB IP where the pain lands first”*  
        - You can pair it with the **Platform Exposure Risk table** and **Spotlight IPs** to show both **micro** (this title) and **macro** (franchise & platform) impact.
        """
            )
        else:
            st.info("No titles available to explain for this scenario.")

        # ---- AI DEAL ANALYST (PREMIUM MODE) ----
        st.markdown(
            "<div style='font-size:1.1rem; font-weight:600; margin-bottom:6px;'>Ask the AI Deal Analyst About The Netflix–WB Deal:</div>",
            unsafe_allow_html=True,
        )

        st.markdown("scenario impacts, or how to use this dashboard for reporting and notes.")

        user_q = st.text_input(
            "Your question",
            placeholder="e.g., Which platforms lose the most leverage if Netflix consolidates the top WB franchises?",
        )

        # Only log when user types something
        if user_q:
            log_usage("ai_question_typed")
            log_ai_question(user_q, scenario)


        # Helper to build structured context for the LLM
        def build_context():
            ctx = {
                "deal_summary": deal_summary_text,
                "scenario": scenario,
                "scenario_multiplier": scenario_multiplier,
                "top_risk_titles": risk_top.to_dict(orient="records"),
                "spotlight_titles": titles_sorted.head(8).to_dict(orient="records"),
            }
            return json.dumps(ctx, indent=2)


        # ---- Handle AI Response ----
        if user_q:
            # If OpenAI is not available, show graceful fallback
            if not _openai_ok:
                st.warning(
                    "⚠️ The AI Deal Analyst is temporarily unavailable on this deployment "
                    "(OpenAI client is not installed or configured). The rest of the dashboard works normally."
                )
            else:
                with st.spinner("Analyzing with your scenario, exposure scores, and franchise data..."):
                    try:
                        sys_prompt = f"""
        You are an elite senior media strategy analyst specializing in mergers, content economics,
        and platform exposure. You answer concisely, using evidence from the provided structured data only.

        Here is the structured context from the dashboard (deal summary, scenario, top risk titles, spotlight IP, risk scores):
        {build_context()}

        Rules:
        - Never hallucinate titles not in the provided data.
        - You MUST tie your answer to scenario label and risk scores where relevant.
        - If user asks for a prediction, phrase it as directional analysis not certainty.
        - No more than 8 sentences unless asked.
        """

                        completion = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": user_q},
                            ],
                            max_tokens=350,
                        )

                        answer = completion.choices[0].message["content"]
                        st.markdown(f"### AI Analyst Response\n{answer}")

                    except Exception as e:
                        st.error(
                            "AI unavailable. Falling back to basic answers.\n\n"
                            f"Error: {e}"
                        )





        st.dataframe(
            risk_top.rename(
                columns={
                    "title": "Title",
                    "franchise_group": "Franchise",
                    "current_platform": "Currently On",
                    "risk_score": "Risk Score",
                }
            ),
            use_container_width=True,
        )

        st.markdown(
            """
**What this table tells you**

- Ranks WB titles by how much **competitive disruption** they create for other platforms  
- **Higher scores** = more valuable + less already controlled by Netflix/WBD  
- Useful for thinking about **licensing fallout, churn risk, and negotiating leverage**
            """
        )

        with st.expander("How exposure is calculated"):
            st.markdown(
                """
- **Max titles** → low incremental exposure (already owned by Warner Bros Discovery)  
- **Netflix/Max titles** → medium exposure (shared or licensed)  
- **Other platforms** → highest exposure if Netflix consolidates WB IP  
Risk score = *value score × platform-exposure multiplier*.
                """
            )

    else:
        st.info(
            "Enriched title data not available yet. Run the enrichment step to enable the Deal Deck view."
        )
else:
    # For now, keep the app stable and on-message if users switch buyer.
    st.info(
        "This special edition focuses on the Netflix × Warner Bros deal. "
        "For the full Business-Impact view, set Buyer to Netflix and Target to Warner Brothers HBO."
    )

st.markdown("<div id='target-audiance'></div>", unsafe_allow_html=True)
st.markdown("### 🔗 WHO THIS DASHBOARD IS BUILT FOR")    

#st.markdown(
#    "<div class='section-heading'>WHO THIS DASHBOARD IS BUILT FOR</div>",
#    unsafe_allow_html=True,
#)


st.markdown(
    """
**Who might actually use this in real life**

- **Equity & credit analysts**  
  Quickly gauge *which WB IP drives leverage* for Netflix and *which platforms are most exposed* if content is pulled in-house.

- **Streaming & content strategy teams**  
  Use it as a first-pass view before deeper Excel / internal rights work when prepping **churn / pricing / windowing** scenarios.

- **Reporters & newsletter writers**  
  Gut-check headlines like *“Netflix wins the HBO universe”* with a simple **“who keeps what, who loses leverage”** view.

- **Students / junior analysts / interns**  
  As a concrete example of turning messy IP + contract noise into a **structured, evidence-backed framework**.

**How to use it in <5 minutes**

1. **Skim the deal summary** to anchor what’s actually been announced.  
2. **Pick your scenario** (Conservative / Base / Aggressive) in *Platform Exposure Risk*.  
3. Look at **Spotlight IPs** to see which WB franchises matter most to Netflix on a value basis.  
4. Check **Platform Exposure Risk** to see which titles create the most disruption outside Netflix.  
5. Use the **deal-impact snapshot** sentence as a starting line for decks, emails, or notes.

It's meant to be a **first-pass deal prep tool** — something you open *before* the 50-page model or 80-page bank deck, not instead of them.
"""
)

# Normalize display selections to canonical internal brand ids
def _canonical(b: str) -> str:
    b = (b or "").strip()
    if b == "Max":                         # treat Max as the HBO brand
        return "Warner Brothers HBO"
    if b == "Peacock":                     # treat Peacock as Comcast (NBCU)
        return "Comcast (NBCUniversal)"
    return b

buyer_c = _canonical(buyer)
target_c = _canonical(target)

# keep labels for optional sentiment; store canonical for logic
st.session_state["buyer_label"] = buyer
st.session_state["target_label"] = target
st.session_state["buyer_canon"] = buyer_c
st.session_state["target_canon"] = target_c

# --- Side-by-side: IP Similarity Map (left) and Rippleboard (right) ---
# --- Side-by-side: IP Similarity Map (left) and Rippleboard (right) ---
L, R = st.columns([1, 1], vertical_alignment="top")

with L:
    st.subheader("✣ IP Similarity Map")
    st.markdown(
    """
        **What this shows**  
        Turning titles into vectors (fancy math), squash to 2D, and color by cluster.  
        Closer dots → **similar audience DNA**. Use it like a cross-sell radar.  
        Positions are from a 2-D projection of text embeddings (UMAP/PCA).
        
        **How to read the IP Similarity Map**
        - Each dot = a show/film.
        - Closer dots = more similar audience DNA (genre/keywords/description).
        - Colors = rough clusters (e.g., prestige drama, sci-fi, comedy).
        - Use it to spot good fits for the buyer (dots near the buyer’s current slate) vs outliers (harder brand fit).
        - **Caveat:** it’s a content-text signal, not a rights contract—pair with the Rippleboard and sources.
        
        **Investor lens**
        - Fit suggests cross-sell/retention upside if pulled exclusive.
        - Outliers may be better left licensed out (cash engine) rather than pulled in.
        - Combine with Stay/Licensed/Exclusive calls for a quick timing + exposure picture.
        """
)


    # ---------------- Map engine: Embeddings → TF-IDF fallback ----------------
    try:
        # Build corpus: title + optional tags
        if "genre_tags" not in fr.columns:
            fr["genre_tags"] = ""
        corpus = (fr["title"].astype(str).str.strip() + " " +
                  fr["genre_tags"].astype(str).str.strip()).str.strip()
        mask = corpus.str.len() > 0
        df_map = fr.loc[mask].reset_index(drop=True).copy()

        X = None
        used_embeddings = False

        # 1) Try Sentence-Transformers
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            X = model.encode(
                df_map["title"].astype(str).str.cat(df_map["genre_tags"].astype(str), sep=" ").tolist(),
                batch_size=64, show_progress_bar=False, normalize_embeddings=True
            ).astype("float32")
            used_embeddings = True
        except Exception:
            # 2) Fall back to TF-IDF if ST is unavailable (e.g., Streamlit Cloud)
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
                tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=1)
                X = tfidf.fit_transform(corpus.tolist()).toarray().astype("float32")
                used_embeddings = False
            except Exception as e:
                st.caption(f"Map unavailable (no embeddings, no scikit-learn): {e}")
                X = None

        engine_label = "Embeddings • FAISS/ST" if used_embeddings else "Classic • TF-IDF"
        st.caption(f"Similarity engine: {engine_label}")

        if X is not None and len(df_map) >= 3:
            mx, my = project_points(X)  # uses np.ptp internally (NumPy 2-safe)
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

with R:
    st.subheader("Rippleboard: The Future of Content")
    st.markdown(
    """
        **What this is**  
        A post-deal TV guide for suits (but in plain English).  
        Each title gets a status (**stay / licensed / exclusive**) and a one-liner.  
        Tweak the rule and watch the board shift.
        
        **How to read this**
        - **Stay** → Likely the contract/window keeps it where it is for now.
        - **Licensed** → Shared/syndicated outcome likely (may move in parts or by region).
        - **Exclusive** → If the buyer owns the IP, they’d likely pull it in-house at renewal.
        - *Note:* Spin-offs depend on **derivative rights**, not just today’s streamer.
        """
)


    # Filter rows related to Buyer or Target (platform or origin label)
    rb = filter_for_buyer_target(fr, buyer, target)  # use FR, not df

    # Visible columns
    rb_view = rb[["title", "predicted_policy", "current_platform"]].copy()
    rb_view.rename(columns={
        "title": "IP / Franchise",
        "predicted_policy": "Predicted Status",
        "current_platform": "Current Platform"
    }, inplace=True)

    # Synthesize notes when missing
    if "Notes" not in rb.columns:
        rb_view["Notes"] = rb.apply(
            lambda r: (str(r.get("predicted_policy", "")).strip()
                       or synth_note(str(r.get("origin_label", "")), buyer, target)),
            axis=1,
        )
        # Reorder
        rb_view = rb_view[["IP / Franchise", "Predicted Status", "Notes", "Current Platform"]]

    # Alias HBO/Max for consistency
    rb_view["Current Platform"] = rb_view["Current Platform"].astype(str).apply(apply_platform_aliases)

    st.dataframe(rb_view, hide_index=True, use_container_width=True)

st.caption("Investor lens: ‘Stay’ often = locked windows/contracts; ‘Exclusive’ = potential near-term pull-in.")

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
    & (fr.get("original_brand", "").astype(str).str.lower() == target_c.lower())
    ][["title", "original_network", "producer_list", "current_platform", "source_urls"]].copy()


    # if empty, infer by platform/network brand
    if orig.empty:
        mask = fr.apply(
            lambda r: infer_brand_text(r.get("current_platform", "")) == target_c
            or infer_brand_text(r.get("original_network", "")) == target_c,
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

st.markdown("<div id='sources-section'></div>", unsafe_allow_html=True)
#st.markdown("### 🔗 Deal coverage & sources")
st.markdown("### 🔗 Deal coverage & sources")

st.markdown(
    """
Use this as a **one-stop reading list** around the Netflix–Warner Bros deal.  
The analytics on this page are meant to *complement*, not replace, these sources.

**Official transaction details**

- Netflix IR — *Netflix to Acquire Warner Bros Following the Separation of Discovery Global*  
  – [Press release](https://about.netflix.com/en/news/netflix-to-acquire-warner-bros) :contentReference[oaicite:0]{index=0}  
- Transaction terms & consideration mix  
  – [PR Newswire summary](https://www.prnewswire.com/news-releases/netflix-to-acquire-warner-bros-following-the-separation-of-discovery-global-for-a-total-enterprise-value-of-82-7-billion-equity-value-of-72-0-billion-302633998.html) :contentReference[oaicite:1]{index=1}  

**Hard news coverage**

- AP News — *Netflix to acquire Warner Bros studio and streaming business for $72 billion*  
  – [AP coverage](https://apnews.com/article/netflix-warner-acquisition-studio-hbo-streaming-f4884402cadfd07a99af0c8e4353bd83) :contentReference[oaicite:2]{index=2}  
- Reuters — *Instant view: Netflix to buy Warner Bros Discovery's studios, streaming unit*  
  – [Reuters instant view](https://www.reuters.com/legal/transactional/view-netflix-buy-warner-bros-discoverys-studios-streaming-unit-72-billion-2025-12-05/) :contentReference[oaicite:3]{index=3}  
- Financial Times — *Netflix agrees $83bn takeover of Warner Bros Discovery*  
  – [FT deal write-up](https://www.ft.com/content/6532be94-c0bf-4101-8126-f249aa6be3c5) :contentReference[oaicite:4]{index=4}  

**Financing & market angle**

- Bloomberg — *Netflix lines up $59bn of debt for Warner Bros deal*  
  – [Financing piece](https://www.bloomberg.com/news/articles/2025-12-05/netflix-lines-up-59-billion-of-debt-for-warner-bros-deal) :contentReference[oaicite:5]{index=5}  

**Reaction & impact**

- AP News — *Notable early reaction to Netflix’s deal to acquire Warner Bros*  
  – [Industry & political reaction](https://apnews.com/article/netflix-warner-bros-deal-reaction-3acea5d81e630d20560299764bf4c37c) :contentReference[oaicite:6]{index=6}  

This dashboard **sits on top of** that reporting:

- It uses these pieces to define the **scope** of the deal (what’s included / excluded).  
- Then it layers in **catalog data + simple rules** to visualize **who keeps what, who loses leverage, and which IP is most valuable** under different scenarios.
"""
)


with st.expander("Tech Stack"):
    st.markdown("""
- **Streamlit** for the interactive UI  
- **Pandas** for catalog modeling and scoring  
- **TMDb API** for title-level metadata  
- **Rule-based logic** for value scores & exposure  
- **LLM explainers** and **vector search** for deeper IP similarity.
    """)



# --- Quick feedback (lightweight, local only)


st.markdown("<div class='section-heading'>FEEDBACK</div>", unsafe_allow_html=True)
st.write("How useful was this dashboard?")

rating = st.radio(
    "Your rating:",
    ["Very useful", "Useful", "Not useful"],
    horizontal=True,
    key="feedback_rating",
)

feedback_text = st.text_area(
    "Optional comments",
    placeholder="What helped? What should we improve?",
    key="feedback_text",
)

if st.button("Submit feedback"):
    FB_PATH = os.path.join(ANALYTICS_DIR, "feedback.csv")
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    row = pd.DataFrame([{
        "timestamp_utc": now,
        "session_id": SESSION_ID,
        "rating": rating,
        "comment": feedback_text,
    }])

    if not os.path.exists(FB_PATH):
        row.to_csv(FB_PATH, index=False)
    else:
        row.to_csv(FB_PATH, mode="a", header=False, index=False)

    log_usage("feedback_submitted")
    st.success("Thank you! Your feedback has been recorded.")


# ---------------- Usage Footnote ----------------
#try:
#    if LOG_PATH.exists():
#        import pandas as pd  # uses existing import if already there
#        usage_df = pd.read_csv(LOG_PATH)
#        total_runs = len(usage_df)
#        st.caption(f"{total_runs} scenarios run so far.")
#except Exception as e:
#    # We never want a usage counter error to break the app
#    print(f"[usage counter error] {e}")

# ---------------- footer ----------------
#st.caption(
#    "Hobby demo for media M&A what-ifs. Data: TMDB where available; status/notes are testing for illustration."
#)'''
