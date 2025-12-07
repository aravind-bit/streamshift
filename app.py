# -------------------------------
# Imports & basic setup
# -------------------------------
import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import re
from pathlib import Path

# ---------- Lightweight CSV logging (usage + feedback) ----------
import csv
from datetime import datetime

# --- OpenAI availability flag ---
try:
    from openai import OpenAI
    client = OpenAI()
    _openai_ok = True
except Exception:
    _openai_ok = False

LOG_DIR = Path("analytics")
LOG_DIR.mkdir(exist_ok=True)

USAGE_LOG = LOG_DIR / "usage_log.csv"
FEEDBACK_LOG = LOG_DIR / "feedback_log.csv"


def _append_csv(path: Path, fieldnames, row: dict) -> None:
    """Append a row to CSV, writing header if file is new."""
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def log_usage(event_type: str, scenario: str | None = None, extra: dict | None = None) -> None:
    """
    Log simple usage events: page load, scenario change, AI question, etc.
    event_type: short code like 'page_load', 'scenario_change', 'ai_question'
    scenario:   'Conservative' / 'Base case' / 'Aggressive', or None
    extra:      optional dict that will be JSON-stringified
    """
<<<<<<< HEAD
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
    "<div class='hero-link'><a href='#target-audiance'> This Dashboard Is Built For: →</a></div>",
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
=======
    row = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "event_type": event_type,
        "scenario": scenario or "",
        "extra": json.dumps(extra, ensure_ascii=False) if extra is not None else "",
    }
    _append_csv(
        USAGE_LOG,
        fieldnames=["timestamp_utc", "event_type", "scenario", "extra"],
        row=row,
>>>>>>> 7dbf3bd (Update: WB sample expanded to 173 + AI Analyst + fallout simulator visuals)
    )

# --- Helper: derive "Top 5 Fallout Findings" from current data ----------------
def build_shock_findings(titles_df: pd.DataFrame, summary: dict, platform_exposure: pd.DataFrame, scenario: str) -> list[str]:
    findings = []

    # 1) Platform exposure headline
    top_platform = summary.get("top_platform", "another platform")
    outside_pct = summary.get("outside_pct", 0.0)
    findings.append(
        f"**{top_platform}** currently carries about **{outside_pct:.0f}%** of the modeled WB value that still lives outside Netflix."
    )

    # 2) Franchise headline
    top_franchise = summary.get("top_franchise", "a key WB franchise")
    findings.append(
        f"**{top_franchise}** screens as the single biggest WB franchise in this sample by modeled content value."
    )

    # 3) Rival platform “hurt index”
    if isinstance(platform_exposure, pd.DataFrame) and not platform_exposure.empty:
        pe = platform_exposure.sort_values("value", ascending=False).reset_index(drop=True)
        if len(pe) > 0:
            p_name = pe.loc[0, "platform"]
            p_share = float(pe.loc[0, "share"]) * 100.0 if "share" in pe.columns else float("nan")
            findings.append(
                f"Under the current catalog, **{p_name}** is the rival service most exposed to a Netflix–WB consolidation "
                f"(about **{p_share:.0f}%** of the modeled WB value that sits off Netflix)."
            )

        if len(pe) > 1:
            p2_name = pe.loc[1, "platform"]
            findings.append(
                f"Second in line: **{p2_name}**, which could lose key WB depth even if the very top franchises stay put."
            )
    else:
        findings.append(
            "Rival platform exposure is still being modeled; add more WB titles to sharpen where the pain lands."
        )

    # 4) Scenario framing
    scenario_label = {
        "Conservative": "only the most exposed titles move first",
        "Base case": "a measured pull-in of the highest-leverage WB series",
        "Aggressive": "a hard pivot where Netflix gradually becomes the default WB home",
    }.get(scenario, "a measured consolidation of WB content into Netflix")

    findings.append(
        f"With the slider set to **{scenario}**, the model assumes **{scenario_label}** — use this to stress-test best, base, and worst-case views."
    )

    # 5) “Long tail” note (if we have enough titles)
    if isinstance(titles_df, pd.DataFrame) and "title" in titles_df.columns:
        n_titles = int(titles_df["title"].nunique())
        findings.append(
            f"This view is built on **{n_titles}** modeled WB titles — enough to show how value clusters in a few big franchises while the long tail spreads across rivals."
        )

    # Keep exactly 5 lines (truncate if more)
    return findings[:5]


def log_feedback(rating: int, comment: str, scenario: str | None = None) -> None:
    """
    Log quick user feedback (1–5 stars + free-text comment).
    """
    row = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "rating": rating,
        "comment": comment.strip(),
        "scenario": scenario or "",
    }
    _append_csv(
        FEEDBACK_LOG,
        fieldnames=["timestamp_utc", "rating", "comment", "scenario"],
        row=row,
    )
# ---------- End logging helpers ----------


# If you removed logging_utils, you can delete this import.
from logging_utils import log_usage
#, log_ai_question

# Page config MUST be the first Streamlit call.
st.set_page_config(
    page_title="Netflix / Warner Bros Deal Impact Lab",
    layout="wide",
)

def render_poster_strip():
    """
    Shows a thin horizontal collage of WB / Netflix posters under the hero.
    Non-blocking: if there are no images, it silently does nothing.
    """
    poster_dir = Path("assets/posters")
    if not poster_dir.exists():
        return

    files = sorted(list(poster_dir.glob("*.jpg")) + list(poster_dir.glob("*.png")))

    if not files:
        return  # nothing to show

    # Keep it to first 8 to avoid clutter
    files = files[:8]

    st.markdown(
        "<div style='margin-top:10px; margin-bottom:6px; font-size:0.8rem; "
        "opacity:0.75;'>Sample of WB IP in this model</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(len(files))
    for col, f in zip(cols, files):
        # OLD: col.image(str(f), use_container_width=True)
        col.image(str(f), use_column_width=True)



# -------------------------------
# Paths / constants
# -------------------------------
DATA_DIR = Path("data")
#CATALOG_PATH = DATA_DIR / "titles_catalog.csv"
ENRICHED_PATH = DATA_DIR / "tites_enriched.csv"
#FRANCHISE_PATH = DATA_DIR / "franchises.csv"

# -------------------------------
# Base CSS & small UI helpers
# -------------------------------
def inject_base_css() -> None:
    """Inject shared CSS for tiles and headings."""
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .metric-tile {
            background: #111827;
            border-radius: 18px;
            padding: 1.1rem 1.3rem;
            border: 1px solid #1f2937;
        }
        .metric-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            color: #9CA3AF;
        }
        .metric-value {
            font-size: 1.7rem;
            font-weight: 700;
            color: #F9FAFB;
        }
        .section-heading {
            font-size: 1.05rem;
            font-weight: 600;
            margin: 1.5rem 0 0.6rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



def metric_tile(label: str, value: str, caption: str = "") -> None:
    """Reusable metric tile."""
    st.markdown(
        f"""
        <div class="metric-tile">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div style="font-size:0.8rem; color:#9CA3AF; margin-top:0.25rem;">
            {caption}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def section_heading(text: str) -> None:
    """Consistent section heading."""
    st.markdown(f"<div class='section-heading'>{text}</div>", unsafe_allow_html=True)


# Call once near the top of the app
inject_base_css()


# -------------------------------
# Data loading & core metrics
# -------------------------------

# #@st.cache_data(show_spinner=False)
# def load_titles() -> pd.DataFrame:
#     """Load catalog + enriched data and return a clean titles DataFrame."""
#     # Base catalog
#     try:
#         catalog = pd.read_csv(CATALOG_PATH)
#     except FileNotFoundError:
#         st.error(f"Missing catalog file at {CATALOG_PATH}")
#         return pd.DataFrame()

#     # Optional enrichment
#     enriched = None
#     try:
#         enriched = pd.read_csv(ENRICHED_PATH)
#     except FileNotFoundError:
#         enriched = None

#     if enriched is not None:
#         # Prefer merge on tmdb_id if available
#         if "tmdb_id" in catalog.columns and "tmdb_id" in enriched.columns:
#             df = catalog.merge(enriched, on="tmdb_id", how="left")
#         elif len(catalog) == len(enriched):
#             # Fallback: assume row-aligned and concat
#             df = pd.concat(
#                 [catalog.reset_index(drop=True), enriched.reset_index(drop=True)],
#                 axis=1,
#             )
#         else:
#             df = catalog.copy()
#     else:
#         df = catalog.copy()

#     # Deduplicate columns
#     df = df.loc[:, ~df.columns.duplicated()]
#     # ---- Normalize title column (handle title_x/title_y from merges) ----
#     title_cols = [c for c in df.columns if c.lower().startswith("title")]
#     if "title" in df.columns:
#         df["title"] = df["title"].astype(str)
#     elif title_cols:
#         # Take the first title-like column (e.g., title_x) as canonical
#         df["title"] = df[title_cols[0]].astype(str)
#     else:
#         # Fallback if somehow no title present
#         df["title"] = ""


#     # ---- Ensure core columns exist ----
#     # current_platform
#     if "current_platform" not in df.columns:
#         plat_cols = [c for c in df.columns if "platform" in c.lower()]
#         if plat_cols:
#             df["current_platform"] = df[plat_cols[0]]
#         else:
#             df["current_platform"] = ""
#     df["current_platform"] = df["current_platform"].astype(str).fillna("")

#     # franchise_group
#     if "franchise_group" not in df.columns:
#         fr_cols = [c for c in df.columns if "franchise" in c.lower()]
#         if fr_cols:
#             df["franchise_group"] = df[fr_cols[0]]
#         else:
#             df["franchise_group"] = "Standalone / other"
#     df["franchise_group"] = df["franchise_group"].astype(str).fillna("Standalone / other")

#     # value_score_norm
#     if "value_score_norm" not in df.columns:
#         if "value_score_raw" in df.columns:
#             raw = df["value_score_raw"].astype(float)
#             df["value_score_norm"] = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
#         else:
#             df["value_score_norm"] = 1.0  # flat fallback

#     # Sort by modeled value, highest first
#     df = df.sort_values("value_score_norm", ascending=False).reset_index(drop=True)
#     return df

DATA = Path("data")
TITLES_ENRICHED_PATH = DATA / "titles_enriched.csv"

@st.cache_data(show_spinner=False)
def load_titles() -> pd.DataFrame:
    """
    Load the modeled WB titles directly from titles_enriched.csv.

    For v2 we treat titles_enriched.csv (built from wb_seed_titles.csv + TMDb)
    as the single source of truth, instead of merging with titles_catalog.
    """
    df = pd.read_csv(TITLES_ENRICHED_PATH)

    # Ensure basic columns exist
    if "title" not in df.columns:
        raise ValueError("titles_enriched.csv must have a 'title' column")
    if "current_platform" not in df.columns:
        # default to empty if missing, but you really want the TMDb-driven value
        df["current_platform"] = ""
    if "franchise_group" not in df.columns:
        df["franchise_group"] = "Standalone"

    # Normalize column types
    df["title"] = df["title"].astype(str)
    df["current_platform"] = df["current_platform"].astype(str).fillna("")
    df["franchise_group"] = df["franchise_group"].astype(str).fillna("Standalone")

    # Ensure value_score_norm exists
    if "value_score_norm" not in df.columns:
        if "value_score_raw" in df.columns:
            raw = df["value_score_raw"].astype(float)
            df["value_score_norm"] = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        else:
            # fallback: everyone equal weight
            df["value_score_norm"] = 1.0 / max(len(df), 1)

    # Sort by modeled value
    df_sorted = df.sort_values("value_score_norm", ascending=False).reset_index(drop=True)
    return df_sorted


def _canonical_platform(name: str) -> str:
    """Map messy platform strings to a small set of platform labels."""
    if not isinstance(name, str):
        return "Other/Unknown"
    n = name.lower()
    if "netflix" in n:
        return "Netflix"
    if "max" in n or "hbo" in n:
        return "Max"
    if "prime" in n or "amazon" in n:
        return "Prime Video"
    if "hulu" in n:
        return "Hulu"
    if "disney" in n:
        return "Disney+"
    return name.strip() or "Other/Unknown"


def build_platform_exposure_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take titles_df and return a row-per-(title, platform) table with
    a value_share column that splits value_score_norm across platforms.

    Example:
      current_platform = 'Netflix/Max', value_score_norm = 10
      -> Netflix: 5, Max: 5
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["title", "platform", "value_share"])

    df = df.copy()
    df["current_platform"] = df["current_platform"].fillna("Other/Unknown").astype(str)

    rows = []
    for _, row in df.iterrows():
        raw = row["current_platform"]
        # split on / , & etc.
        parts = re.split(r"[\/,&]", raw)
        platforms = [_canonical_platform(p) for p in parts if p.strip()]
        if not platforms:
            platforms = ["Other/Unknown"]

        value = row.get("value_score_norm", 1.0)
        share = value / len(platforms)

        for p in platforms:
            rows.append(
                {
                    "title": row.get("title", ""),
                    "platform": p,
                    "value_share": share,
                }
            )

    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def compute_summary_metrics(df: pd.DataFrame) -> dict:
    """
    High-level metrics for the hero tiles.

    Uses the same shared-window exposure model as the platform chart:
    multi-platform titles split their value across the services they appear on.
    """
    if df is None or df.empty:
        empty_plat = pd.DataFrame(columns=["platform", "value", "share", "value_share"])
        return {
            # old keys (still used in some places)
            "total_titles": 0,
            "outside_pct": 0.0,
            "top_franchise": "N/A",
            "top_platform": "N/A",
            # new / nicer keys
            "share_outside_netflix": 0.0,      # 0–1 for % formatting
            "platform_exposure": empty_plat,   # for charts
        }

    df = df.copy()

    # How many unique WB titles are we modeling?
    if "title" in df.columns:
        total_titles = int(df["title"].nunique())
    else:
        total_titles = int(len(df))

    # ---- Platform exposure using shared-window model ----
    plat_exposure = build_platform_exposure_df(df)  # uses value_score_norm

    if plat_exposure.empty:
        outside_share = 0.0
        outside_pct = 0.0
        top_platform = "N/A"
    else:
        total_value = plat_exposure["value_share"].sum()
        outside_value = plat_exposure.loc[
            plat_exposure["platform"] != "Netflix", "value_share"
        ].sum()

        outside_share = (outside_value / total_value) if total_value > 0 else 0.0
        outside_pct = outside_share * 100.0

        non_nf = plat_exposure[plat_exposure["platform"] != "Netflix"]
        if non_nf.empty:
            top_platform = "None"
        else:
            plat_agg = (
                non_nf.groupby("platform")["value_share"]
                .sum()
                .reset_index()
                .sort_values("value_share", ascending=False)
            )
            top_platform = str(plat_agg.iloc[0]["platform"])

    # ---- Top franchise by modeled value ----
    if "franchise_group" in df.columns:
        fr_col = "franchise_group"
    elif "franchise" in df.columns:
        fr_col = "franchise"
    else:
        fr_col = None

    if fr_col is None:
        top_franchise = "Not labeled yet"
    else:
        fr_agg = (
            df.groupby(fr_col)["value_score_norm"]
            .sum()
            .reset_index()
            .sort_values("value_score_norm", ascending=False)
        )
        top_franchise = str(fr_agg.iloc[0][fr_col])

    return {
        # old keys (for any legacy references)
        "total_titles": total_titles,
        "outside_pct": outside_pct,              # 0–100
        "top_franchise": top_franchise,
        "top_platform": top_platform,
        # new / nicer keys used by tiles & donut
        "share_outside_netflix": outside_share,  # 0–1, use with :.0%
        "platform_exposure": plat_exposure,
    }


# --- Franchise normalizer for seed data ---------------------------------------
def normalize_franchise(raw: str) -> str:
    """Map messy franchise labels from wb_seed_titles.csv into clean buckets."""
    if not isinstance(raw, str) or not raw.strip():
        return "Other WB IP"

    s = raw.strip()

    # Thrones / Westeros universe
    if any(k in s for k in ["Game of Thrones", "Song of Ice and Fire", "Westeros"]):
        return "Game of Thrones / Westeros"

    # Big Bang universe
    if "Big Bang Theory" in s:
        return "The Big Bang Theory Franchise"

    # Wizarding World
    if any(k in s for k in ["Harry Potter", "Wizarding World", "Fantastic Beasts"]):
        return "Wizarding World / Harry Potter"

    # DC + Arrowverse + Gotham/Metropolis heroes
    if any(
        k in s
        for k in [
            "Arrowverse",
            "DC Extended Universe",
            "DC Movies",
            "Batman",
            "Superman",
            "Justice League",
            "Wonder Woman",
            "Gotham",
        ]
    ):
        return "DC / Gotham / Metropolis"

    # Sopranos world
    if "Sopranos" in s:
        return "The Sopranos Franchise"

    # Friends
    if "Friends" in s:
        return "Friends"

    # Fallback: generic bucket for remaining WB titles
    return "Other WB IP"



def build_platform_exposure_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a tidy dataframe with modeled WB value on non-Netflix platforms.

    Columns: platform, value, share, value_share
    """
    base = df.copy()

    # Clean up platform labels a bit
    plat_raw = base["current_platform"].fillna("Unknown")

    plat_clean = (
        plat_raw
        .str.replace("HBO Max", "Max", regex=False)
        .str.replace("HBOMax", "Max", regex=False)
        .str.replace("Netflix / Max", "Netflix + Max", regex=False)
        .str.strip()
    )

    base["platform_clean"] = plat_clean

    # Keep only non-Netflix platforms for this view
    non_nf = base[~base["platform_clean"].str.contains("Netflix", case=False, na=False)]

    if non_nf.empty:
        return pd.DataFrame(columns=["platform", "value", "share", "value_share"])

    plat_exposure = (
        non_nf
        .groupby("platform_clean")["value_score_norm"]
        .sum()
        .reset_index()
        .rename(columns={"platform_clean": "platform", "value_score_norm": "value"})
    )

    total = plat_exposure["value"].sum()
    plat_exposure["share"] = (
        plat_exposure["value"] / total if total > 0 else 0.0
    )
    # keep a separate column so older code using "value_share" still works
    plat_exposure["value_share"] = plat_exposure["share"]

    return plat_exposure




# Load data once and compute summary
titles_df = load_titles()
# ------------------------------------------
# Normalize title column across datasets
# ------------------------------------------
possible_title_cols = ["title", "Title", "name", "Name", "show", "movie"]

found_title = None
for col in possible_title_cols:
    if col in titles_df.columns:
        found_title = col
        break

if found_title:
    titles_df = titles_df.rename(columns={found_title: "title"})
else:
    titles_df["title"] = ""  # fallback if nothing found

if "franchise_group" in titles_df.columns:
    titles_df["franchise_group"] = titles_df["franchise_group"].apply(normalize_franchise)



# --- Helper: representative poster per franchise (for hover) ---
poster_lookup = {}
if (
    isinstance(titles_df, pd.DataFrame)
    and "franchise_group" in titles_df.columns
    and "tmdb_poster_path" in titles_df.columns
):
    poster_lookup = (
        titles_df.dropna(subset=["franchise_group", "tmdb_poster_path"])
        .groupby("franchise_group")["tmdb_poster_path"]
        .first()
        .to_dict()
    )

TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w185"


titles_sorted = titles_df  # backwards-compat with older code
SUMMARY = compute_summary_metrics(titles_df)
PLATFORM_EXPOSURE = build_platform_exposure_df(titles_df)

# --- Ensure risk_top is available for the AI context -------------------------
try:
    _risk_df = titles_df.copy()

    # Compute base risk using your scenario multiplier logic
    _risk_df["risk_score"] = _risk_df["value_score_norm"] * (
        1.0 if scenario == "Base case"
        else 0.6 if scenario == "Conservative"
        else 1.4
    )

    # Pick the columns we need
    cols = [
        c for c in _risk_df.columns
        if c in ("title", "franchise_group", "current_platform", "risk_score")
    ]

    risk_top = (
        _risk_df[cols]
        .sort_values("risk_score", ascending=False)
        .head(6)
        .reset_index(drop=True)
    )

except Exception as e:
    print("Risk model fallback:", e)
    risk_top = None


# -------------------------------
# Hero + headline metrics
# -------------------------------

if titles_df is None or titles_df.empty:
    st.error("Catalog data could not be loaded. Check the CSVs in the data/ folder.")
    st.stop()

total_titles = SUMMARY["total_titles"]
outside_pct = SUMMARY["outside_pct"]
top_franchise = SUMMARY["top_franchise"]
top_platform = SUMMARY["top_platform"]

# ---- Hero title & version ----
st.markdown(
    """
    <div style='text-align:center; margin-top:0.5rem; margin-bottom:0.2rem;'>
      <span style='font-size:3.8rem; font-weight:800; color:#E50914; '>Netflix</span>
      <span style='font-size:3.8rem; font-weight:800; color:#DDDDDD; '>&nbsp;/</span>
      <span style='font-size:3.8rem; font-weight:800; color:#FFC72C; '>&nbsp;Warner&nbsp;Bros</span>
    </div>
    <div style='text-align:center; font-size:2.0rem; color:#BBBBBB; margin-bottom:0.8rem;'>
      Deal Impact Lab • A.I Analyst • <span style='font-weight:600;'>Media Merger Analysis v2</span>
    </div>
    """,
    unsafe_allow_html=True,
)

log_usage("page_load")

# # ---- One-line fallout insight ----
# st.markdown(
#     f"""
#     <div style='text-align:center; max-width:900px; margin:0 auto 1.3rem auto; font-size:0.98rem; color:#DDDDDD;'>
#       <strong>Fallout at a glance:</strong> roughly <strong>{outside_pct:0.0f}%</strong> of the modeled WB content value in this sample
#       currently lives on <em>non-Netflix platforms</em>. If Netflix consolidates the WB library, those services take the first hit.
#     </div>
#     """,
#     unsafe_allow_html=True,
# )
# --- Narrative Hook ---
st.markdown(
    """
    <div style='font-size:1.15rem; font-weight:500; margin-top:-10px; margin-bottom:25px; opacity:0.9;'>
        <span style='color:#ffffff;'>If Netflix takes Warner Bros,</span>
        <b>who bleeds first?</b><br>
        This Fallout Simulator models which <b>platforms</b> and <b>franchises</b> feel the pain first 
        under Conservative / Base / Aggressive pull-in scenarios.
    </div>
    """,
    unsafe_allow_html=True,
)

# st.markdown(
#     """
# <div style='margin-top:10px; padding:10px; border-left: 3px solid #E50914;'>
# <strong>AI Deal Analyst available:</strong>  
# You can ask questions about the Netflix × Warner Bros deal — scenario impacts, platform exposure, 
# and title-level dynamics. Responses are generated using the dashboard’s real data and selected assumptions.
# </div>
# """,
#     unsafe_allow_html=True,
# )

# ---- AI DEAL ANALYST — NETFLIX × WB CO-PILOT ----
#st.markdown("---")
# st.markdown(
#     """
# <div style='padding:14px 16px; border-radius:12px; border:1px solid rgba(255,255,255,0.08);
#             background:rgba(10,10,10,0.75); margin-bottom:12px;'>
#   <div style='font-size:1.75rem; font-weight:600; margin-bottom:4px;'>
#     AI Deal Analyst — ask how this Netflix–Warner Bros scenario plays out
#   </div>
#   <div style='font-size:0.85rem; opacity:0.85; margin-bottom:6px;'>
#     Answers are grounded in this dashboard’s live data: franchise value, platform exposure,
#     your scenario setting, and the modeled WB catalog sample.
#   </div>
# </div>
# """,
#     unsafe_allow_html=True,
# )

st.markdown(
    """
<style>
/* 1. Global Injection for High-Contrast Text */
.data-analyst-box p {
    font-family: monospace, sans-serif; /* Monospace for a terminal feel */
}
/* 2. Style for the accent color used on "AI Deal Analyst" */
.accent-title {
    color: #00FF7F; /* Bright, high-contrast Financial Green */
    font-size: 1.25rem !important; 
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
/* 3. Style for the main description text */
.sub-text {
    font-size: 0.95rem;
    color: #AAAAAA; /* Softer white for secondary information */
    opacity: 0.85;
}
/* 4. Style for the grounded data text */
.grounded-data-text {
    color: #4CAF50; /* A slightly darker green for the emphasis text */
    font-weight: 500;
}
</style>

<div class='data-analyst-box' style='
    padding: 16px; 
    border-radius: 6px; 
    border-left: 5px solid #00FF7F; /* Prominent left border (Financial accent) */
    background: #1C1C1C; /* Very dark charcoal background */
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4); /* Subtle drop shadow for depth */
    margin-bottom: 20px;
'>
  <div class='accent-title' style='margin-bottom: 8px;'>
    AI Deal Analyst — ask how this Netflix–Warner Bros scenario plays out
  </div>
  <div class='sub-text' style='margin-bottom: 4px;'>
    Answers are grounded in this dashboard’s <span class='grounded-data-text'>live data</span>: franchise value, platform exposure,
    your scenario setting, and the modeled WB catalog sample.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

risk_top = globals().get("risk_top", None)

import json

def build_context_for_llm(
    titles_df: pd.DataFrame,
    platform_exposure: pd.DataFrame | None,
    risk_top: pd.DataFrame | None,
    scenario: str,
    hero_summary: dict,
) -> str:
    """
    Build a compact, JSON-serializable context object for the AI analyst.

    We ONLY send small slices of data, and everything is converted to
    primitive Python types (lists/dicts/floats/strings).
    """

    # --- Titles slice: top 25 by modeled value ---
    if titles_df is not None and not titles_df.empty:
        cols = [
            c
            for c in titles_df.columns
            if c in ("title", "franchise_group", "current_platform", "value_score_norm")
        ]
        titles_slice = (
            titles_df[cols]
            .sort_values("value_score_norm", ascending=False)
            .head(25)
            .reset_index(drop=True)
            .to_dict(orient="records")
        )
    else:
        titles_slice = []

    # --- Platform exposure slice for donut chart data ---
    if platform_exposure is not None and not platform_exposure.empty:
        plat_slice = (
            platform_exposure[["platform", "value", "share"]]
            .reset_index(drop=True)
            .to_dict(orient="records")
        )
    else:
        plat_slice = []

    # --- Top risk titles slice ---
    if risk_top is not None and not isinstance(risk_top, list):
        try:
            risk_slice = risk_top.reset_index(drop=True).to_dict(orient="records")
        except Exception:
            risk_slice = []
    else:
        risk_slice = risk_top or []

    # --- Hero summary already a dict (from compute_summary_metrics) ---
    hero_safe = {
        "total_titles": float(hero_summary.get("total_titles", 0)),
        "outside_pct": float(hero_summary.get("outside_pct", 0.0)),
        "top_franchise": str(hero_summary.get("top_franchise", "")),
        "top_platform": str(hero_summary.get("top_platform", "")),
    }

    ctx = {
        "scenario": scenario,
        "hero_summary": hero_safe,
        "titles_sample": titles_slice,
        "platform_exposure": plat_slice,
        "top_risk_titles": risk_slice,
    }

    # Make absolutely sure everything is serializable
    return json.dumps(ctx, indent=2, default=str)


scenario = st.radio(
    r"$\textsf{ How aggressively do you think Netflix will pull WB content in-house?}$",
    options=["Conservative", "Base case", "Aggressive"],
    index=1,
    horizontal=True,
    help=(
        "Conservative: only the most exposed titles move.\n"
        "Base case: measured consolidation of high-value WB tentpoles.\n"
        "Aggressive: deep consolidation where Netflix pulls most WB content in-house."
    ),
)
log_usage("scenario_change", scenario=scenario)

user_q = st.text_input(
    r"$\textsf{\Large Question to the analyst}$",
    placeholder="e.g., Which platforms lose the most leverage if Netflix pulls the top HBO dramas in-house?",
    key="ai_question",
)



if user_q:
    if not _openai_ok:
        st.warning(
            "The AI Deal Analyst is temporarily unavailable on this deployment. "
            "Check that OPENAI_API_KEY is set and the OpenAI client is installed."
        )
    else:
        with st.spinner("Analyzing using your scenario and WB exposure model..."):
            try:
                context_json = build_context_for_llm(
                    titles_df=titles_df,
                    platform_exposure=PLATFORM_EXPOSURE,
                    risk_top=risk_top,
                    scenario=scenario,
                    hero_summary=SUMMARY,   # your hero metrics dict
                )

                sys_prompt = f"""
You are an elite senior media strategy analyst specializing in mergers, content economics,
and platform exposure. Use ONLY the structured data below to answer questions.

Here is the structured context (JSON):
{context_json}

Rules:
- Never hallucinate titles that are not in the data.
- Always tie answers to platforms, franchises, and relative exposure.
- Treat values as directional, not precise forecasts.
"""

                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_q},
                    ],
                    max_tokens=350,
                )

                answer = completion.choices[0].message.content
                st.markdown(f"### AI Analyst Response\n{answer}")
                #st.markdown(answer)

            except Exception as e:
                st.error(f"AI unavailable. Falling back to basic answers.\n\nError: {e}")


# ---- Metric tiles row ----
c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_tile(
        "WB titles modeled",
        f"{total_titles}",
        "Number of WB series & films included in this scenario sample.",
    )

with c2:
    metric_tile("WB VALUE ON NON-NETFLIX PLATFORMS",
                f"{outside_pct}",
        "Share of modeled WB content value that, in this sample"
        "other services (Max, Hulu, Prime etc.)"
    )

with c3:
    metric_tile(
        "Top leverage franchise",
        top_franchise or "—",
        "Franchise with the highest modeled Warner Bros value in this sample.",
    )

with c4:
    metric_tile(
        "Most exposed platform",
        top_platform or "—",
        "Non-Netflix platform with the most WB value at risk if Netflix consolidates WB.",
    )

# -------------------------------
# Quick Fallout Snapshot + Franchises
# -------------------------------

#section_heading("Quick Fallout Snapshot")


col_left, col_right = st.columns([1.2, 1.8])

with col_left:
    st.markdown("## Quick Fallout Snapshot")

    # --- Sharper Quick Snapshot ---
    st.markdown(
        """
        <div style='font-size:1rem; line-height:1.6; margin-top:-10px;'>
        
        Most merger coverage focuses on debt, regulators, and Hollywood politics.<br>
        <b>This tool focuses on one question:</b>  
        <span style='font-size:1.05rem;'><b>Which platforms and franchises actually feel it first?</b></span>

        <ul style='margin-top:10px;'>
            <li><b>Which WB franchises drive real leverage</b> (not just fan sentiment)</li>
            <li><b>Which rival platforms lose the most value first</b> if Netflix pulls WB in-house</li>
            <li><b>How the fallout shifts</b> across Conservative / Base / Aggressive scenarios</li>
        </ul>

        <i style='opacity:0.8;'>Use this as a content-impact lens for decks, memos, or reporting.</i>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
    "<div class='hero-link'><a href='#sources-section'>Jump to deal coverage & risk →</a></div>",
    unsafe_allow_html=True,
    )


# with col_right:
#     PLATFORM_EXPOSURE = SUMMARY["PLATFORM_EXPOSURE"]  # from compute_summary_metrics

#     labels = PLATFORM_EXPOSURE["platform"].tolist()
#     sizes = PLATFORM_EXPOSURE["share"].tolist()

#     fig, ax = plt.subplots(figsize=(3.3, 3.3))
#     wedges, _ = ax.pie(
#         sizes,
#         labels=labels,
#         startangle=90,
#         wedgeprops=dict(width=0.35, edgecolor="#0E1117"),
#     )
#     ax.axis("equal")

#     st.pyplot(fig)  # no use_container_width → respects figsize

# st.caption(
#     f"In this sample, {PLATFORM_EXPOSURE.iloc[0]['platform']} carries about "
#     f"{PLATFORM_EXPOSURE.iloc[0]['share']:.0%} of the modeled WB value that sits outside Netflix. "
#     "As additional platforms are tagged in the dataset, this view will split across more services."
# )

st.markdown("<div id='sources-section'></div>", unsafe_allow_html=True)
with col_right:
    st.markdown("#### Rival platforms with WB value at risk")

    if PLATFORM_EXPOSURE.empty:
        st.info("Not enough non-Netflix platform data in this sample yet.")
    else:
        # Brand-ish colors for the main streaming platforms
        brand_colors = {
            "Max": "#8B48DD",          # Max purple
            "Hulu": "#1CE783",         # Hulu green
            "Prime Video": "#00A8E1",  # Prime Video cyan
            "Peacock": "#F5C518",      # Peacock yellow
            "Disney+": "#113CCF",      # Disney+ blue
            "Rent/Buy": "#6C757D",     # neutral gray for TVOD / rent-buy
        }

        # Build a color map with a sensible default for anything not listed above
        default_color = "#9E86FF"  # fallback WB purple
        color_map = {
            plat: brand_colors.get(plat, default_color)
            for plat in PLATFORM_EXPOSURE["platform"].unique()
        }

        platform_fig = px.pie(
            PLATFORM_EXPOSURE,
            names="platform",
            values="value",
            hole=0.45,
            color="platform",
            color_discrete_map=color_map,
        )

        platform_fig.update_traces(
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Modeled WB value at risk: %{value:.2f}<extra></extra>",
        )
        platform_fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=20, b=0),
        )

        st.plotly_chart(platform_fig, use_container_width=True)

        top_row = PLATFORM_EXPOSURE.sort_values("share", ascending=False).iloc[0]
        st.caption(
            f"In this sample, **{top_row['platform']}** carries about "
            f"**{top_row['share']*100:.1f}%** of the modeled WB value that sits outside Netflix."
        )

        # top_row = PLATFORM_EXPOSURE.sort_values("share", ascending=False).iloc[0]
        # st.caption(
        #     f"In this sample, **{top_row['platform']}** carries about "
        #     f"**{top_row['share']*100:.1f}%** of the modeled WB value that sits outside Netflix."
        # )

# --- Top 5 Fallout Findings strip --------------------------------------------
shock_findings = build_shock_findings(titles_df, SUMMARY, PLATFORM_EXPOSURE, scenario)

if shock_findings:
    st.markdown("## Top 5 fallout findings from this scenario")

    # Glassy card container
    st.markdown(
        "<div style='padding:10px 12px; border-radius:12px; "
        "background:rgba(10,10,20,0.85); border:1px solid rgba(255,255,255,0.08);'>",
        unsafe_allow_html=True,
    )

    for i, line in enumerate(shock_findings, start=1):
        st.markdown(f"- {line}")

    st.markdown("</div>", unsafe_allow_html=True)


section_heading("Top WB franchises by modeled value")

# Franchise Value Stack: who matters most?
fr_df = titles_df.copy()

if isinstance(fr_df, pd.DataFrame) and "franchise_group" in fr_df.columns:
    fr_col_name = "franchise_group"
else:
    fr_col_name = None

if fr_col_name is None or "value_score_norm" not in fr_df.columns:
    st.info("Franchise metadata not available in this sample yet.")
else:
    # Clean up franchise labels
    fr_df[fr_col_name] = (
        fr_df[fr_col_name]
        .fillna("Standalone / other")
        .astype(str)
    )

    # Aggregate modeled value by franchise
    fr_agg = (
        fr_df.groupby(fr_col_name)["value_score_norm"]
        .sum()
        .reset_index()
        .sort_values("value_score_norm", ascending=False)
    )

    if fr_agg.empty:
        st.info("No franchise-level value data to display yet.")
    else:
        TOP_N = 8
        top_franchises = fr_agg.head(TOP_N).copy()

        # Aggregate the rest into an "All other WB IP" bucket (kept in data, not plotted)
        others_value = fr_agg["value_score_norm"].iloc[TOP_N:].sum()
        if others_value > 0:
            top_franchises = pd.concat(
                [
                    top_franchises,
                    pd.DataFrame(
                        [
                            {
                                fr_col_name: "All other WB IP",
                                "value_score_norm": others_value,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

        fr_chart_df = top_franchises.rename(
            columns={fr_col_name: "Franchise", "value_score_norm": "Modeled value"}
        ).copy()
        
        plot_df = fr_chart_df[fr_chart_df["Franchise"] != "All other WB IP"].copy()

        # Compute percentage share
        total_val = fr_chart_df["Modeled value"].sum()
        fr_chart_df["pct"] = fr_chart_df["Modeled value"] / total_val
        fr_chart_df["pct_label"] = (fr_chart_df["pct"] * 100).round(1).astype(str) + "%"

        # Attach a poster URL if we have one
        def _poster_for(fr_name: str) -> str:
            path = poster_lookup.get(fr_name)
            if not path:
                return ""
            return TMDB_POSTER_BASE + path

        fr_chart_df["poster_url"] = fr_chart_df["Franchise"].map(_poster_for)

        # Clean up franchise labels using normalizer and generic bucket
        fr_df[fr_col_name] = fr_df[fr_col_name].apply(normalize_franchise)
        fr_df[fr_col_name] = (
            fr_df[fr_col_name]
            .fillna("Other WB IP")
            .astype(str)
        )


        plot_df = fr_chart_df[
            ~fr_chart_df["Franchise"].isin(["All other WB IP", "Other WB IP"])
        ].copy()


        # Multi-color bars + poster link in hover
        fig_fr = px.bar(
            plot_df,
            x="pct",
            y="Franchise",
            orientation="h",
            text="pct_label",
            color="Franchise",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            custom_data=["poster_url"],
        )

        fig_fr.update_traces(
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "% share of modeled WB value: %{x:.1%}<br>"
                "<span style='font-size:0.8rem;'>"
                "Poster: %{customdata}"
                "</span>"
                "<extra></extra>"
            ),
        )

        fig_fr.update_layout(
            xaxis_title="% of modeled WB value",
            yaxis_title="Franchise / IP group",
            template="plotly_dark",
            margin=dict(l=0, r=20, t=10, b=10),
            height=360,
            showlegend=False,
        )

        # Percent ticks on x-axis
        fig_fr.update_xaxes(
            tickformat="0%",
            range=[0, plot_df["pct"].max() * 1.1],
        )
        
        st.markdown(
            f"""
            <div style='font-size:1.05rem; font-weight:600; margin-bottom:5px;'>
                Rival platforms with WB value at risk
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style='font-size:1.05rem; font-weight:600; margin-top:20px; margin-bottom:5px;'>
                Franchise leverage: who holds the real power in this content universe?
            </div>
            """,
            unsafe_allow_html=True,
        )


        st.plotly_chart(fig_fr, use_container_width=True)

        st.markdown(
            "<div style='font-size:0.75rem; opacity:0.7; margin-top:-8px;'>"
            "% share of modeled Warner Bros content value in this sample. "
            "Hover to see a TMDB poster link for each franchise."
            "</div>",
            unsafe_allow_html=True,
        )


# # Franchise Value Stack: who matters most?
# fr_df = titles_df.copy()
# if "franchise_group" in fr_df.columns:
#     fr_col_name = "franchise_group"
# else:
#     fr_col_name = None

# if fr_col_name is None:
#     st.info("Franchise metadata not available in this sample yet.")
# else:
#     fr_df[fr_col_name] = (
#         fr_df[fr_col_name]
#         .fillna("Standalone / other")
#         .astype(str)
#     )

#     fr_agg = (
#         fr_df.groupby(fr_col_name)["value_score_norm"]
#         .sum()
#         .reset_index()
#         .sort_values("value_score_norm", ascending=False)
#     )

#     TOP_N = 8
#     top_franchises = fr_agg.head(TOP_N).copy()

#     others_value = fr_agg["value_score_norm"].iloc[TOP_N:].sum()
#     if others_value > 0:
#         top_franchises = pd.concat(
#             [
#                 top_franchises,
#                 pd.DataFrame(
#                     [
#                         {
#                             fr_col_name: "All other WB IP",
#                             "value_score_norm": others_value,
#                         }
#                     ]
#                 ),
#             ],
#             ignore_index=True,
#         )

#     fr_chart_df = top_franchises.rename(
#         columns={fr_col_name: "Franchise", "value_score_norm": "Modeled value"}
#     ).copy()
#     fr_chart_df["value_label"] = fr_chart_df["Modeled value"].round(1)

#     fig_fr = px.bar(
#         fr_chart_df,
#         x="Modeled value",
#         y="Franchise",
#         orientation="h",
#         text="value_label",
#     )
#     fig_fr.update_traces(
#         marker_color="#9E86FF",  # HBOMAX
#         hovertemplate="<b>%{y}</b><br>Modeled value: %{x:.1f}<extra></extra>",
#         textposition="outside",
#     )
#     fig_fr.update_layout(
#         xaxis_title="Modeled WB value (relative units)",
#         yaxis_title="Franchise / IP group",
#         template="plotly_dark",
#         margin=dict(l=0, r=20, t=10, b=10),
#         height=360,
#     )

#     st.plotly_chart(fig_fr, use_container_width=True)

#     st.caption(
#         "Ranked view of WB franchises in this sample by modeled content value. "
#         "It shows which IP clusters matter most if Netflix consolidates WB content."
#     )

# -------------------------------
# Scenario assumptions + Platform Exposure Risk
# -------------------------------

# --- Fallout Dial ---
# -------------------------------
# Consumer Impact Card
# -------------------------------

st.markdown("## How this affects **you** as a streaming subscriber")

exp_df = build_platform_exposure_df(titles_df)
non_nf = exp_df[exp_df["platform"] != "Netflix"]

max_share = (
    non_nf[non_nf["platform"].str.contains("Max", case=False, na=False)]["value_share"].sum()
    / non_nf["value_share"].sum() * 100
    if not non_nf.empty else 0
)

prime_share = (
    non_nf[non_nf["platform"].str.contains("Prime", case=False, na=False)]["value_share"].sum()
    / non_nf["value_share"].sum() * 100
    if not non_nf.empty else 0
)

hulu_share = (
    non_nf[non_nf["platform"].str.contains("Hulu", case=False, na=False)]["value_share"].sum()
    / non_nf["value_share"].sum() * 100
    if not non_nf.empty else 0
)

st.markdown(
    f"""
<div style='padding:14px; border-radius:10px; background-color:#111111; border:1px solid #333;'>
<b>If you're a Max subscriber:</b>  
Max carries <b>{max_share:.1f}%</b> of WB value outside Netflix today.  
Under the <b>{scenario}</b> scenario, this is the platform that feels it first if Netflix consolidates WB content.
<br><br>
<b>If you're a Prime Video viewer:</b>  
Prime carries ~<b>{prime_share:.1f}%</b> of external WB value. Mostly catalog titles — low tentpole impact.
<br><br>
<b>If you're on Hulu:</b>  
Hulu carries ~<b>{hulu_share:.1f}%</b> of WB value. Exposure is long-tail rather than prestige draw.
<br><br>
<b>If you're a Netflix user:</b>  
Under <b>{scenario}</b>, Netflix increasingly becomes the home of WB's prestige franchises (Harry Potter, GoT, DC, etc.).
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------------
# Shareable takeaway card
# -------------------------------

st.markdown("## One-slide takeaway deck")
#section_heading("One-slide takeaway deck")

# Reuse exposure model to estimate how much WB value could consolidate into Netflix
exp_df_takeaway = build_platform_exposure_df(titles_df)
total_value_takeaway = exp_df_takeaway["value_share"].sum()

non_nf_takeaway = exp_df_takeaway[exp_df_takeaway["platform"] != "Netflix"]
rivals_value = non_nf_takeaway["value_share"].sum()

# If Netflix ultimately pulls most WB content in-house, this rivals_value is what gets redistributed
netflix_gain_pct = (rivals_value / total_value_takeaway * 100.0) if total_value_takeaway > 0 else 0.0

# Top 2 WB franchises by modeled value
fr_df_takeaway = titles_df.copy()
if "franchise_group" in fr_df_takeaway.columns:
    fr_df_takeaway["franchise_group"] = (
        fr_df_takeaway["franchise_group"].fillna("Standalone / other").astype(str)
    )
    fr_agg_takeaway = (
        fr_df_takeaway.groupby("franchise_group")["value_score_norm"]
        .sum()
        .sort_values(ascending=False)
    )
    top_franchises_list = list(fr_agg_takeaway.index[:2])
else:
    top_franchises_list = []

scenario_blurb = {
    "Conservative": "light consolidation of only the most exposed WB titles",
    "Base case": "measured consolidation of WB’s highest-value tentpoles",
    "Aggressive": "deep consolidation where Netflix pulls most WB content in-house",
}.get(scenario, "measured consolidation of key WB content")

top_fr_1 = top_franchises_list[0] if len(top_franchises_list) > 0 else "top WB franchises"
top_fr_2 = top_franchises_list[1] if len(top_franchises_list) > 1 else ""

takeaway_line_2 = (
    f"{top_fr_1}"
    if not top_fr_2
    else f"{top_fr_1} and {top_fr_2}"
)

st.markdown(
    f"""
<div style='padding:16px; border-radius:14px; background-color:#050816; border:1px solid #27272f;'>
  <div style='font-size:0.8rem; text-transform:uppercase; letter-spacing:.12em; color:#9CA3AF; margin-bottom:4px;'>
    Netflix / WB Fallout Simulator • Snapshot
  </div>
  <div style='font-size:1.1rem; font-weight:600; margin-bottom:6px;'>
    Under the <span style='font-weight:700;'>{scenario}</span> scenario ({scenario_blurb}),
    Netflix could effectively absorb around <span style='font-weight:700;'>{netflix_gain_pct:.1f}%</span>
    of the modeled WB content value currently sitting on rival platforms in this sample.
  </div>
  <div style='font-size:0.95rem; color:#D1D5DB;'>
    The leverage comes from franchises like <b>{takeaway_line_2}</b>, which dominate the modeled value pool
    and shift perception of where prestige WB storytelling lives.
  </div>
  <div style='font-size:0.8rem; color:#9CA3AF; margin-top:8px;'>
    Screenshot this card for decks, memos, or social posts – it’s generated directly from the scenario,
    franchise scores, and platform exposure model used above.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<<<<<<< HEAD
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
"""
=======
    <div style='font-size:1.1rem; font-weight:600; margin-top:25px;'>
        Fallout Dial: How aggressively does Netflix pull WB in-house?
    </div>
    """,
    unsafe_allow_html=True,
>>>>>>> 7dbf3bd (Update: WB sample expanded to 173 + AI Analyst + fallout simulator visuals)
)

scenario = st.radio(
    "Select a fallout mode:",
    options=["Conservative", "Base case", "Aggressive"],
    index=1,
    horizontal=True,
)



st.caption(
    "This setting scales how much WB content you assume Netflix ultimately consolidates. "
    "It only affects the exposure views below — not the raw catalog numbers above."
)

scenario_multiplier = {
    "Conservative": 0.6,
    "Base case": 1.0,
    "Aggressive": 1.4,
}[scenario]


def compute_risk_score(row: pd.Series) -> float:
    """
    Simple content-risk heuristic:

    - Max (HBO / WBD-owned)  : lowest incremental pain (already in the WB family)
    - Netflix                : medium (partially consolidated already)
    - Everyone else          : highest — they stand to lose licensed WB value
    """
    platform = str(row.get("current_platform", ""))
    value = float(row.get("value_score_norm", 0.0))

    if "max" in platform.lower() or "hbo" in platform.lower():
        return value * 0.2  # low incremental disruption
    if "netflix" in platform.lower():
        return value * 0.4  # medium exposure
    return value * 1.0      # highest exposure for third-party platforms


# Build a scenario-adjusted risk view per title
risk_df = titles_df.copy()
risk_df["current_platform"] = risk_df["current_platform"].astype(str)

risk_df["risk_score_base"] = risk_df.apply(compute_risk_score, axis=1)
risk_df["Risk Score"] = risk_df["risk_score_base"] * scenario_multiplier


# Build display-friendly columns safely
risk_display = risk_df.copy()

# Title column: fallback to empty string if not present
if "title" in risk_display.columns:
    risk_display["Title"] = risk_display["title"].astype(str)
elif "Title" in risk_display.columns:
    risk_display["Title"] = risk_display["Title"].astype(str)
else:
    risk_display["Title"] = ""

# Franchise column: prefer franchise_group, then franchise, else placeholder
if "franchise_group" in risk_display.columns:
    risk_display["Franchise"] = risk_display["franchise_group"].astype(str)
elif "franchise" in risk_display.columns:
    risk_display["Franchise"] = risk_display["franchise"].astype(str)
else:
    risk_display["Franchise"] = "Not labeled"

# Currently On: from current_platform
risk_display["Currently On"] = risk_display["current_platform"].astype(str)

cols_for_display = ["Title", "Franchise", "Currently On", "Risk Score"]

# Top titles that most hurt rivals if pulled in-house
risk_top = (
    risk_display[cols_for_display]
    .sort_values("Risk Score", ascending=False)
    .head(6)
)

st.dataframe(
    risk_top,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Risk Score is a simple heuristic: modeled content value × how exposed that title is "
    "to non-Netflix platforms, scaled by the scenario above."
)


def render_deal_sources():
    st.markdown("## Deal coverage & sources")

    st.markdown(
        """
Use this as a one-stop reading list around the Netflix–Warner Bros deal.  
The analytics on this page are meant to **complement, not replace**, these sources.
        """
    )

    st.markdown("#### Official / primary sources")
    st.markdown(
        """
- **Company communications**
  - Netflix IR — *Deal announcement & strategic rationale*  
    - [Press release](https://about.netflix.com/en/news/netflix-to-acquire-warner-bros)  
  - Warner Bros Discovery — *Strategic update on studio & streaming assets*  
    - [Corporate News](https://www.prnewswire.com/in/news-releases/netflix-to-acquire-warner-bros-following-the-separation-of-discovery-global-for-a-total-enterprise-value-of-82-7-billion-equity-value-of-72-0-billion-302633998.html)
        """
    )
#     - **Transaction structure**
#   - Deal terms, consideration mix, and spin-off details  
#     - [Deal summary / prospectus](https://example.com/deal-terms)

    st.markdown("#### Hard news coverage")
    st.markdown(
        """
    - AP News — *Netflix to acquire Warner Bros studio and streaming business for $72 billion*  
    – [AP coverage](https://apnews.com/article/netflix-warner-acquisition-studio-hbo-streaming-f4884402cadfd07a99af0c8e4353bd83) :contentReference[oaicite:2]{index=2}  
    - Reuters — *Instant view: Netflix to buy Warner Bros Discovery's studios, streaming unit*  
    – [Reuters instant view](https://www.reuters.com/legal/transactional/view-netflix-buy-warner-bros-discoverys-studios-streaming-unit-72-billion-2025-12-05/) :contentReference[oaicite:3]{index=3}  
    - Financial Times — *Netflix agrees $83bn takeover of Warner Bros Discovery*  
    – [FT deal write-up](https://www.ft.com/content/6532be94-c0bf-4101-8126-f249aa6be3c5) :contentReference[oaicite:4]{index=4} """)

    st.markdown("#### Financing & market angl")

    st.markdown("""- Bloomberg — *Netflix lines up $59bn of debt for Warner Bros deal*  
    – [Financing piece](https://www.bloomberg.com/news/articles/2025-12-05/netflix-lines-up-59-billion-of-debt-for-warner-bros-deal) :contentReference[oaicite:5]{index=5} """) 

    st.markdown("#### HReaction & impact")

    st.markdown("""- AP News — *Notable early reaction to Netflix’s deal to acquire Warner Bros*  
    – [Industry & political reaction](https://apnews.com/article/netflix-warner-bros-deal-reaction-3acea5d81e630d20560299764bf4c37c) :contentReference[oaicite:6]{index=6}  
    """
    )

render_deal_sources()


st.markdown("---")
st.markdown("### Quick feedback")

st.markdown(
    "<span style='font-size:0.85rem; opacity:0.8;'>"
    "Help me improve this Netflix–WB Fallout Simulator. "
    "This takes less than 10 seconds."
    "</span>",
    unsafe_allow_html=True,
)

with st.form("feedback_form"):
    col_rating, col_comment = st.columns([1, 3])

    with col_rating:
        rating = st.radio(
            "Useful?",
            options=[5, 4, 3, 2, 1],
            format_func=lambda x: {5: "Great", 4: "Good", 3: "Ok", 2: "More Info", 1: "Pineapple on pizza!"}[x],
            horizontal=True,
        )

    with col_comment:
        comment = st.text_input(
            "Optional note (what did you like / what’s missing? / Feature you'd like to see)",
            placeholder="e.g., ‘Good for a quick sanity check on Max exposure.’",
        )

    submitted = st.form_submit_button("Send feedback")

if submitted:
    try:
        current_scenario = SCENARIO if "SCENARIO" in globals() else None
        log_feedback(rating, comment, scenario=current_scenario)
        log_usage("feedback_submitted", scenario=current_scenario)
        st.success("Thanks — your feedback is logged.")
    except Exception as e:
        st.warning(f"Feedback could not be saved: {e}")


st.markdown(
    """
<div style="font-size:0.8rem; opacity:0.65; margin-top:1.5rem;">
Content metadata sourced from <strong>TMDB</strong> (The Movie Database) and manually
curated coverage from outlets like AP, Reuters, Bloomberg, and the Financial Times.
All modeling and scoring is independent and for analysis only.
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------------
# Share link with click-to-copy
# -------------------------------
SHARE_URL = "https://streamshift-aiaptrnvxydq4ka23avbsz.streamlit.app/"

with st.expander("Share this dashboard"):
    st.markdown(
        f"""
        <div style="margin-bottom:8px;">
            Copy and share this link with colleagues or friends:
        </div>
        <div style="display:flex; gap:8px; align-items:center;">
            <input id="share-link-input" type="text"
                   value="{SHARE_URL}"
                   readonly
                   style="flex:1; padding:6px 8px; border-radius:6px;
                          border:1px solid #374151; background:#020617; color:#E5E7EB;
                          font-size:0.85rem;" />
            <button onclick="navigator.clipboard.writeText(document.getElementById('share-link-input').value);
                             alert('Link copied to clipboard!');"
                    style="padding:6px 10px; border-radius:6px; border:none;
                           background:#2563EB; color:white; font-size:0.8rem; cursor:pointer;">
                Copy link
            </button>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown(
    """
<div style='font-size:0.7rem; opacity:0.65; margin-top:24px;'>
  Data sample built from publicly available Warner Bros titles and TMDB metadata 
  (© TMDB & contributors). This tool is a content-impact simulator, not legal or financial advice.
</div>
""",
    unsafe_allow_html=True,
)

