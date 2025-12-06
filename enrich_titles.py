"""
enrich_titles.py

Reads data/titles_catalog.csv,
fetches metadata from TMDb for each title,
computes a simple value_score,
and writes data/titles_enriched.csv.

Run with:
    TMDB_API_KEY=your_key_here python enrich_titles.py
"""

import math
import os
import time
from pathlib import Path
from typing import Dict, Any

import requests
import pandas as pd

CATALOG_PATH = Path("data") / "titles_catalog.csv"
ENRICHED_PATH = Path("data") / "titles_enriched.csv"

TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"


def fetch_tmdb_details(tmdb_id: str, content_type: str) -> Dict[str, Any]:
    """
    Fetch a single title's details from TMDb.
    content_type: 'film' -> /movie, 'series' -> /tv
    """
    if not TMDB_API_KEY:
        print("[error] TMDB_API_KEY is not set inside script")
        return {}

    tmdb_id = str(tmdb_id).strip()
    if not tmdb_id:
        print("[warn] No tmdb_id provided, skipping TMDb fetch")
        return {}

    endpoint_type = "movie" if content_type == "film" else "tv"
    url = f"{TMDB_BASE_URL}/{endpoint_type}/{tmdb_id}"

    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
    except Exception as e:
        print(f"[error] Request error for {endpoint_type} {tmdb_id}: {e}")
        return {}

    if resp.status_code != 200:
        print(f"[warn] TMDb {endpoint_type} {tmdb_id} returned {resp.status_code}: {resp.text[:120]}")
        return {}

    data = resp.json()
    return {
        "tmdb_popularity": data.get("popularity", 0.0),
        "tmdb_vote_count": data.get("vote_count", 0),
        "tmdb_vote_average": data.get("vote_average", 0.0),
        "tmdb_poster_path": data.get("poster_path") or "",
        "tmdb_original_title": data.get("original_title") or data.get("name") or "",
    }


def main() -> None:
    print(f"[info] Using TMDB_API_KEY prefix: {str(TMDB_API_KEY)[:4]}..., length={len(TMDB_API_KEY) if TMDB_API_KEY else 0}")

    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Catalog file not found: {CATALOG_PATH}")

    # Use pandas to read the catalog robustly
    df_catalog = pd.read_csv(CATALOG_PATH)

    # Clean up and ensure required columns exist
    expected_cols = ["title", "type", "franchise_group", "owner", "current_platform", "tmdb_id"]
    for col in expected_cols:
        if col not in df_catalog.columns:
            raise RuntimeError(f"Expected column '{col}' not found in {CATALOG_PATH}")

    # Drop any rows with missing title
    df_catalog["title"] = df_catalog["title"].astype(str)
    df_catalog = df_catalog[df_catalog["title"].str.strip() != ""].copy()

    print(f"[info] Loaded {len(df_catalog)} non-empty title rows from {CATALOG_PATH}")
    if df_catalog.empty:
        raise RuntimeError("No usable rows found in titles_catalog.csv")

    enriched_rows = []

    for idx, row in df_catalog.iterrows():
        title = row["title"]
        content_type = str(row["type"]).strip().lower() or "film"
        tmdb_id = row.get("tmdb_id", "")

        print(f"[{idx + 1}/{len(df_catalog)}] Fetching TMDb for: {title} (type={content_type}, id={tmdb_id})")

        details = fetch_tmdb_details(tmdb_id, content_type)
        print(f"    -> details keys: {list(details.keys())}")

        record = {
            "title": title,
            "type": row["type"],
            "franchise_group": row["franchise_group"],
            "owner": row["owner"],
            "current_platform": row["current_platform"],
            "tmdb_id": tmdb_id,
            **details,
        }
        enriched_rows.append(record)

        time.sleep(0.25)  # Be kind to TMDb

    df = pd.DataFrame(enriched_rows)
    print(f"[info] Built DataFrame with shape {df.shape}")
    print(f"[info] Columns: {df.columns.tolist()}")

    # Ensure expected TMDb columns exist even if some fetches failed
    for col, default, dtype in [
        ("tmdb_popularity", 0.0, float),
        ("tmdb_vote_count", 0, int),
        ("tmdb_vote_average", 0.0, float),
    ]:
        if col not in df.columns:
            print(f"[warn] No '{col}' column found, creating default {default} column")
            df[col] = default
        df[col] = df[col].fillna(default).astype(dtype)

    # Compute a raw value score (simple heuristic)
    df["value_score_raw"] = df.apply(
        lambda r: float(r["tmdb_popularity"]) * math.log(r["tmdb_vote_count"] + 1),
        axis=1,
    )

    # Normalize to 0–100 for easier interpretation
    min_raw = df["value_score_raw"].min()
    max_raw = df["value_score_raw"].max()
    if max_raw > min_raw:
        df["value_score_norm"] = df["value_score_raw"].apply(
            lambda x: 100 * (x - min_raw) / (max_raw - min_raw)
        )
    else:
        df["value_score_norm"] = 0.0

    ENRICHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ENRICHED_PATH, index=False)
    print(f"[done] Wrote enriched titles to {ENRICHED_PATH}")


if __name__ == "__main__":
    main()
