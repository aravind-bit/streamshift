#!/usr/bin/env python
"""
Build titles_enriched.csv from wb_seed_titles.csv using TMDB.

- Reads data/wb_seed_titles.csv
- For each tmdb_id, tries tv then movie (or a seed media_type hint)
- Fetches details + watch providers
- Computes simple value scores
- Writes data/titles_enriched.csv

Run:
    (venv) python tools/build_titles_from_seed.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

BASE_URL = "https://api.themoviedb.org/3"
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")

ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / "data" / "wb_seed_titles.csv"
OUT_PATH = ROOT / "data" / "titles_enriched.csv"


# -------------------------------------------------------------------
# Low-level TMDB helpers
# -------------------------------------------------------------------
def tmdb_request(path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Call TMDB v3 with api_key query param. Returns JSON or None on error."""
    if params is None:
        params = {}

    if not TMDB_API_KEY:
        print("[error] TMDB_API_KEY is not set in the environment.", file=sys.stderr)
        sys.exit(1)

    params = {"api_key": TMDB_API_KEY, **params}
    url = f"{BASE_URL}{path}"

    try:
        resp = requests.get(url, params=params, timeout=10)
    except Exception as e:
        print(f"[warn] TMDB request failed: {url} :: {e}")
        return None

    if resp.status_code == 404:
        print(f"[warn] TMDB {path} -> status 404")
        return None

    if resp.status_code != 200:
        print(f"[warn] TMDB {path} -> status {resp.status_code}")
        return None

    try:
        return resp.json()
    except Exception as e:
        print(f"[warn] TMDB JSON parse error for {path}: {e}")
        return None


def fetch_tmdb_bundle(
    tmdb_id: int,
    media_type_hint: Optional[str] = None,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Try to fetch details + watch providers for this TMDB id.

    Returns:
        (media_type_used, details_json, providers_json)
        media_type_used is 'tv' or 'movie' or None if not found.
    """
    tried: List[str] = []

    candidates: List[str] = []
    if media_type_hint in {"tv", "movie"}:
        candidates.append(media_type_hint)
    candidates += ["tv", "movie"]
    # de-dup while preserving order
    candidates = list(dict.fromkeys(candidates))

    details: Optional[Dict[str, Any]] = None
    media_type_used: Optional[str] = None

    for mtype in candidates:
        tried.append(mtype)
        details = tmdb_request(f"/{mtype}/{tmdb_id}", params={"language": "en-US"})
        if details is not None:
            media_type_used = mtype
            break

    if details is None:
        print(f"[warn] TMDB id {tmdb_id} not found as any of {tried}")
        return None, None, None

    providers = tmdb_request(f"/{media_type_used}/{tmdb_id}/watch/providers")

    return media_type_used, details, providers


# -------------------------------------------------------------------
# Transform helpers
# -------------------------------------------------------------------
def derive_current_platform(providers_json: Optional[Dict[str, Any]]) -> str:
    """
    Very simple rule to derive a primary 'current_platform' label.

    Looks at US flatrate/ads providers and maps common names.
    """
    if not providers_json:
        return "Rent/Buy"

    results = providers_json.get("results", {})
    us = results.get("US") or results.get("GB") or {}

    buckets = []
    for key in ["flatrate", "ads", "rent", "buy"]:
        buckets.extend(us.get(key, []))

    names = {p.get("provider_name", "") for p in buckets if p.get("provider_name")}

    # Simple mapping to your app's labels
    map_rules = {
        "Netflix": "Netflix",
        "Max": "Max",
        "HBO Max": "Max",
        "HBO": "Max",
        "Amazon Prime Video": "Prime Video",
        "Prime Video": "Prime Video",
        "Hulu": "Hulu",
        "Peacock": "Peacock",
        "Disney+": "Disney+",
    }

    normalized: List[str] = []
    for name in names:
        mapped = None
        for raw, label in map_rules.items():
            if raw.lower() in name.lower():
                mapped = label
                break
        if mapped:
            normalized.append(mapped)

    if not normalized:
        return "Rent/Buy"

    # If on Netflix + Max, keep that; else pick first
    normalized = sorted(set(normalized))
    if "Netflix" in normalized and "Max" in normalized:
        return "Netflix + Max"

    return normalized[0]


def compute_value_score(details: Dict[str, Any]) -> float:
    """
    Simple heuristic value score from TMDB stats.
    You can tweak this without breaking the app.
    """
    pop = float(details.get("popularity") or 0.0)
    votes = float(details.get("vote_count") or 0.0)
    rating = float(details.get("vote_average") or 0.0)

    # Weighted mix: popularity + scaled rating + log(votes)
    from math import log10

    vote_term = log10(votes + 1.0)
    return pop + rating * 2.0 + vote_term


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------
def main() -> None:
    if not SEED_PATH.exists():
        print(f"[error] Seed file not found: {SEED_PATH}", file=sys.stderr)
        sys.exit(1)

    seed_df = pd.read_csv(SEED_PATH)
    if "tmdb_id" not in seed_df.columns:
        print("[error] wb_seed_titles.csv must have a 'tmdb_id' column.", file=sys.stderr)
        sys.exit(1)

    enriched_rows: List[Dict[str, Any]] = []

    for idx, row in seed_df.iterrows():
        raw_title = str(row.get("title", "")).strip()
        tmdb_raw = row.get("tmdb_id")

        if pd.isna(tmdb_raw):
            print(f"[warn] Row {idx} has no tmdb_id; skipping")
            continue

        try:
            tmdb_id = int(tmdb_raw)
        except Exception:
            print(f"[warn] Row {idx} has invalid tmdb_id={tmdb_raw!r}; skipping")
            continue

        # Try to read media type from any of these columns
        media_hint = None
        for col in ("media_type", "tmdb_media_type", "tmdb_type"):
            if col in seed_df.columns:
                val = row.get(col)
                if pd.notna(val):
                    media_hint = str(val).lower()
                    break

        media_type_used, details, providers = fetch_tmdb_bundle(
            tmdb_id=tmdb_id,
            media_type_hint=media_hint,
        )


        if details is None:
            # already logged a warning
            continue

        # Title fields
        name = (
            details.get("name")
            or details.get("title")
            or details.get("original_name")
            or details.get("original_title")
            or raw_title
        )
        original_title = (
            details.get("original_name")
            or details.get("original_title")
            or name
        )

        franchise_group = row.get("franchise_group", "Standalone")
        owner = row.get("owner", "Warner Bros Discovery")

        current_platform = derive_current_platform(providers)

        value_score = compute_value_score(details)

        enriched_rows.append(
            {
                "title": name,
                "franchise_group": franchise_group,
                "owner": owner,
                "current_platform": current_platform,
                "tmdb_id": tmdb_id,
                "tmdb_media_type": media_type_used,
                "tmdb_popularity": details.get("popularity", 0.0),
                "tmdb_vote_count": details.get("vote_count", 0),
                "tmdb_vote_average": details.get("vote_average", 0.0),
                "tmdb_poster_path": details.get("poster_path"),
                "tmdb_original_title": original_title,
                "value_score_raw": value_score,
            }
        )

    if not enriched_rows:
        print("[error] No rows enriched; check TMDB_API_KEY and network.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(enriched_rows)

    total_raw = df["value_score_raw"].sum()
    if total_raw > 0:
        df["value_score_norm"] = df["value_score_raw"] / total_raw
    else:
        df["value_score_norm"] = 0.0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"[info] Wrote {len(df)} enriched titles to {OUT_PATH}")


if __name__ == "__main__":
    main()
