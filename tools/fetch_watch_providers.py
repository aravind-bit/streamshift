import os
import time
import json
from pathlib import Path
from typing import Literal, Optional, Dict, Any

import requests
import pandas as pd

# -------- CONFIG --------

DATA_PATH = Path("data/titles_enriched.csv")
OUT_PATH = Path("data/titles_enriched.csv")  # overwrite in-place (we already backed up)
CACHE_PATH = Path("data/tmdb_watch_cache.json")

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_REGION = "US"

# If you have a media type column, set its name here. Otherwise default_type will be used.
MEDIA_TYPE_COL = "tmdb_media_type"  # e.g. "tv" or "movie"
DEFAULT_MEDIA_TYPE: Literal["tv", "movie"] = "tv"

# Map TMDb provider_name -> internal platform label used in app
PROVIDER_MAP = {
    "Netflix": "Netflix",
    "Netflix Kids": "Netflix",
    "HBO Max": "Max",
    "Max": "Max",
    "Amazon Prime Video": "Prime Video",
    "Amazon Prime": "Prime Video",
    "Hulu": "Hulu",
    "Disney Plus": "Disney+",
    "Disney+": "Disney+",
    "Paramount Plus": "Paramount+",
    "Paramount+": "Paramount+",
    "Peacock": "Peacock",
}

REQUEST_SLEEP_SECONDS = 0.25  # small delay to be nice to TMDb

# -------- HELPERS --------


def load_cache() -> Dict[str, Any]:
    if CACHE_PATH.exists():
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(exist_ok=True, parents=True)
    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def fetch_watch_providers(
    tmdb_id: int,
    media_type: Literal["tv", "movie"],
    cache: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Call TMDb watch/providers for given id, trying tv first then movie (or vice versa
    depending on media_type). This handles cases where a tmdb_id is actually a movie
    but we guessed 'tv', or the other way around.

    Caches responses by key: f"{resolved_type}:{tmdb_id}".
    Returns the US entry (dict) or None if missing.
    """
    if TMDB_API_KEY is None:
        raise RuntimeError("TMDB_API_KEY environment variable is not set.")

    # We'll try two candidates in order:
    #   1) the declared media_type
    #   2) the opposite type
    candidates: list[Literal["tv", "movie"]] = (
        [media_type, "movie" if media_type == "tv" else "tv"]
        if media_type in {"tv", "movie"}
        else ["tv", "movie"]
    )

    last_status = None
    for mt in candidates:
        key = f"{mt}:{tmdb_id}"
        if key in cache:
            return cache[key]

        url = f"https://api.themoviedb.org/3/{mt}/{tmdb_id}/watch/providers"
        params = {"api_key": TMDB_API_KEY}

        resp = requests.get(url, params=params, timeout=10)
        last_status = resp.status_code

        if resp.status_code == 200:
            data = resp.json()
            us_entry = data.get("results", {}).get(TMDB_REGION)
            cache[key] = us_entry  # store raw US block (or None)
            time.sleep(REQUEST_SLEEP_SECONDS)
            return us_entry

        # For non-404 errors, log and stop trying
        if resp.status_code != 404:
            print(f"[warn] TMDb providers {mt}:{tmdb_id} -> status {resp.status_code}")
            cache[key] = None
            return None

        # If 404, fall through and try the next candidate

    # If we’re here, all candidates failed (likely 404 on both tv and movie)
    print(f"[warn] TMDb providers {tmdb_id} -> no match on tv/movie (last status={last_status})")
    cache[f"unknown:{tmdb_id}"] = None
    return None



def derive_platform_label(us_entry: Optional[Dict[str, Any]]) -> (str, str):
    """
    From the US watch providers block, derive:
    - internal label for current_platform (e.g. 'Netflix / Max')
    - raw provider names joined (for traceability)
    """
    if not us_entry:
        return "Unknown", ""

    flatrate = us_entry.get("flatrate") or []
    if not flatrate:
        # You could also consider "rent"/"buy" here if you want
        return "Unknown", ""

    providers_raw = [p.get("provider_name", "").strip() for p in flatrate if p.get("provider_name")]
    providers_raw = [p for p in providers_raw if p]  # drop blanks

    if not providers_raw:
        return "Unknown", ""

    # Map to internal labels
    mapped = [PROVIDER_MAP.get(p, p) for p in providers_raw]

    # Deduplicate & sort
    unique_mapped = sorted(set(mapped))
    unique_raw = sorted(set(providers_raw))

    internal_label = " / ".join(unique_mapped)
    raw_label = " / ".join(unique_raw)

    return internal_label, raw_label


def determine_media_type(row: pd.Series) -> Literal["tv", "movie"]:
    if MEDIA_TYPE_COL in row and isinstance(row[MEDIA_TYPE_COL], str):
        t = row[MEDIA_TYPE_COL].strip().lower()
        if t in {"tv", "tv_show", "series"}:
            return "tv"
        if t in {"movie", "film"}:
            return "movie"
    return DEFAULT_MEDIA_TYPE


# -------- MAIN --------


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Expected data file at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "tmdb_id" not in df.columns:
        raise ValueError("Expected a 'tmdb_id' column in titles_enriched.csv")

    cache = load_cache()

    updated_rows = 0

    # We'll populate two columns:
    # - current_platform_raw: raw provider_name list
    # - watch_providers_json: JSON text of the US block
    if "current_platform_raw" not in df.columns:
        df["current_platform_raw"] = ""
    if "watch_providers_json" not in df.columns:
        df["watch_providers_json"] = ""

    for idx, row in df.iterrows():
        tmdb_id = row["tmdb_id"]
        try:
            tmdb_id_int = int(tmdb_id)
        except Exception:
            # Skip rows without a valid ID
            continue

        media_type = determine_media_type(row)
        us_entry = fetch_watch_providers(tmdb_id_int, media_type, cache)

        internal_label, raw_label = derive_platform_label(us_entry)

        # Only update if we got something meaningful
        if internal_label != "Unknown" or raw_label:
            df.at[idx, "current_platform"] = internal_label
            df.at[idx, "current_platform_raw"] = raw_label
            df.at[idx, "watch_providers_json"] = json.dumps(
                us_entry, ensure_ascii=False
            ) if us_entry is not None else ""
            updated_rows += 1

        if idx % 25 == 0:
            print(f"...processed {idx} rows, updated so far: {updated_rows}")

    print(f"\nDone. Updated {updated_rows} rows with TMDb watch provider data.")

    # Quick look at resulting platform mix
    if "current_platform" in df.columns:
        print("\nResulting current_platform distribution (top 10):")
        print(df["current_platform"].value_counts().head(10))

    # Save cache and updated CSV
    save_cache(cache)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nWritten updated data to {OUT_PATH.absolute()}")



if __name__ == "__main__":
    main()

