from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Try sentence-transformers (Torch). If not available, try FastEmbed (ONNX).
USE_ST = True
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    USE_ST = False

if not USE_ST:
    try:
        from fastembed import TextEmbedding  # type: ignore
    except Exception:
        raise SystemExit("No embedding backend available. Install sentence-transformers or fastembed.")

import faiss  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CSV  = DATA / "franchises.csv"

OUT_INDEX = DATA / "embeddings.index"
OUT_META  = DATA / "embeddings_meta.parquet"

def load_titles() -> pd.DataFrame:
    if not CSV.exists():
        raise SystemExit(f"Missing {CSV}")
    df = pd.read_csv(CSV)
    for c in ["title", "genre_tags"]:
        if c not in df.columns:
            df[c] = ""
    # text field: title + tags
    df["_text"] = (df["title"].astype(str).str.strip() + " " + df["genre_tags"].astype(str).str.strip()).str.strip()
    # drop empty
    df = df[df["_text"].str.len() > 0].copy()
    df.reset_index(drop=True, inplace=True)
    return df

def build_embeddings(texts: list[str]) -> np.ndarray:
    if USE_ST:
        model_name = os.getenv("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name)
        emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        return np.asarray(emb, dtype="float32")
    else:
        model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        embs = []
        for batch in model.embed(texts, batch_size=64):
            embs.extend(batch)
        arr = np.asarray(embs, dtype="float32")
        # L2 normalize
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr

def main():
    df = load_titles()
    if df.empty:
        raise SystemExit("No rows to embed (empty _text).")
    texts = df["_text"].tolist()
    print(f"[build] rows: {len(texts)}")

    embs = build_embeddings(texts)
    dim = embs.shape[1]
    print(f"[build] embedding shape: {embs.shape}")

    # FAISS index (L2 / inner product). We normalized, so use inner product.
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, str(OUT_INDEX))
    print(f"[write] faiss index: {OUT_INDEX} (ntotal={index.ntotal})")

    meta = pd.DataFrame({
        "row_id": np.arange(len(df), dtype=np.int32),
        "title": df["title"].astype(str),
        "text": df["_text"].astype(str)
    })
    meta.to_parquet(OUT_META, index=False)
    print(f"[write] meta parquet: {OUT_META} (rows={len(meta)})")

if __name__ == "__main__":
    main()
