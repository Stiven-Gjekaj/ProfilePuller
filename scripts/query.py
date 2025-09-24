import os, sys, json, argparse, faiss
import numpy as np
from utils.embed import image_to_embeddings, l2_normalize

ROOT = os.path.dirname(__file__)
INDEX_PATH = os.path.join(ROOT, "..", "faiss_index.bin")
META_PATH  = os.path.join(ROOT, "..", "faiss_meta.json")

def load_index():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("Index not found. Run scripts/build_index.py first.")
    return faiss.read_index(INDEX_PATH)

def load_meta():
    if not os.path.exists(META_PATH):
        raise FileNotFoundError("Meta not found. Run scripts/build_index.py first.")
    return json.load(open(META_PATH, "r", encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser(description="Query FAISS face index with an image.")
    ap.add_argument("image", help="Path to query image")
    ap.add_argument("--k", type=int, default=5, help="Top-k results")
    ap.add_argument("--threshold", type=float, default=0.65, help="Cosine similarity threshold (0-1). Higher = more similar")
    args = ap.parse_args()

    index = load_index()
    meta  = load_meta()

    embs = image_to_embeddings(args.image, model="hog")
    if not embs:
        raise SystemExit("No faces detected in query image.")

    Q = np.stack(embs).astype("float32")
    Q = l2_normalize(Q)

    D, I = index.search(Q, args.k)  # cosine scores (higher = better)

    for qi, (scores, idxs) in enumerate(zip(D, I), start=1):
        print(f"\n[Query face #{qi}]")
        for rank, (s, idx) in enumerate(zip(scores, idxs), start=1):
            if idx < 0 or idx >= len(meta):  # FAISS can return -1 if empty
                continue
            m = meta[idx]
            flag = "✅" if s >= args.threshold else "—"
            print(f"#{rank}: {flag} score={s:.4f} | label={m['label']} | path={m['path']}")

if __name__ == "__main__":
    main()