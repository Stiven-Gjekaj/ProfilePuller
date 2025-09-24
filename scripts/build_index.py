import os, json, faiss
import numpy as np
from utils.iohelpers import iter_image_paths, label_from_path
from utils.embed import image_to_embeddings, l2_normalize

# Adjust if your downloader saves elsewhere
KNOWN_DIR   = os.path.join(os.path.dirname(__file__), "..", "faces", "known")
INDEX_PATH  = os.path.join(os.path.dirname(__file__), "..", "faiss_index.bin")
META_PATH   = os.path.join(os.path.dirname(__file__), "..", "faiss_meta.json")

def main():
    vectors = []
    metadata = []
    count_images = 0
    for p in iter_image_paths(KNOWN_DIR):
        count_images += 1
        try:
            embs = image_to_embeddings(p, model="hog")  # use "cnn" if you have it set up
            lbl = label_from_path(p, KNOWN_DIR)
            for e in embs:
                vectors.append(e)
                metadata.append({"path": os.path.relpath(p, start=os.path.dirname(__file__)), "label": lbl})
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")

    if not vectors:
        raise SystemExit("No embeddings found. Make sure faces are in faces/known/<label>/...")

    X = np.stack(vectors).astype("float32")  # (N, 128)
    X = l2_normalize(X)  # cosine similarity

    d = X.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product == cosine on normalized vectors
    index.add(X)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[OK] Indexed {len(vectors)} faces from {count_images} images")
    print(f"     Index: {INDEX_PATH}")
    print(f"     Meta : {META_PATH}")

if __name__ == "__main__":
    main()