import os, json, argparse, faiss, cv2
import numpy as np
import face_recognition as fr
from utils.embed import l2_normalize

ROOT = os.path.dirname(__file__)
INDEX_PATH = os.path.join(ROOT, "..", "faiss_index.bin")
META_PATH  = os.path.join(ROOT, "..", "faiss_meta.json")

def load_index_and_meta():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise SystemExit("Index/meta not found. Run scripts/build_index.py first.")
    index = faiss.read_index(INDEX_PATH)
    meta  = json.load(open(META_PATH, "r", encoding="utf-8"))
    return index, meta

def main():
    ap = argparse.ArgumentParser(description="Live webcam face search with FAISS")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.65)
    args = ap.parse_args()

    index, meta = load_index_and_meta()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = frame[:, :, ::-1]  # BGR -> RGB
        boxes = fr.face_locations(rgb, model="hog")
        encs  = fr.face_encodings(rgb, boxes)  # list of 128-d float64

        if encs:
            Q = np.array(encs, dtype="float32")
            Q = l2_normalize(Q)
            D, I = index.search(Q, args.k)  # cosine scores

            for (top, right, bottom, left), scores, idxs in zip(boxes, D, I):
                # best match
                best_score = scores[0]
                best_idx   = idxs[0] if len(idxs) else -1
                if best_idx >= 0 and best_idx < len(meta) and best_score >= args.threshold:
                    label = f"{meta[best_idx]['label']} ({best_score:.2f})"
                    color = (0, 200, 0)
                else:
                    label = f"Unknown ({best_score:.2f})"
                    color = (0, 0, 200)

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                y = top - 10 if top - 10 > 10 else top + 20
                cv2.putText(frame, label, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Live FAISS Face Search", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()