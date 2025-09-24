#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

try:  # pragma: no cover - import resolution
    import cv2
except Exception as exc:  # pragma: no cover - library may be missing system deps
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:  # pragma: no cover - import resolution
    _CV2_IMPORT_ERROR = None

import faiss
import numpy as np
from rich.console import Console

try:  # pragma: no cover - import resolution
    from scripts.utils.embed import l2_normalize
    from scripts.utils.iohelpers import load_json
except ImportError:  # pragma: no cover - executed when run as a script
    SCRIPT_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    from scripts.utils.embed import l2_normalize
    from scripts.utils.iohelpers import load_json

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

console = Console()
DEFAULT_INDEX = ROOT_DIR / "faiss_index.bin"
DEFAULT_META = ROOT_DIR / "faiss_meta.json"
DEFAULT_THRESHOLD = 0.65


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live webcam face search with FAISS")
    parser.add_argument("--cam", type=int, default=0, help="Camera device index")
    parser.add_argument("--k", type=int, default=3, help="Top-k results per face")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD, help="Match threshold"
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="Path to FAISS index")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META, help="Path to metadata JSON")
    parser.add_argument(
        "--model", choices=["hog", "cnn"], default="hog", help="face_recognition model"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return parser.parse_args(argv)


def load_index_and_meta(index_path: Path, meta_path: Path) -> tuple[faiss.Index, dict]:
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Index/meta not found. Run scripts/build_index.py first.")
    index = faiss.read_index(str(index_path))
    meta = load_json(meta_path)
    if isinstance(meta, list):
        meta = {"metric": "cosine", "dim": 128, "count": len(meta), "items": meta}
    meta.setdefault("metric", "cosine")
    meta.setdefault("items", [])
    meta.setdefault("dim", index.d)
    return index, meta


def main(argv=None) -> int:
    args = parse_args(argv)

    if cv2 is None:
        console.print(
            "[red]OpenCV (cv2) is required for live search. "
            "Install opencv-python-headless or the full OpenCV build."
        )
        if _CV2_IMPORT_ERROR:
            console.print(f"[red]{_CV2_IMPORT_ERROR}")
        return 1

    try:
        import face_recognition as fr
    except Exception as exc:  # pragma: no cover - dependency hinting
        console.print(
            "[red]face_recognition is required for live search. "
            "Install face-recognition and its models (see README)."
        )
        console.print(f"[red]{exc}")
        return 1

    try:
        index, meta = load_index_and_meta(args.index, args.meta)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}")
        return 1

    metric = meta.get("metric", "cosine")
    threshold = args.threshold
    if args.threshold == DEFAULT_THRESHOLD and metric == "l2":
        threshold = 0.6
    items = meta.get("items", [])
    if index.ntotal == 0 or not items:
        console.print("[yellow]Index is empty. Add faces and rebuild the index.")
        return 0

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        console.print(f"[red]Could not open webcam {args.cam}.")
        return 1

    console.print("Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                console.print("[red]Failed to read from webcam.")
                break

            rgb = frame[:, :, ::-1]
            boxes = fr.face_locations(rgb, model=args.model)
            if args.verbose:
                console.log(f"Detected {len(boxes)} face(s)")
            encodings = fr.face_encodings(rgb, boxes)
            if encodings:
                query = np.asarray(encodings, dtype=np.float32)
                if metric == "cosine":
                    query = l2_normalize(query)
                distances, indices = index.search(query, args.k)
            else:
                distances = []
                indices = []

            for (top, right, bottom, left), scores, idxs in zip(boxes, distances, indices):
                if not len(scores):
                    continue
                best_idx = idxs[0]
                best_score = scores[0]
                label = "Unknown"
                color = (0, 0, 200)
                if best_idx >= 0 and best_idx < len(items):
                    entry = items[best_idx]
                    if metric == "cosine":
                        ok = best_score >= threshold
                        display_score = best_score
                    else:
                        dist = math.sqrt(max(best_score, 0.0))
                        ok = dist <= threshold
                        display_score = dist
                    if ok:
                        label = f"{entry.get('label', '?')} ({display_score:.2f})"
                        color = (0, 200, 0)
                    else:
                        label = f"Unknown ({display_score:.2f})"
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                y = top - 10 if top - 10 > 10 else top + 20
                cv2.putText(frame, label, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Live FAISS Face Search", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:  # pragma: no cover - interactive
        console.print("\n[bold yellow]Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
