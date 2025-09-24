#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import time

import faiss
import numpy as np
from rich.console import Console
from tqdm import tqdm

try:  # pragma: no cover - import resolution
    from scripts.utils.embed import image_to_embeddings, l2_normalize
    from scripts.utils.iohelpers import ensure_dir, iter_image_paths, label_from_path, save_json
except ImportError:  # pragma: no cover - executed when run as a script
    SCRIPT_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    from scripts.utils.embed import image_to_embeddings, l2_normalize
    from scripts.utils.iohelpers import ensure_dir, iter_image_paths, label_from_path, save_json

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

console = Console()
DEFAULT_KNOWN = ROOT_DIR / "faces" / "known"
DEFAULT_INDEX = ROOT_DIR / "faiss_index.bin"
DEFAULT_META = ROOT_DIR / "faiss_meta.json"
DEFAULT_DIM = 128


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS index from known faces.")
    parser.add_argument(
        "--known", type=Path, default=DEFAULT_KNOWN, help="Directory of known faces"
    )
    parser.add_argument(
        "--out-index", type=Path, default=DEFAULT_INDEX, help="Output path for FAISS index"
    )
    parser.add_argument(
        "--out-meta", type=Path, default=DEFAULT_META, help="Output path for metadata JSON"
    )
    parser.add_argument(
        "--model", choices=["hog", "cnn"], default="hog", help="face_recognition model"
    )
    parser.add_argument(
        "--metric", choices=["cosine", "l2"], default="cosine", help="Similarity metric"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return parser.parse_args(argv)


def create_index(metric: str, dimension: int) -> faiss.Index:
    if metric == "cosine":
        return faiss.IndexFlatIP(dimension)
    return faiss.IndexFlatL2(dimension)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    def _handle_interrupt(signum, frame):  # pragma: no cover - user interaction
        console.print("\n[bold yellow]Interrupted. Cleaning up...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_interrupt)

    image_paths = list(iter_image_paths(args.known))
    if not image_paths:
        console.print(f"[yellow]No images found under {args.known}. Building an empty index.")

    vectors: list[np.ndarray] = []
    metadata: list[dict] = []
    faces_indexed = 0

    face_bar = tqdm(total=0, unit="face", leave=False, dynamic_ncols=True, desc="Faces indexed")
    try:
        for image_path in tqdm(
            image_paths, desc="Scanning images", unit="image", dynamic_ncols=True
        ):
            embeddings = image_to_embeddings(image_path, model=args.model)
            label = label_from_path(image_path, args.known)
            if not embeddings:
                console.print(f"[yellow]No faces detected in {image_path}")
                continue

            relative_path = str(Path(image_path).resolve())
            try:
                relative_path = str(Path(image_path).relative_to(args.known))
            except ValueError:
                pass

            for embedding in embeddings:
                vectors.append(embedding.astype(np.float32))
                metadata.append(
                    {
                        "label": label,
                        "path": relative_path,
                    }
                )
            faces_indexed += len(embeddings)
            face_bar.update(len(embeddings))
            if args.verbose:
                console.log(f"Indexed {len(embeddings)} faces from {image_path}")
    except KeyboardInterrupt:  # pragma: no cover - handled for UX
        face_bar.close()
        console.print("[bold yellow]Index build interrupted by user.")
        return 1
    finally:
        face_bar.close()

    if not vectors:
        console.print("[yellow]No embeddings created; writing empty index files.")
        dimension = DEFAULT_DIM
        index = create_index(args.metric, dimension)
    else:
        matrix = np.stack(vectors).astype(np.float32)
        if args.metric == "cosine":
            matrix = l2_normalize(matrix)
        dimension = matrix.shape[1]
        index = create_index(args.metric, dimension)
        index.add(matrix)

    ensure_dir(args.out_index)
    ensure_dir(args.out_meta)
    try:
        faiss.write_index(index, str(args.out_index))
    except Exception as exc:  # pragma: no cover - disk errors are rare
        console.print(f"[red]Failed to write index: {exc}")
        return 1

    meta_payload = {
        "created_utc": time.time(),
        "metric": args.metric,
        "dim": dimension,
        "count": faces_indexed,
        "items": metadata,
    }
    save_json(meta_payload, args.out_meta)

    console.print("\n[bold green]Index build complete")
    console.print(f"Images scanned: {len(image_paths)}")
    console.print(f"Faces indexed: {faces_indexed}")
    console.print(f"Index saved to: {args.out_index}")
    console.print(f"Metadata saved to: {args.out_meta}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
