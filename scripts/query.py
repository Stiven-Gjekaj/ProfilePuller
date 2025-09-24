#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import faiss
import numpy as np
from rich.console import Console
from rich.table import Table

try:  # pragma: no cover - import resolution
    from scripts.utils.embed import image_to_embeddings, l2_normalize
    from scripts.utils.iohelpers import load_json
except ImportError:  # pragma: no cover - executed when run as a script
    SCRIPT_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    from scripts.utils.embed import image_to_embeddings, l2_normalize
    from scripts.utils.iohelpers import load_json

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

console = Console()
DEFAULT_INDEX = ROOT_DIR / "faiss_index.bin"
DEFAULT_META = ROOT_DIR / "faiss_meta.json"
DEFAULT_COSINE_THRESHOLD = 0.65
DEFAULT_L2_THRESHOLD = 0.6


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query FAISS face index with an image.")
    parser.add_argument("image", type=Path, help="Path to query image")
    parser.add_argument("--k", type=int, default=5, help="Top-k results")
    parser.add_argument(
        "--threshold",
        type=float,
        help="Match threshold (cosine >= threshold or L2 <= threshold)",
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="Path to FAISS index")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META, help="Path to metadata JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return parser.parse_args(argv)


def load_index(index_path: Path) -> faiss.Index:
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")
    try:
        return faiss.read_index(str(index_path))
    except Exception as exc:  # pragma: no cover - rare corruption
        raise RuntimeError(f"Failed to read index {index_path}: {exc}") from exc


def load_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    try:
        data = load_json(meta_path)
    except Exception as exc:  # pragma: no cover - malformed json
        raise RuntimeError(f"Failed to read metadata {meta_path}: {exc}") from exc
    if isinstance(data, list):  # backwards compatibility
        data = {
            "metric": "cosine",
            "dim": 128,
            "count": len(data),
            "items": data,
        }
    data.setdefault("metric", "cosine")
    data.setdefault("dim", 128)
    data.setdefault("items", [])
    data.setdefault("count", len(data["items"]))
    return data


def compute_threshold(metric: str, user_threshold: float | None) -> float:
    if user_threshold is not None:
        return user_threshold
    return DEFAULT_COSINE_THRESHOLD if metric == "cosine" else DEFAULT_L2_THRESHOLD


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        index = load_index(args.index)
        meta = load_meta(args.meta)
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]{exc}")
        return 1

    metric = meta.get("metric", "cosine")
    threshold = compute_threshold(metric, args.threshold)
    items = meta.get("items", [])
    expected_dim = int(meta.get("dim", index.d))

    if index.d != expected_dim:
        console.print(f"[red]Index dimension ({index.d}) does not match metadata ({expected_dim}).")
        return 1

    if index.ntotal == 0 or not items:
        console.print("[yellow]Index is empty. Add faces and rebuild the index.")
        return 0

    embeddings = image_to_embeddings(args.image)
    if not embeddings:
        console.print("[yellow]No faces detected in query image.")
        return 0

    matrix = np.stack(embeddings).astype(np.float32)
    if metric == "cosine":
        matrix = l2_normalize(matrix)
        distances, indices = index.search(matrix, args.k)
    else:
        distances, indices = index.search(matrix, args.k)

    if args.verbose:
        console.log(f"Query embeddings: {matrix.shape} | Metric: {metric} | Threshold: {threshold}")

    table = Table(title=f"Results for {args.image}")
    table.add_column("Query #")
    table.add_column("Rank")
    table.add_column("Score")
    table.add_column("Label")
    table.add_column("Path")

    for qi, (scores, idxs) in enumerate(zip(distances, indices), start=1):
        for rank, (score, idx) in enumerate(zip(scores, idxs), start=1):
            if idx < 0 or idx >= len(items):
                continue
            entry = items[idx]
            if metric == "cosine":
                ok = score >= threshold
                display_score = f"{score:.4f}"
            else:
                dist = math.sqrt(max(score, 0.0))
                ok = dist <= threshold
                display_score = f"{dist:.4f}"
            flag = "✅" if ok else "—"
            table.add_row(
                str(qi),
                f"#{rank}",
                f"{flag} {display_score}",
                entry.get("label", "?"),
                entry.get("path", "?"),
            )

    console.print(table)
    console.print(f"Metric: {metric} | Threshold: {threshold}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
