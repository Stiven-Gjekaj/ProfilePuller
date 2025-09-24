from __future__ import annotations

from collections.abc import Iterable
import hashlib
import os
from pathlib import Path

import numpy as np

FAKE_EMB = os.getenv("FAKE_EMB", "0") == "1"

if not FAKE_EMB:
    try:  # pragma: no cover - import guarded by FAKE_EMB in tests
        import face_recognition as fr
    except ImportError as exc:  # pragma: no cover - better error for missing dep
        raise ImportError(
            "face_recognition is required for real embeddings. "
            "Set FAKE_EMB=1 for deterministic test embeddings."
        ) from exc
else:  # pragma: no cover - exercised indirectly in tests
    fr = None


def _fake_embedding(path: Path) -> np.ndarray:
    """Create a deterministic 128-d vector based on the file path."""
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).digest()
    # Repeat digest to reach 128 values, then normalise to [0, 1].
    repeated = (digest * ((128 // len(digest)) + 1))[:128]
    arr = np.frombuffer(repeated, dtype=np.uint8).astype(np.float32)
    return arr / 255.0


def _normalise_upsample(upsample_times: int | Iterable[int] | None) -> list[int]:
    """Normalise ``upsample_times`` into a list of unique integers."""
    if upsample_times is None:
        return [1, 2]
    if isinstance(upsample_times, int):
        return [upsample_times]

    normalised: list[int] = []
    seen: set[int] = set()
    for value in upsample_times:
        try:
            int_value = int(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
        if int_value not in seen:
            seen.add(int_value)
            normalised.append(int_value)
    return normalised or [1]


def image_to_embeddings(
    path: str | Path,
    model: str = "hog",
    *,
    upsample_times: int | Iterable[int] | None = None,
) -> list[np.ndarray]:
    """Return a list of embeddings for every detected face in ``path``."""
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if FAKE_EMB:
        return [_fake_embedding(image_path)]

    if fr is None:  # pragma: no cover - safety net
        raise RuntimeError("face_recognition library failed to import")

    image = fr.load_image_file(str(image_path))
    boxes: list[tuple[int, int, int, int]] = []
    for upsample in _normalise_upsample(upsample_times):
        boxes = fr.face_locations(image, number_of_times_to_upsample=upsample, model=model)
        if boxes:
            break
    encodings = fr.face_encodings(image, boxes)
    return [np.asarray(encoding, dtype=np.float32) for encoding in encodings]


def encode_one_face(
    path: str | Path,
    model: str = "hog",
    *,
    upsample_times: int | Iterable[int] | None = None,
) -> np.ndarray:
    """Return the first detected face embedding or raise if none are found."""
    embeddings = image_to_embeddings(path, model=model, upsample_times=upsample_times)
    if not embeddings:
        raise ValueError(f"No face found in image: {path}")
    return embeddings[0]


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalise each row of ``vectors`` to unit length."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, eps)
