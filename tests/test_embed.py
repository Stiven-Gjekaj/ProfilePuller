from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import numpy as np


def test_image_to_embeddings_retries_with_higher_upsample(tmp_path, monkeypatch) -> None:
    calls: list[int] = []

    def fake_load_image(path: str) -> np.ndarray:  # pragma: no cover - exercised via stub
        return np.zeros((10, 10, 3), dtype=np.uint8)

    def fake_face_locations(
        image: np.ndarray,
        number_of_times_to_upsample: int = 1,
        model: str = "hog",
    ) -> list[tuple[int, int, int, int]]:
        calls.append(number_of_times_to_upsample)
        if number_of_times_to_upsample >= 2:
            return [(0, 1, 2, 3)]
        return []

    def fake_face_encodings(
        image: np.ndarray, boxes: list[tuple[int, int, int, int]]
    ) -> list[np.ndarray]:
        return [np.ones(128, dtype=np.float32)] if boxes else []

    fake_fr = SimpleNamespace(
        load_image_file=fake_load_image,
        face_locations=fake_face_locations,
        face_encodings=fake_face_encodings,
    )

    monkeypatch.setenv("FAKE_EMB", "0")
    monkeypatch.setitem(sys.modules, "face_recognition", fake_fr)

    import scripts.utils.embed as embed

    embed = importlib.reload(embed)

    image_path = tmp_path / "face.jpg"
    image_path.write_bytes(b"fake")

    embeddings = embed.image_to_embeddings(image_path, model="hog")

    assert len(embeddings) == 1
    assert calls == [1, 2]

    # Restore module state for subsequent tests.
    monkeypatch.setenv("FAKE_EMB", "1")
    importlib.reload(embed)
