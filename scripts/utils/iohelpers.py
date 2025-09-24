from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_image_paths(root_dir: str | Path) -> Iterator[Path]:
    """Yield image files under ``root_dir`` recursively."""
    root_path = Path(root_dir)
    if not root_path.exists():
        return

    for path in sorted(root_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            yield path


def label_from_path(path: str | Path, known_root: str | Path) -> str:
    """Infer the label for ``path`` relative to ``known_root``."""
    image_path = Path(path)
    base = Path(known_root)
    try:
        rel = image_path.relative_to(base)
    except ValueError:
        return image_path.stem

    parts = rel.parts
    if len(parts) >= 2:
        return parts[0]
    return image_path.stem


def ensure_dir(path: str | Path) -> None:
    """Create ``path`` (or its parent) if missing."""
    target = Path(path)
    directory = target if target.is_dir() else target.parent
    if directory:
        directory.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: str | Path) -> None:
    """Write ``obj`` to ``path`` as pretty JSON."""
    file_path = Path(path)
    ensure_dir(file_path)
    file_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    """Load JSON data from ``path``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
