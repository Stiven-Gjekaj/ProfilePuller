import os
import json
from typing import Iterator, Any

# File extensions that count as images
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def iter_image_paths(root_dir: str) -> Iterator[str]:
    """
    Recursively yield all image file paths under root_dir.
    """
    for base, _, files in os.walk(root_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXTS:
                yield os.path.join(base, f)

def label_from_path(path: str, known_root: str) -> str:
    """
    Extract a 'label' for an image.
    If the image is inside a subfolder under known_root, that subfolder is used as the label.
    Example: faces/known/alice/img1.jpg -> 'alice'

    If the image is directly inside known_root with no subfolder,
    fall back to the filename stem.
    """
    rel = os.path.relpath(path, known_root)
    parts = rel.replace("\\", "/").split("/")
    if len(parts) >= 2:
        return parts[0]  # subfolder
    return os.path.splitext(os.path.basename(path))[0]

def ensure_dir(path: str) -> None:
    """
    Create the directory (and parents) if it doesn't exist.
    If given a file path, creates its parent directory.
    """
    dir_path = path if os.path.isdir(path) else os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

def save_json(obj: Any, path: str) -> None:
    """
    Save a Python object as pretty JSON.
    """
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Any:
    """
    Load JSON into a Python object.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)