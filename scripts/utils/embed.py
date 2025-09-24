from typing import List, Tuple
import numpy as np
import face_recognition as fr

def image_to_embeddings(path: str, model: str = "hog") -> List[np.ndarray]:
    """
    Returns a list of 128-d float32 face embeddings for all faces in the image.
    model: "hog" (fast, CPU) or "cnn" (slower, better; requires dlib with CUDA to be fast)
    """
    img = fr.load_image_file(path)
    boxes = fr.face_locations(img, model=model)
    encs = fr.face_encodings(img, boxes)  # float64
    return [e.astype("float32") for e in encs]

def encode_one_face(path: str, model: str = "hog") -> np.ndarray:
    """Encode first detected face; raise if none."""
    embs = image_to_embeddings(path, model=model)
    if not embs:
        raise ValueError(f"No face found in image: {path}")
    return embs[0]

def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize rows to unit length for cosine similarity via inner product."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)