# Knot-Echo Upgrade TODOs

## Outstanding

### 🔥 Embeddings & Accuracy

- Add backend adapter system (`emb_backends.py`)
- Implement `FaceRecBackend` (existing)
- Implement `InsightFaceBackend` (ArcFace via InsightFace + onnxruntime)
- Add `--backend {facerec,insight}` flag to `scripts/build_index.py` and `scripts/query.py`
- Store backend selection in `faiss_meta.json`

### ⚡ Index Scaling

- Replace IndexFlat with IVF + HNSW (for large galleries)
- Tune `nlist` and `nprobe` defaults

### 📦 Batch & Parallelism

- Add `--batch-size` option to `scripts/build_index.py` and `scripts/query.py`
- Parallelize image decoding/alignment with multiprocessing
- Batch embeddings to reduce model calls

### 🧹 Data Hygiene

- Implement duplicate detection (cosine > 0.98 → keep one)
- Add clustering (DBSCAN/HDBSCAN) for unlabeled identities
- Store per-label average embedding for faster lookup

### 🎯 Threshold Calibration

- Create `scripts/calibrate.py` for eval pairs
- Sweep thresholds to produce ROC/F1/EER plots
- Write recommended thresholds into `faiss_meta.json`

### 🌐 API & Frontend

- Build REST API (FastAPI) with endpoints: `/index`, `/search`, `/healthz`
- Serve results as JSON with paths, labels, and scores
- Add simple React dashboard for drag-and-drop query images

### 💾 Storage Strategy

- Standardize avatars/logos as WebP (quality 75–80)
- Save source URL alongside each image in JSON metadata
- Add cold storage option (S3/Backblaze) with signed URL metadata

### 🛡️ ProfilePuller Reliability

- Add `--js` mode (optional headless fetch via Playwright)
- Strengthen filename normalization and hashing for deduplication

### 🔒 Privacy & Safety

- Add `--consent-only` mode (skip non-consented URLs)
- Add NSFW / non-face filter before indexing
- Add retention policy (`--purge-after Ndays`)

### 🧑‍💻 Dev Experience

- Add missing Makefile targets (`build`, `api`, `test`) to round out developer shortcuts

## Completed

- Index builder supports a `--metric {cosine,l2}` flag and persists the selected metric together with the embedding dimensionality in `faiss_meta.json`.
- Avatar downloads implement retry and exponential backoff handling for HTTP 429 responses.
- A `FAKE_EMB=1` mode is available to bypass `face_recognition` and return deterministic test embeddings.
- Pytest coverage includes suites for the embedding helpers, IO helpers, and CLI `--help` smoke tests.
- GitHub Actions CI runs on Ubuntu and Windows with Python 3.10, formatting via Black/Ruff and executing pytest.
- Ruff and Black formatting settings are defined in `pyproject.toml`.
- The Makefile exposes `setup`, `build-index`, and `query` shortcuts.
