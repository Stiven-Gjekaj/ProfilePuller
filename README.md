# ProfilePuller

ProfilePuller fetches public avatars, embeds them with `face_recognition`, and lets you search
against a FAISS index from the command line or a webcam feed. The repository now ships with a
small but complete toolchain, logging, tests, and CI so you can reproduce runs across
macOS/Linux/Windows.

⚠️ **Always respect each website's Terms of Service.** These scripts only work with public pages
and metadata such as `og:image` or `twitter:image`.

## Quickstart

```bash
# Create a virtual environment and install dependencies
make setup                # Windows PowerShell: make PYTHON=python setup

# (Optional) fetch sample avatars listed in profiles.txt
python scripts/pull_avatar.py --from-file profiles.txt --skip-existing

# Build the FAISS index and metadata
make build-index

# Try a query (searches faces/queries/ if any images exist)
make query
```

### Manual environment setup

If `make` is unavailable you can run the equivalent commands manually:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Directory layout

```
.
├── faces/
│   ├── known/      # place reference images here (subfolders = labels)
│   └── queries/    # place query photos here
├── scripts/
│   ├── pull_avatar.py
│   ├── build_index.py
│   ├── query.py
│   ├── live_search.py
│   └── random_profile_collector.py
└── tests/
    └── ...
```

## Avatar collection (`pull_avatar.py`)

```
python scripts/pull_avatar.py URL [URL ...]
python scripts/pull_avatar.py --from-file profiles.txt --out faces/known --skip-existing
```

Key flags:

- `--retries` – retry failed requests (default `2`).
- `--sleep` – pause between downloads per worker.
- `--verify-ssl / --no-verify-ssl` – control TLS validation (on by default).
- `--skip-existing` – make downloads idempotent.
- `--verbose / -v` – print full error tracebacks.

The script uses Rich to colourise `[OK]` and `[!!]` lines while preserving the original output
format. HTTP 429 responses trigger a polite exponential backoff.

## Random profile discovery (`random_profile_collector.py`)

```bash
python scripts/random_profile_collector.py --domain https://fiber.al/ --max-profiles 200 \
    --out random_profiles.txt --obey-robots
```

- Crawls a single domain, following in-domain links and heuristically recognising profile pages.
- Supports multiple seed URLs and resumes work via `--state` checkpoint files.
- Polite by design: adjustable delay/jitter, concurrency cap, robots.txt respect, and optional
  Playwright fallback for JavaScript-heavy pages (`--use-playwright`).
- Outputs a shuffled list of profile URLs, truncated to `--max-profiles`, and saves crawl state for
  later resumption.
- Validates `/profile/` slugs by decoding their payloads so the output only includes real user pages.
- Pass `--include-posts` to collect `/post/...` URLs (with an optional `--max-posts` limit) alongside
  profiles.

## Embedding & indexing

```
python scripts/build_index.py --known faces/known --out-index faiss_index.bin --out-meta faiss_meta.json
```

Highlights:

- Supports `hog` (CPU) and `cnn` (GPU) models from `face_recognition`.
- Two metrics: `cosine` (default, uses `IndexFlatIP`) and `l2` (uses `IndexFlatL2`).
- Progress bars via `tqdm`, verbose logging via `-v`.
- Metadata JSON stores metric, dimensionality, count, timestamp, and per-face records.
- Gracefully writes an empty index when no faces are found (helpful for first runs).

## Querying the index

```
python scripts/query.py path/to/query.jpg --k 5
```

- Automatically normalises embeddings when the index metric is cosine.
- Default thresholds: cosine `0.65`, L2 `0.60` (squared distances are converted back to L2).
- Renders results in a Rich table and never exits with an error code for “no match”.

## Live search (webcam)

```
python scripts/live_search.py --cam 0 --k 3
```

- Draws bounding boxes, match scores, and labels on the live feed.
- Threshold defaults to `0.65` (or `0.60` for L2 indices).
- Press `q` to quit. The script exits gracefully on `Ctrl+C`.

> ℹ️ Live search requires a working webcam. The CI pipeline skips this step; it is intended for
> manual use only.

## Fake embedding mode for tests

Set `FAKE_EMB=1` to avoid loading `face_recognition` and `dlib`. The embedding utilities will return
stable, deterministic 128-d vectors derived from file paths. This mode powers the unit tests and is
helpful on machines where building `dlib` is inconvenient.

```
FAKE_EMB=1 python scripts/build_index.py --known faces/known
```

## Troubleshooting

- **Installing `face_recognition`** – it depends on `dlib`. Wheels are available for most Linux and
  macOS environments. On Windows you may need a Visual Studio build chain or the prebuilt wheels
  from the [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#dlib). If that
  is not feasible, consider swapping in `deepface` or `insightface` (you can add an alternate code
  path behind a flag) while keeping the default behaviour intact.
- **SSL issues** – pass `--no-verify-ssl` to `pull_avatar.py` when scraping sites with unusual TLS
  setups.
- **Empty index** – if `scripts/build_index.py` finds no faces it still writes empty artifacts and
  prints a warning. Add labelled photos under `faces/known/` and rerun.

## Testing & CI

```
FAKE_EMB=1 pytest
ruff check .
black --check .
```

GitHub Actions runs the same checks on Ubuntu and Windows for every push or pull request. CI also
uses `FAKE_EMB=1`, keeping the pipeline lightweight while the real code paths remain available.

## Changelog

- **v1.1** – Complete tooling overhaul: added Rich logging, retries, progress bars, fake embedding
  mode, FAISS metadata, CLI polish, Makefile helpers, tests, and cross-platform CI.
- **v1.0** – Initial script for downloading avatars from `og:image` / `twitter:image` metadata.
