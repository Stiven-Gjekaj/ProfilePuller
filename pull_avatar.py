#!/usr/bin/env python3
import argparse, concurrent.futures as futures, os, re, sys, time
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup

DEFAULT_UA = "Mozilla/5.0 (compatible; avatar-batch-fetcher/1.1)"
SESSION = requests.Session()

def guess_ext_from_ctype(ctype: str) -> str:
    ctype = (ctype or "").lower()
    if "png" in ctype: return ".png"
    if "webp" in ctype: return ".webp"
    if "gif" in ctype: return ".gif"
    if "jpeg" in ctype or "jpg" in ctype: return ".jpg"
    return ".jpg"

def sanitize_name(name: str) -> str:
    name = re.sub(r"[^\w\.-]+", "_", name.strip())
    name = name.strip("._")
    return name or "avatar"

def save_image(img_url: str, outdir: str, headers: dict, timeout: int, basename_hint: str = "") -> str:
    r = SESSION.get(img_url, headers=headers, timeout=timeout, stream=True, allow_redirects=True)
    r.raise_for_status()
    ext = guess_ext_from_ctype(r.headers.get("Content-Type", ""))
    # filename from URL path or hint
    path_part = sanitize_name(os.path.basename(urlparse(img_url).path)) or sanitize_name(basename_hint)
    if not os.path.splitext(path_part)[1]:
        path_part += ext
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, path_part)
    with open(path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    return path

def find_og_image(profile_url: str, headers: dict, timeout: int) -> str | None:
    resp = SESSION.get(profile_url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for key in [
        ("property", "og:image"),
        ("name", "og:image"),
        ("name", "twitter:image"),
        ("property", "twitter:image"),
    ]:
        tag = soup.find("meta", attrs={key[0]: key[1]})
        if tag and tag.get("content"):
            content = tag["content"].strip()
            # Some sites use relative paths
            return urljoin(profile_url, content)
    return None

def is_direct_image_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"))

def process_single_url(url: str, outdir: str, headers: dict, timeout: int, skip_existing: bool, sleep_between: float) -> tuple[str, str | None]:
    """Returns (url, saved_path or error_message)."""
    try:
        url = url.strip()
        if not url:
            return (url, "empty line")

        # quick path: direct image URL
        if is_direct_image_url(url):
            dest_name = sanitize_name(os.path.basename(urlparse(url).path))
            dest_path = os.path.join(outdir, dest_name)
            if skip_existing and os.path.exists(dest_path):
                return (url, f"skipped (exists: {dest_path})")
            saved = save_image(url, outdir, headers, timeout)
            if sleep_between > 0:
                time.sleep(sleep_between)
            return (url, saved)

        # general page: look for og:image/twitter:image
        img = find_og_image(url, headers, timeout)
        if not img:
            return (url, "no og:image / twitter:image found (page may be private, JS-rendered, or blocked)")

        # derive a helpful basename hint from host + maybe last path part
        hint = sanitize_name((urlparse(url).netloc or "avatar") + "_" + (os.path.basename(urlparse(url).path) or ""))
        dest_name = hint if os.path.splitext(hint)[1] else hint + os.path.splitext(urlparse(img).path)[1]
        dest_path = os.path.join(outdir, dest_name)
        if skip_existing and os.path.exists(dest_path):
            return (url, f"skipped (exists: {dest_path})")

        saved = save_image(img, outdir, headers, timeout, basename_hint=hint)
        if sleep_between > 0:
            time.sleep(sleep_between)
        return (url, saved)
    except requests.HTTPError as e:
        return (url, f"HTTP error: {e}")
    except requests.RequestException as e:
        return (url, f"network error: {e}")
    except Exception as e:
        return (url, f"error: {e}")

def read_urls_from_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines()]
    # dedupe while preserving order
    seen, out = set(), []
    for ln in lines:
        if ln and not ln.startswith("#") and ln not in seen:
            seen.add(ln)
            out.append(ln)
    return out

def main():
    ap = argparse.ArgumentParser(description="Fetch profile pictures via public og:image/twitter:image (no login).")
    ap.add_argument("targets", nargs="*", help="One or more profile URLs (ignored if --from-file is used).")
    ap.add_argument("--from-file", help="Path to .txt containing profile URLs (one per line).")
    ap.add_argument("--out", default="avatars", help="Output directory (default: avatars)")
    ap.add_argument("--concurrency", type=int, default=6, help="Parallel workers (default: 6)")
    ap.add_argument("--timeout", type=int, default=15, help="HTTP timeout seconds (default: 15)")
    ap.add_argument("--limit", type=int, help="Only process the first N URLs")
    ap.add_argument("--user-agent", default=DEFAULT_UA, help="Custom User-Agent string")
    ap.add_argument("--skip-existing", action="store_true", help="Skip downloads if target file already exists")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between downloads per worker (politeness)")
    args = ap.parse_args()

    headers = {"User-Agent": args.user_agent}

    urls = []
    if args.from_file:
        if not os.path.exists(args.from_file):
            sys.exit(f"missing file: {args.from_file}")
        urls.extend(read_urls_from_file(args.from_file))
    urls.extend(args.targets)

    if args.limit:
        urls = urls[:args.limit]

    if not urls:
        sys.exit("No URLs provided. Pass targets or --from-file file.txt")

    os.makedirs(args.out, exist_ok=True)

    successes, failures = 0, 0
    results = []
    with futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        jobs = [ex.submit(process_single_url, u, args.out, headers, args.timeout, args.skip_existing, args.sleep) for u in urls]
        for job in futures.as_completed(jobs):
            url, outcome = job.result()
            if outcome and os.path.exists(outcome):
                successes += 1
                print(f"[OK] {url} -> {outcome}")
            else:
                failures += 1
                print(f"[!!] {url} -> {outcome}")
            results.append((url, outcome))

    print("\n=== Summary ===")
    print(f"Total: {len(urls)} | Saved: {successes} | Failed/Skipped: {failures}")

if __name__ == "__main__":
    main()