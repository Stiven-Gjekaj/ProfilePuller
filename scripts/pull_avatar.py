#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Iterable
import concurrent.futures as futures
from pathlib import Path
import re
import sys
import time
import traceback
import unicodedata
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import requests
from rich.console import Console

console = Console()
DEFAULT_UA = "Mozilla/5.0 (compatible; avatar-batch-fetcher/1.1)"


def sanitize_name(name: str) -> str:
    """Return a filesystem-safe filename component."""
    normalized = unicodedata.normalize("NFKC", name)
    normalized = normalized.replace("/", "_").replace("\\", "_")
    normalized = re.sub(r"\s+", "_", normalized)
    sanitized = re.sub(r"[^0-9A-Za-z._-]", "_", normalized)
    sanitized = sanitized.strip("._")
    return sanitized or "face"


def guess_ext_from_ctype(ctype: str) -> str:
    ctype = (ctype or "").lower()
    if "png" in ctype:
        return ".png"
    if "webp" in ctype:
        return ".webp"
    if "gif" in ctype:
        return ".gif"
    if "jpeg" in ctype or "jpg" in ctype:
        return ".jpg"
    return ".jpg"


def request_with_retries(
    session: requests.Session,
    url: str,
    *,
    headers: dict[str, str],
    timeout: int,
    verify_ssl: bool,
    retries: int,
    sleep_base: float,
    verbose: bool,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = session.get(
                url,
                headers=headers,
                timeout=timeout,
                allow_redirects=True,
                stream=True,
                verify=verify_ssl,
            )
            if response.status_code == 429:
                if attempt < retries:
                    wait = max(sleep_base, 1.0) * (attempt + 1)
                    console.print(f"[yellow]429 Too Many Requests from {url}; sleeping {wait:.1f}s")
                    time.sleep(wait)
                    continue
            response.raise_for_status()
            return response
        except requests.HTTPError as exc:
            last_exc = exc
            status = getattr(exc.response, "status_code", "?")
            if status == 429 and attempt < retries:
                wait = max(sleep_base, 1.0) * (attempt + 1)
                console.print(f"[yellow]Retrying {url} after 429 in {wait:.1f}s")
                time.sleep(wait)
                continue
            if verbose:
                console.print(traceback.format_exc())
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries:
                wait = max(sleep_base, 1.0) * (attempt + 1)
                console.print(f"[yellow]Network issue fetching {url}; retrying in {wait:.1f}s")
                time.sleep(wait)
                continue
            if verbose:
                console.print(traceback.format_exc())
        break
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed to fetch {url}")


def save_image(
    session: requests.Session,
    img_url: str,
    outdir: Path,
    headers: dict[str, str],
    timeout: int,
    verify_ssl: bool,
    retries: int,
    sleep_base: float,
    basename_hint: str = "",
    verbose: bool = False,
) -> Path:
    response = request_with_retries(
        session,
        img_url,
        headers=headers,
        timeout=timeout,
        verify_ssl=verify_ssl,
        retries=retries,
        sleep_base=sleep_base,
        verbose=verbose,
    )
    ext = guess_ext_from_ctype(response.headers.get("Content-Type", ""))
    path_part = sanitize_name(Path(urlparse(img_url).path).name or basename_hint)
    if not Path(path_part).suffix:
        path_part = f"{path_part}{ext}"
    path = outdir / path_part
    outdir.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        for chunk in response.iter_content(chunk_size=8192):
            fh.write(chunk)
    response.close()
    return path


def find_og_image(
    session: requests.Session,
    profile_url: str,
    headers: dict[str, str],
    timeout: int,
    verify_ssl: bool,
    retries: int,
    sleep_base: float,
    verbose: bool,
) -> str | None:
    response = request_with_retries(
        session,
        profile_url,
        headers=headers,
        timeout=timeout,
        verify_ssl=verify_ssl,
        retries=retries,
        sleep_base=sleep_base,
        verbose=verbose,
    )
    soup = BeautifulSoup(response.text, "html.parser")
    for attr, value in [
        ("property", "og:image"),
        ("name", "og:image"),
        ("name", "twitter:image"),
        ("property", "twitter:image"),
    ]:
        tag = soup.find("meta", attrs={attr: value})
        if tag and tag.get("content"):
            content = tag["content"].strip()
            return urljoin(profile_url, content)
    return None


def is_direct_image_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"))


def process_single_url(
    url: str,
    outdir: Path,
    headers: dict[str, str],
    timeout: int,
    skip_existing: bool,
    sleep_between: float,
    verify_ssl: bool,
    retries: int,
    verbose: bool,
) -> tuple[str, str | None, bool]:
    """Return (url, message_or_path, success)."""
    session = requests.Session()
    try:
        url = url.strip()
        if not url:
            return url, "empty line", False

        if is_direct_image_url(url):
            filename = sanitize_name(Path(urlparse(url).path).name)
            if not Path(filename).suffix:
                filename += Path(urlparse(url).path).suffix or ".jpg"
            dest_path = outdir / filename
            if skip_existing and dest_path.exists():
                return url, f"skipped (exists: {dest_path})", False
            saved_path = save_image(
                session,
                url,
                outdir,
                headers,
                timeout,
                verify_ssl,
                retries,
                sleep_between,
                basename_hint=filename,
                verbose=verbose,
            )
            if sleep_between > 0:
                time.sleep(sleep_between)
            return url, str(saved_path), True

        img_url = find_og_image(
            session,
            url,
            headers,
            timeout,
            verify_ssl,
            retries,
            sleep_between,
            verbose,
        )
        if not img_url:
            return url, "no og:image / twitter:image found", False

        parsed = urlparse(url)
        hint = sanitize_name(f"{parsed.netloc}_{Path(parsed.path).stem}")
        if not Path(hint).suffix:
            ext = Path(urlparse(img_url).path).suffix or ""
            hint = f"{hint}{ext}"
        dest_path = outdir / hint
        if skip_existing and dest_path.exists():
            return url, f"skipped (exists: {dest_path})", False

        saved_path = save_image(
            session,
            img_url,
            outdir,
            headers,
            timeout,
            verify_ssl,
            retries,
            sleep_between,
            basename_hint=hint,
            verbose=verbose,
        )
        if sleep_between > 0:
            time.sleep(sleep_between)
        return url, str(saved_path), True
    except requests.HTTPError as exc:
        message = f"HTTP error: {exc}"
        if verbose:
            message += f"\n{traceback.format_exc()}"
        return url, message, False
    except requests.RequestException as exc:
        message = f"network error: {exc}"
        if verbose:
            message += f"\n{traceback.format_exc()}"
        return url, message, False
    except Exception as exc:  # pragma: no cover - defensive
        message = f"error: {exc}"
        if verbose:
            message += f"\n{traceback.format_exc()}"
        return url, message, False
    finally:
        session.close()


def read_urls_from_file(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    seen = set()
    urls: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line not in seen:
            seen.add(line)
            urls.append(line)
    return urls


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch profile pictures via public og:image/twitter:image (no login)."
    )
    parser.add_argument("targets", nargs="*", help="Profile URLs (ignored if --from-file is used)")
    parser.add_argument("--from-file", type=Path, help="Path to .txt containing profile URLs")
    parser.add_argument("--out", type=Path, default=Path("faces/known"), help="Output directory")
    parser.add_argument("--concurrency", type=int, default=6, help="Parallel workers")
    parser.add_argument("--timeout", type=int, default=15, help="HTTP timeout seconds")
    parser.add_argument("--limit", type=int, help="Only process the first N URLs")
    parser.add_argument("--user-agent", default=DEFAULT_UA, help="Custom User-Agent string")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing files")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between downloads")
    parser.add_argument("--retries", type=int, default=2, help="Number of retries on failure")
    parser.add_argument(
        "--verify-ssl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify SSL certificates (use --no-verify-ssl to disable)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    headers = {"User-Agent": args.user_agent}

    urls: list[str] = []
    if args.from_file:
        if not args.from_file.exists():
            console.print(f"[red]missing file: {args.from_file}")
            return 1
        urls.extend(read_urls_from_file(args.from_file))
    urls.extend(args.targets)

    if args.limit is not None:
        urls = urls[: args.limit]

    if not urls:
        console.print("[red]No URLs provided. Pass targets or --from-file file.txt")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    successes = 0
    failures = 0
    try:
        with futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            jobs = [
                executor.submit(
                    process_single_url,
                    url,
                    args.out,
                    headers,
                    args.timeout,
                    args.skip_existing,
                    args.sleep,
                    args.verify_ssl,
                    args.retries,
                    args.verbose,
                )
                for url in urls
            ]
            for job in futures.as_completed(jobs):
                try:
                    url, outcome, ok = job.result()
                except KeyboardInterrupt:  # pragma: no cover - handled above
                    raise
                if ok and outcome:
                    successes += 1
                    console.print(f"[OK] {url} -> {outcome}", style="green")
                else:
                    failures += 1
                    console.print(f"[!!] {url} -> {outcome}", style="bold red")
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user.")
        return 130

    console.print("\n[bold]=== Summary ===")
    console.print(f"Total: {len(urls)} | Saved: {successes} | Failed/Skipped: {failures}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
