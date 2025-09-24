#!/usr/bin/env python3
"""
Crawl any single domain to discover random profile URLs and save them.
- Async crawler with polite throttling
- Playwright fallback for JS-rendered pages
- Optional `/post/` discovery alongside user profiles
- CLI with argparse
- Resumable via JSON checkpoint

Examples:
  python scripts/random_profile_collector.py \
      --seeds https://fiber.al/ \
      --out profiles.txt \
      --max-profiles 200
  python scripts/random_profile_collector.py --resume --state crawl_state.json
  python scripts/random_profile_collector.py --use-playwright
  python scripts/random_profile_collector.py --include-posts --max-posts 100

NOTE: Respect site ToS.
"""

import argparse
import asyncio
import base64
import binascii
import json
import os
import random
import re
import sys
import time
import traceback
from urllib.parse import unquote, urljoin, urlparse
import urllib.robotparser as robotparser

import aiohttp
import async_timeout
from bs4 import BeautifulSoup

# Optional Playwright import deferred until needed
try:
    from playwright.async_api import async_playwright

    HAVE_PLAYWRIGHT = True
except Exception:
    HAVE_PLAYWRIGHT = False

DEFAULT_UA = "Mozilla/5.0 (compatible; RandomProfileCollector/1.1; +https://example.com/bot)"

PROFILE_PREFIX = "/profile/"
POST_REGEX = re.compile(r"/post/[A-Za-z0-9_-]{6,}/?", re.IGNORECASE)


def _decode_base64_slug(slug: str) -> dict | None:
    slug = slug.strip()
    if not slug:
        return None
    # Normalize to standard Base64 with padding
    normalized = slug
    padding = (-len(normalized)) % 4
    if padding:
        normalized += "=" * padding
    try:
        decoded = base64.urlsafe_b64decode(normalized)
    except (binascii.Error, ValueError):
        return None
    try:
        return json.loads(decoded.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _looks_like_profile_payload(payload: dict | None) -> bool:
    if not isinstance(payload, dict):
        return False
    required = {"_id", "name"}
    return required.issubset(payload.keys())


def normalize(url: str) -> str:
    p = urlparse(url)
    # Remove fragments and normalize trailing slash
    p = p._replace(fragment="")
    s = p.geturl()
    if s.endswith("/") and len(p.path) > 1:
        s = s[:-1]
    return s


def same_domain(a: str, b: str) -> bool:
    return urlparse(a).netloc == urlparse(b).netloc


def link_is_profile(url: str) -> bool:
    path = urlparse(url).path
    if not path.startswith(PROFILE_PREFIX):
        return False
    slug = unquote(path[len(PROFILE_PREFIX) :])
    if not slug:
        return False
    payload = _decode_base64_slug(slug)
    return _looks_like_profile_payload(payload)


def link_is_post(url: str) -> bool:
    return bool(POST_REGEX.fullmatch(unquote(urlparse(url).path)))


def extract_links(html: str, base_url: str) -> set[str]:
    soup = BeautifulSoup(html, "html.parser")
    found = set()
    for a in soup.select("a[href]"):
        href = a["href"].strip()
        if href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        found.add(urljoin(base_url, href))
    return found


def load_state(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_state(path: str, state: dict):
    if not path:
        return
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


async def fetch_http(session: aiohttp.ClientSession, url: str, timeout_s: float) -> str | None:
    try:
        async with async_timeout.timeout(timeout_s):
            async with session.get(url, allow_redirects=True) as resp:
                if resp.status == 200 and "text/html" in (resp.headers.get("Content-Type") or ""):
                    return await resp.text()
                # Soft-allow HTML even if content-type missing—some sites are sloppy
                if resp.status == 200:
                    text = await resp.text()
                    if "<html" in text.lower():
                        return text
                return None
    except Exception:
        return None


async def fetch_with_playwright(url: str, timeout_ms: int = 15000) -> str | None:
    if not HAVE_PLAYWRIGHT:
        return None
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            ctx = await browser.new_context(user_agent=DEFAULT_UA)
            page = await ctx.new_page()
            await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            html = await page.content()
            await ctx.close()
            await browser.close()
            return html
    except Exception:
        return None


async def polite_sleep(min_delay: float, max_jitter: float):
    await asyncio.sleep(min_delay + random.random() * max_jitter)


async def crawl(args):
    random.seed(args.seed if args.seed is not None else None)

    # Prepare robots.txt
    parsed_domain = urlparse(args.seeds[0]).netloc if args.seeds else urlparse(args.domain).netloc
    robots_url = f"https://{parsed_domain}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        pass  # if robots can’t be read, fall back to allow unless --obey-robots is set

    # Load or init state
    if args.resume:
        st = load_state(args.state) or {}
    else:
        st = {}

    discovered_profiles = set(st.get("discovered_profiles", []))
    discovered_posts = set(st.get("discovered_posts", []))
    visited = set(st.get("visited", []))
    to_visit = list(st.get("to_visit", []))

    if not to_visit:
        seeds = [normalize(s) for s in args.seeds] if args.seeds else [normalize(args.domain)]
        to_visit.extend(seeds)

    # session setup
    conn = aiohttp.TCPConnector(limit=args.max_concurrency)
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    headers = {"User-Agent": args.user_agent or DEFAULT_UA, "Accept-Language": "en-US,en;q=0.9"}

    async with aiohttp.ClientSession(connector=conn, timeout=timeout, headers=headers) as session:
        pages_fetched = 0

        def needs_more_results() -> bool:
            if len(discovered_profiles) < args.max_profiles:
                return True
            if args.include_posts and len(discovered_posts) < args.max_posts:
                return True
            return False

        while to_visit and needs_more_results() and pages_fetched < args.max_pages:

            # pick random URL to add randomness to traversal
            url = normalize(to_visit.pop(random.randrange(len(to_visit))))
            if url in visited:
                continue
            if args.domain and not same_domain(url, args.domain):
                continue

            # robots check
            if args.obey_robots and not rp.can_fetch(headers["User-Agent"], url):
                visited.add(url)
                continue

            # polite delay
            await polite_sleep(args.delay, args.jitter)

            html = await fetch_http(session, url, args.request_timeout)

            # Playwright fallback if requested or if empty HTML
            if args.use_playwright and (
                not html
                or (args.playwright_on_weak and (html.strip() == "" or "<body" not in html.lower()))
            ):
                html = await fetch_with_playwright(url)

            visited.add(url)
            if not html:
                continue

            pages_fetched += 1

            # extract links, shuffle to randomize queue growth
            links = list(extract_links(html, url))
            random.shuffle(links)

            for lk in links:
                lk_n = normalize(lk)
                if args.domain and not same_domain(lk_n, args.domain):
                    continue

                if link_is_profile(lk_n):
                    if len(discovered_profiles) < args.max_profiles:
                        discovered_profiles.add(lk_n)
                    continue

                if args.include_posts and link_is_post(lk_n):
                    if len(discovered_posts) < args.max_posts:
                        discovered_posts.add(lk_n)
                    continue

                # queue for crawling if not seen
                if lk_n not in visited:
                    to_visit.append(lk_n)

            # checkpoint periodically
            if pages_fetched % max(1, args.checkpoint_every) == 0:
                state_obj = {
                    "discovered_profiles": sorted(discovered_profiles),
                    "discovered_posts": sorted(discovered_posts),
                    "visited": list(visited),
                    "to_visit": to_visit,
                    "timestamp": int(time.time()),
                }
                save_state(args.state, state_obj)

    # Final save
    final_state = {
        "discovered_profiles": sorted(discovered_profiles),
        "discovered_posts": sorted(discovered_posts),
        "visited": list(visited),
        "to_visit": to_visit,
        "timestamp": int(time.time()),
    }
    save_state(args.state, final_state)

    # Write output file (truncate to limits)
    out_profiles = sorted(discovered_profiles)
    random.shuffle(out_profiles)
    out_profiles = out_profiles[: args.max_profiles]

    out_posts: list[str] = []
    if args.include_posts:
        out_posts = sorted(discovered_posts)
        random.shuffle(out_posts)
        out_posts = out_posts[: args.max_posts]

    out = out_profiles + out_posts
    random.shuffle(out)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for u in out:
            f.write(u + "\n")

    print(
        f"[done] Saved {len(out_profiles)} profiles"
        + (f" and {len(out_posts)} posts" if out_posts else "")
        + f" to {args.out}"
    )
    if args.state:
        print(f"[state] Checkpoint saved at {args.state}")
    if args.use_playwright and not HAVE_PLAYWRIGHT:
        print(
            "[note] --use-playwright was set but Playwright is not installed. "
            "See README/requirements.",
            file=sys.stderr,
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Discover random profile links on a domain.")
    parser.add_argument(
        "--seeds",
        nargs="*",
        default=["https://fiber.al/"],
        help="Seed URLs to start crawling from.",
    )
    parser.add_argument(
        "--domain",
        default="https://fiber.al/",
        help="Root domain to restrict crawling (scheme+host).",
    )
    parser.add_argument(
        "--out",
        default="profiles.txt",
        help="Output file for profile URLs.",
    )
    parser.add_argument(
        "--state",
        default="crawl_state.json",
        help="Path for checkpoint state JSON.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint state.",
    )
    parser.add_argument(
        "--max-profiles",
        type=int,
        default=200,
        help="Max number of verified profile URLs to collect.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1000,
        help="Safety cap on total pages fetched.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=6,
        help="Max concurrent HTTP connections.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.6,
        help="Base polite delay (seconds) between requests.",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.5,
        help="Extra random jitter added to delay (seconds).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=15.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Persist crawl state after this many pages (default: 25).",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_UA,
        help="Custom User-Agent string.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--obey-robots",
        action="store_true",
        help="Respect robots.txt (recommended).",
    )

    parser.add_argument(
        "--include-posts",
        action="store_true",
        help="Also collect /post/... URLs in addition to profiles.",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=None,
        help="Max number of post URLs to save (defaults to --max-profiles).",
    )

    # Playwright options
    parser.add_argument(
        "--use-playwright",
        action="store_true",
        help="Enable Playwright fallback for JS-rendered pages.",
    )
    parser.add_argument(
        "--playwright-on-weak",
        action="store_true",
        help=(
            "Only use Playwright when HTML looks empty/weak. "
            "If not set, never auto-fallback unless --use-playwright."
        ),
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Ensure domain normalized to scheme+host root
    d = urlparse(args.domain)
    if not d.scheme or not d.netloc:
        print("--domain must include scheme and host, e.g., https://fiber.al/", file=sys.stderr)
        sys.exit(2)
    args.domain = f"{d.scheme}://{d.netloc}"

    if args.max_posts is None:
        args.max_posts = args.max_profiles

    # Basic courtesy warning
    print("Heads-up: Check robots.txt and ToS. Crawl gently. 🍃")

    try:
        asyncio.run(crawl(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
    except Exception as e:
        print("[fatal]", e, file=sys.stderr)
        traceback.print_exc()


if __name__ == "__main__":
    main()
