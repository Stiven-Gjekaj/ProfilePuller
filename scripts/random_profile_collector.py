#!/usr/bin/env python3
"""
Crawl any single domain to discover random profile URLs and save them.
- Async crawler with polite throttling
- Playwright fallback for JS-rendered pages
- CLI with argparse
- Resumable via JSON checkpoint

Examples:
  python random_profile_collector.py --seeds https://fiber.al/ --out random_profiles.txt --max-profiles 200
  python random_profile_collector.py --resume --state crawl_state.json
  python random_profile_collector.py --use-playwright

NOTE: Respect site ToS.
"""

import asyncio
import aiohttp
import async_timeout
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import argparse
import json
import os
import random
import re
import sys
import time
import traceback
import urllib.robotparser as robotparser

# Optional Playwright import deferred until needed
try:
    from playwright.async_api import async_playwright
    HAVE_PLAYWRIGHT = True
except Exception:
    HAVE_PLAYWRIGHT = False

DEFAULT_UA = "Mozilla/5.0 (compatible; RandomProfileCollector/1.1; +https://example.com/bot)"

PROFILE_REGEX = re.compile(r"/profile/[A-Za-z0-9_\-\.=]+", re.IGNORECASE)

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
    return bool(PROFILE_REGEX.search(urlparse(url).path))

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
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(path: str, state: dict):
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

        while (to_visit and
               len(discovered_profiles) < args.max_profiles and
               pages_fetched < args.max_pages):

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
            if (not html or (args.playwright_on_weak and (html.strip() == "" or "<body" not in html.lower()))) and args.use_playwright:
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
                    discovered_profiles.add(lk_n)
                    if len(discovered_profiles) >= args.max_profiles:
                        break
                else:
                    # queue for crawling if not seen
                    if lk_n not in visited:
                        to_visit.append(lk_n)

            # checkpoint periodically
            if pages_fetched % max(1, args.checkpoint_every) == 0:
                state_obj = {
                    "discovered_profiles": sorted(discovered_profiles),
                    "visited": list(visited),
                    "to_visit": to_visit,
                    "timestamp": int(time.time()),
                }
                save_state(args.state, state_obj)

    # Final save
    final_state = {
        "discovered_profiles": sorted(discovered_profiles),
        "visited": list(visited),
        "to_visit": to_visit,
        "timestamp": int(time.time()),
    }
    save_state(args.state, final_state)

    # Write output file (truncate to max_profiles)
    out = sorted(discovered_profiles)
    random.shuffle(out)
    out = out[: args.max_profiles]

    with open(args.out, "w", encoding="utf-8") as f:
        for u in out:
            f.write(u + "\n")

    print(f"[done] Saved {len(out)} profile URLs to {args.out}")
    print(f"[state] Checkpoint saved at {args.state}")
    if args.use_playwright and not HAVE_PLAYWRIGHT:
        print("[note] --use-playwright was set but Playwright is not installed. See README/requirements.", file=sys.stderr)


def build_parser():
    p = argparse.ArgumentParser(description="Discover random profile links on a domain.")
    p.add_argument("--seeds", nargs="*", default=["https://fiber.al/"], help="Seed URLs to start crawling from.")
    p.add_argument("--domain", default="https://fiber.al/", help="Root domain to restrict crawling (scheme+host).")
    p.add_argument("--out", default="random_profiles.txt", help="Output file for profile URLs.")
    p.add_argument("--state", default="crawl_state.json", help="Path for checkpoint state JSON.")
    p.add_argument("--resume", action="store_true", help="Resume from existing checkpoint state.")
    p.add_argument("--max-profiles", type=int, default=200, help="Max number of profile URLs to collect.")
    p.add_argument("--max-pages", type=int, default=1000, help="Safety cap on total pages fetched.")
    p.add_argument("--max-concurrency", type=int, default=6, help="Max concurrent HTTP connections.")
    p.add_argument("--delay", type=float, default=0.6, help="Base polite delay (seconds) between requests.")
    p.add_argument("--jitter", type=float, default=0.5, help="Extra random jitter added to delay (seconds).")
    p.add_argument("--request-timeout", type=float, default=15.0, help="Per-request timeout in seconds.")
    p.add_argument("--user-agent", default=DEFAULT_UA, help="Custom User-Agent string.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    p.add_argument("--obey-robots", action="store_true", help="Respect robots.txt (recommended).")

    # Playwright options
    p.add_argument("--use-playwright", action="store_true", help="Enable Playwright fallback for JS-rendered pages.")
    p.add_argument("--playwright-on-weak", action="store_true",
                   help="Only use Playwright when HTML looks empty/weak. If not set, never auto-fallback unless --use-playwright.")
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Ensure domain normalized to scheme+host root
    d = urlparse(args.domain)
    if not d.scheme or not d.netloc:
        print("--domain must include scheme and host, e.g., https://fiber.al/", file=sys.stderr)
        sys.exit(2)
    args.domain = f"{d.scheme}://{d.netloc}"

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