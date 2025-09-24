#!/usr/bin/env python3
"""Randomly discover profile and post URLs from a single domain.

- Async crawler with polite throttling
- Optional Playwright fallback for JS-rendered pages
- Optional `/post/` discovery alongside user profiles
- Base64 slug decoding and validation for Fiber-style profile URLs
- Resumable via JSON checkpoints
- Experimental random slug generator for profiles and posts

NOTE: Respect each site's Terms of Service before scraping.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
from collections.abc import Iterable
import json
import os
import random
import re
import secrets
import sys
import time
import traceback
from typing import Any
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

DEFAULT_UA = "Mozilla/5.0 (compatible; RandomProfileCollector/1.2; +https://example.com/bot)"

PROFILE_PREFIX = "/profile/"
POST_PREFIX = "/post/"
POST_REGEX = re.compile(r"/post/[A-Za-z0-9_-]{6,}/?", re.IGNORECASE)

SOFT_404_PATTERNS = [
    "page not found",
    "not found",
    "error 404",
    "404 not found",
    "doesn't exist",
    "does not exist",
]

FIRST_NAMES = [
    "Avery",
    "Blair",
    "Casey",
    "Devon",
    "Elliott",
    "Harper",
    "Indigo",
    "Jules",
    "Kai",
    "Lennon",
    "Monroe",
    "Nova",
    "Parker",
    "Quinn",
    "River",
    "Sasha",
    "Taylor",
    "Winter",
]

LAST_NAMES = [
    "Ashford",
    "Bennett",
    "Carter",
    "Dalton",
    "Ellison",
    "Fletcher",
    "Grayson",
    "Hayes",
    "Lennox",
    "Mercer",
    "Peyton",
    "Reeves",
    "Sawyer",
    "Thorne",
    "Whitaker",
    "Willow",
]

DEFAULT_PROFILE_TEMPLATE_SLUG = (
    "eyJpc1ZlcmlmaWVkIjpmYWxzZSwidHlwZSI6Im5vcm1hbCIsIl9pZCI6IjY4YzcxZGU4MGUw"
    "ZWE1YTM0MGJjOWRhMyIsIm5hbWUiOiJUYXNoYWFyIiwic3VybmFtZSI6IldpbGxvdyJ9"
)
DEFAULT_POST_IDS = ["68cd5b39d34660ad4b13dfc3"]
MAX_RANDOM_GENERATION_RETRIES = 12


def _decode_base64_slug(slug: str) -> dict[str, Any] | None:
    slug = slug.strip()
    if not slug:
        return None
    normalized = slug
    padding = (-len(normalized)) % 4
    if padding:
        normalized += "=" * padding
    try:
        decoded = base64.urlsafe_b64decode(normalized)
    except (binascii.Error, ValueError):
        return None
    try:
        data = decoded.decode("utf-8")
    except UnicodeDecodeError:
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _encode_profile_payload(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    encoded = base64.urlsafe_b64encode(data.encode("utf-8")).decode("ascii")
    return encoded.rstrip("=")


def _looks_like_profile_payload(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    return {"_id", "name"}.issubset(payload.keys())


def _extract_profile_slug(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    if value.startswith("http://") or value.startswith("https://"):
        path = urlparse(value).path
        if not path.startswith(PROFILE_PREFIX):
            return None
        slug = unquote(path[len(PROFILE_PREFIX) :])
    else:
        slug = value
    slug = slug.strip("/ ")
    return slug or None


def _extract_post_id_input(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    if value.startswith("http://") or value.startswith("https://"):
        path = urlparse(value).path
        if not path.startswith(POST_PREFIX):
            return None
        candidate = path[len(POST_PREFIX) :]
    else:
        candidate = value
    candidate = candidate.strip("/ ")
    if not candidate:
        return None
    candidate = candidate.split("/")[0]
    if re.fullmatch(r"[0-9a-fA-F]{24}", candidate):
        return candidate.lower()
    return None


def decode_profile_payload_from_url(url: str) -> dict[str, Any] | None:
    path = urlparse(url).path
    if not path.startswith(PROFILE_PREFIX):
        return None
    slug = unquote(path[len(PROFILE_PREFIX) :])
    return _decode_base64_slug(slug)


def extract_post_id(url: str) -> str | None:
    path = urlparse(url).path
    if not path.startswith(POST_PREFIX):
        return None
    candidate = path[len(POST_PREFIX) :].strip("/ ")
    if re.fullmatch(r"[0-9a-fA-F]{24}", candidate):
        return candidate.lower()
    return None


def normalize(url: str) -> str:
    p = urlparse(url)
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
    return _looks_like_profile_payload(_decode_base64_slug(slug))


def link_is_post(url: str) -> bool:
    return bool(POST_REGEX.fullmatch(unquote(urlparse(url).path)))


def extract_links(html: str, base_url: str) -> set[str]:
    soup = BeautifulSoup(html, "html.parser")
    found = set()
    for a in soup.select("a[href]"):
        href = a["href"].strip()
        if (
            href.startswith("#")
            or href.lower().startswith("javascript:")
            or href.startswith("mailto:")
        ):
            continue
        found.add(urljoin(base_url, href))
    return found


def looks_like_soft_404(html: str) -> bool:
    lowered = html.lower()
    return any(pat in lowered for pat in SOFT_404_PATTERNS)


def load_state(path: str) -> dict[str, Any] | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_state(path: str, state: dict[str, Any]):
    if not path:
        return
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _random_object_id(base_id: str | None = None) -> str:
    if base_id and re.fullmatch(r"[0-9a-f]{24}", base_id):
        base_int = int(base_id, 16)
        delta = random.randint(-0x7FFFF, 0x7FFFF)
        candidate = (base_int + delta) % (1 << 96)
        return f"{candidate:024x}"
    return secrets.token_hex(12)


def _ensure_templates(templates: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if templates:
        return templates
    default_payload = _decode_base64_slug(DEFAULT_PROFILE_TEMPLATE_SLUG) or {
        "isVerified": False,
        "type": "normal",
        "_id": "0123456789abcdef01234567",
        "name": "Sample",
        "surname": "User",
    }
    return [default_payload]


def generate_random_profile_url(
    domain: str, templates: list[dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    usable = _ensure_templates(templates)
    template = dict(random.choice(usable))
    template["_id"] = _random_object_id(template.get("_id"))
    template["name"] = random.choice(FIRST_NAMES)
    template["surname"] = random.choice(LAST_NAMES)
    slug = _encode_profile_payload(template)
    return normalize(f"{domain}{PROFILE_PREFIX}{slug}"), template


def generate_random_post_url(domain: str, seeds: Iterable[str]) -> tuple[str, str]:
    seeds_list = list(seeds)
    base = random.choice(seeds_list) if seeds_list else None
    post_id = _random_object_id(base)
    return normalize(f"{domain}{POST_PREFIX}{post_id}"), post_id


def merge_profile_templates(
    existing: Iterable[str], extras: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for url in existing:
        payload = decode_profile_payload_from_url(url)
        if payload:
            out.append(payload)
    out.extend(extras)
    return out


def merge_post_ids(existing: Iterable[str], extras: Iterable[str]) -> set[str]:
    out = {pid for pid in extras if re.fullmatch(r"[0-9a-f]{24}", pid)}
    for url in existing:
        post_id = extract_post_id(url)
        if post_id:
            out.add(post_id)
    return out


def save_checkpoint_if_needed(
    pages_fetched: int,
    args: argparse.Namespace,
    discovered_profiles: set[str],
    discovered_posts: set[str],
    visited: set[str],
    to_visit: list[str],
):
    if pages_fetched % max(1, args.checkpoint_every) != 0:
        return
    state_obj = {
        "discovered_profiles": sorted(discovered_profiles),
        "discovered_posts": sorted(discovered_posts),
        "visited": list(visited),
        "to_visit": to_visit,
        "timestamp": int(time.time()),
    }
    save_state(args.state, state_obj)


def state_snapshot(
    discovered_profiles: set[str],
    discovered_posts: set[str],
    visited: set[str],
    to_visit: list[str],
) -> dict[str, Any]:
    return {
        "discovered_profiles": sorted(discovered_profiles),
        "discovered_posts": sorted(discovered_posts),
        "visited": list(visited),
        "to_visit": to_visit,
        "timestamp": int(time.time()),
    }


def register_profile(url: str, discovered: set[str], templates: list[dict[str, Any]], limit: int):
    if len(discovered) >= limit:
        return
    if url in discovered:
        return
    discovered.add(url)
    payload = decode_profile_payload_from_url(url)
    if payload:
        templates.append(payload)


def register_post(url: str, discovered: set[str], post_ids: set[str], limit: int):
    if len(discovered) >= limit:
        return
    if url in discovered:
        return
    discovered.add(url)
    post_id = extract_post_id(url)
    if post_id:
        post_ids.add(post_id)


def process_page(
    url: str,
    html: str,
    args: argparse.Namespace,
    discovered_profiles: set[str],
    discovered_posts: set[str],
    visited: set[str],
    to_visit: list[str],
    profile_templates: list[dict[str, Any]],
    post_ids: set[str],
):
    links = list(extract_links(html, url))
    random.shuffle(links)
    for lk in links:
        lk_n = normalize(lk)
        if args.domain and not same_domain(lk_n, args.domain):
            continue
        if link_is_profile(lk_n):
            register_profile(lk_n, discovered_profiles, profile_templates, args.max_profiles)
            continue
        if args.include_posts and link_is_post(lk_n):
            register_post(lk_n, discovered_posts, post_ids, args.max_posts)
            continue
        if lk_n not in visited and lk_n not in to_visit:
            to_visit.append(lk_n)


def load_templates_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    for slug_value in args.random_profile_template_inputs:
        payload = _decode_base64_slug(slug_value)
        if payload:
            templates.append(payload)
    return templates


def load_post_ids_from_args(args: argparse.Namespace) -> list[str]:
    ids: list[str] = []
    for candidate in args.random_post_seed_inputs:
        if re.fullmatch(r"[0-9a-f]{24}", candidate):
            ids.append(candidate)
    return ids


async def fetch_http(session: aiohttp.ClientSession, url: str, timeout_s: float) -> str | None:
    try:
        async with async_timeout.timeout(timeout_s):
            async with session.get(url, allow_redirects=True) as resp:
                if resp.status == 200 and "text/html" in (resp.headers.get("Content-Type") or ""):
                    return await resp.text()
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


def should_retry_with_playwright(args: argparse.Namespace, html: str | None) -> bool:
    if not args.use_playwright:
        return False
    if html is None:
        return True
    if not args.playwright_on_weak:
        return False
    stripped = html.strip()
    if not stripped:
        return True
    return "<body" not in stripped.lower()


async def crawl(args: argparse.Namespace):
    random.seed(args.seed if args.seed is not None else None)

    parsed_domain = urlparse(args.seeds[0]).netloc if args.seeds else urlparse(args.domain).netloc
    robots_url = f"https://{parsed_domain}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        pass

    if args.resume:
        st = load_state(args.state) or {}
    else:
        st = {}

    discovered_profiles: set[str] = set(st.get("discovered_profiles", []))
    discovered_posts: set[str] = set(st.get("discovered_posts", []))
    visited: set[str] = set(st.get("visited", []))
    to_visit: list[str] = list(st.get("to_visit", []))

    if not to_visit:
        seeds = [normalize(s) for s in args.seeds] if args.seeds else [normalize(args.domain)]
        to_visit.extend(seeds)

    profile_templates = merge_profile_templates(discovered_profiles, args.random_profile_templates)
    post_ids = merge_post_ids(discovered_posts, args.random_post_ids)
    if not post_ids:
        post_ids.update(DEFAULT_POST_IDS)

    conn = aiohttp.TCPConnector(limit=args.max_concurrency)
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    headers = {"User-Agent": args.user_agent or DEFAULT_UA, "Accept-Language": "en-US,en;q=0.9"}

    async with aiohttp.ClientSession(connector=conn, timeout=timeout, headers=headers) as session:
        pages_fetched = 0
        random_profile_attempts = 0
        random_post_attempts = 0
        tried_random_profiles: set[str] = set()
        tried_random_posts: set[str] = set()

        def needs_more_results() -> bool:
            if len(discovered_profiles) < args.max_profiles:
                return True
            if args.include_posts and len(discovered_posts) < args.max_posts:
                return True
            return False

        while needs_more_results() and pages_fetched < args.max_pages:
            if to_visit:
                url = normalize(to_visit.pop(random.randrange(len(to_visit))))
                if url in visited:
                    continue
                if args.domain and not same_domain(url, args.domain):
                    continue
                if args.obey_robots and not rp.can_fetch(headers["User-Agent"], url):
                    visited.add(url)
                    continue

                await polite_sleep(args.delay, args.jitter)
                html = await fetch_http(session, url, args.request_timeout)
                if should_retry_with_playwright(args, html):
                    html = await fetch_with_playwright(url)

                visited.add(url)
                if not html:
                    continue

                pages_fetched += 1
                process_page(
                    url,
                    html,
                    args,
                    discovered_profiles,
                    discovered_posts,
                    visited,
                    to_visit,
                    profile_templates,
                    post_ids,
                )
                save_checkpoint_if_needed(
                    pages_fetched,
                    args,
                    discovered_profiles,
                    discovered_posts,
                    visited,
                    to_visit,
                )
                continue

            attempted_random = False

            if (
                args.random_profile_attempts
                and random_profile_attempts < args.random_profile_attempts
                and len(discovered_profiles) < args.max_profiles
            ):
                attempted_random = True
                random_profile_attempts += 1
                rand_url = None
                for _ in range(MAX_RANDOM_GENERATION_RETRIES):
                    candidate_url, _ = generate_random_profile_url(args.domain, profile_templates)
                    if candidate_url in visited or candidate_url in tried_random_profiles:
                        continue
                    rand_url = candidate_url
                    break
                if rand_url:
                    tried_random_profiles.add(rand_url)
                    if args.obey_robots and not rp.can_fetch(headers["User-Agent"], rand_url):
                        visited.add(rand_url)
                    else:
                        await polite_sleep(args.delay, args.jitter)
                        html = await fetch_http(session, rand_url, args.request_timeout)
                        if should_retry_with_playwright(args, html):
                            html = await fetch_with_playwright(rand_url)

                        visited.add(rand_url)
                        if html and not looks_like_soft_404(html):
                            register_profile(
                                rand_url,
                                discovered_profiles,
                                profile_templates,
                                args.max_profiles,
                            )
                            pages_fetched += 1
                            process_page(
                                rand_url,
                                html,
                                args,
                                discovered_profiles,
                                discovered_posts,
                                visited,
                                to_visit,
                                profile_templates,
                                post_ids,
                            )
                            save_checkpoint_if_needed(
                                pages_fetched,
                                args,
                                discovered_profiles,
                                discovered_posts,
                                visited,
                                to_visit,
                            )
                            continue
                else:
                    continue

            if (
                args.include_posts
                and args.random_post_attempts
                and random_post_attempts < args.random_post_attempts
                and len(discovered_posts) < args.max_posts
            ):
                attempted_random = True
                random_post_attempts += 1
                rand_url = None
                seed_pool: Iterable[str] = post_ids if post_ids else DEFAULT_POST_IDS
                for _ in range(MAX_RANDOM_GENERATION_RETRIES):
                    candidate_url, _ = generate_random_post_url(args.domain, seed_pool)
                    if candidate_url in visited or candidate_url in tried_random_posts:
                        continue
                    rand_url = candidate_url
                    break
                if rand_url:
                    tried_random_posts.add(rand_url)
                    if args.obey_robots and not rp.can_fetch(headers["User-Agent"], rand_url):
                        visited.add(rand_url)
                    else:
                        await polite_sleep(args.delay, args.jitter)
                        html = await fetch_http(session, rand_url, args.request_timeout)
                        if should_retry_with_playwright(args, html):
                            html = await fetch_with_playwright(rand_url)

                        visited.add(rand_url)
                        if html and not looks_like_soft_404(html):
                            register_post(
                                rand_url,
                                discovered_posts,
                                post_ids,
                                args.max_posts,
                            )
                            pages_fetched += 1
                            process_page(
                                rand_url,
                                html,
                                args,
                                discovered_profiles,
                                discovered_posts,
                                visited,
                                to_visit,
                                profile_templates,
                                post_ids,
                            )
                            save_checkpoint_if_needed(
                                pages_fetched,
                                args,
                                discovered_profiles,
                                discovered_posts,
                                visited,
                                to_visit,
                            )
                            continue
                else:
                    continue

            if not attempted_random:
                break

    final_state = state_snapshot(discovered_profiles, discovered_posts, visited, to_visit)
    save_state(args.state, final_state)

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


def build_parser() -> argparse.ArgumentParser:
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
        default="random_profiles.txt",
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
    parser.add_argument(
        "--random-profile-attempts",
        type=int,
        default=0,
        help="Extra attempts to probe random Base64 profile slugs (0 disables).",
    )
    parser.add_argument(
        "--random-post-attempts",
        type=int,
        default=0,
        help="Extra attempts to probe random /post/ IDs (0 disables).",
    )
    parser.add_argument(
        "--random-profile-template",
        action="append",
        default=None,
        help=(
            "Seed slug or full profile URL to guide the random generator. "
            "May be repeated. Defaults to a known Fiber example."
        ),
    )
    parser.add_argument(
        "--random-post-seed",
        action="append",
        default=None,
        help="Seed /post/ ID or URL to guide the random generator (defaults to a sample).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    d = urlparse(args.domain)
    if not d.scheme or not d.netloc:
        print("--domain must include scheme and host, e.g., https://fiber.al/", file=sys.stderr)
        sys.exit(2)
    args.domain = f"{d.scheme}://{d.netloc}"

    if args.max_posts is None:
        args.max_posts = args.max_profiles

    templates_from_cli = args.random_profile_template or []
    if not templates_from_cli:
        templates_from_cli = [DEFAULT_PROFILE_TEMPLATE_SLUG]
    extracted_templates: list[str] = []
    for value in templates_from_cli:
        slug = _extract_profile_slug(value)
        if slug:
            extracted_templates.append(slug)
    args.random_profile_template_inputs = extracted_templates

    post_seeds_from_cli = args.random_post_seed or []
    if not post_seeds_from_cli:
        post_seeds_from_cli = DEFAULT_POST_IDS[:]
    extracted_post_ids: list[str] = []
    for value in post_seeds_from_cli:
        pid = _extract_post_id_input(value)
        if pid:
            extracted_post_ids.append(pid)
    args.random_post_seed_inputs = extracted_post_ids

    args.random_profile_templates = load_templates_from_args(args)
    args.random_post_ids = load_post_ids_from_args(args)

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
