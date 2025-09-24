# ProfilePuller

`pull_avatar.py` is a Python script for fetching public profile pictures from a list of URLs.  
It works by scanning each page for `og:image` or `twitter:image` metadata, or by downloading directly if the URL already points to an image.

⚠️ This script is designed for **public pages only**. It will not bypass logins, captchas, or private APIs. Use responsibly and within each platform’s Terms of Service.

## Features
- Download a single avatar by passing a URL
- Batch mode with `--from-file profiles.txt`
- Parallel downloads with adjustable concurrency
- Skip already-downloaded files
- Custom User-Agent
- Limit the number of URLs processed with `--limit`

## Requirements
Install dependencies with:

    pip install -r requirements.txt

Dependencies:
- `requests`
- `beautifulsoup4`

## Usage

Single URL:

    python pull_avatar.py "https://example.com/someprofile"

Batch from file:

    python pull_avatar.py --from-file profiles.txt

Limit to first 10 URLs:

    python pull_avatar.py --from-file profiles.txt --limit 10

Custom output directory, skip existing files, and 8 workers:

    python pull_avatar.py --from-file profiles.txt --out avatars_out --skip-existing --concurrency 8

## profiles.txt format

    # Comments are ignored
    https://example.com/profile/user123
    https://cdn.somesite.com/u/avatar.png

## License
See [LICENSE](LICENSE) for details.
