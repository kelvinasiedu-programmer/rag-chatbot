"""Download PDF filings into ./data/sample_corpus/ for ingestion.

Usage:
    python scripts/fetch_sample_corpus.py URL1 URL2 URL3 ...

Notes:
- Sets a polite User-Agent because SEC EDGAR rejects anonymous requests.
- If a URL is an SEC EDGAR HTML filing (not a PDF), this script will warn
  rather than save garbage; download the PDF copy from the company's
  investor-relations page instead.
- Files are saved under ./data/sample_corpus/ with their URL basename.
"""
from __future__ import annotations

import sys
import urllib.parse
from pathlib import Path

import requests

USER_AGENT = "RAG Chatbot Sample Loader (kelvinasiedu0807@gmail.com)"
TARGET_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_corpus"


def fetch(url: str) -> Path | None:
    name = Path(urllib.parse.urlparse(url).path).name or "filing.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    out = TARGET_DIR / name

    print(f"Fetching {url}")
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60, stream=True)
    resp.raise_for_status()

    ctype = resp.headers.get("content-type", "").lower()
    if "pdf" not in ctype:
        print(
            f"  ! WARNING: content-type is '{ctype}', not PDF. "
            f"Skipping. Use the IR-page PDF link, not the EDGAR HTML filing."
        )
        return None

    out.write_bytes(resp.content)
    print(f"  -> saved {out.relative_to(TARGET_DIR.parent.parent)} ({out.stat().st_size // 1024} KB)")
    return out


def main(urls: list[str]) -> int:
    if not urls:
        print(__doc__)
        return 1
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    saved = [p for p in (fetch(u) for u in urls) if p]
    print(f"\nSaved {len(saved)} / {len(urls)} files to {TARGET_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
