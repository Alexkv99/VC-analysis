import json
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import requests
from bs4 import BeautifulSoup

# -------------------------
# Config
# -------------------------
BASE = "https://sifted.eu"
LATEST = f"{BASE}/latest"
MAX_LATEST_PAGES = 8          # how far back in listing pages to scan
DAYS = 7                      # last N days
OUTFILE = "sifted_last7d.json"
ERROR_LOG = "errors.log"
SLEEP_S = 0.6                 # be polite

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)

ARTICLE_RE = re.compile(r"^https://sifted\.eu/articles/[^/\s]+/?$")

# Europe/Paris offset (simple fixed offset; ok for weekly)
PARIS_TZ = timezone(timedelta(hours=1))  # CET


# -------------------------
# Logging helpers
# -------------------------
def log(msg: str) -> None:
    print(msg, flush=True)

def log_error(msg: str) -> None:
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print("[ERROR]", msg, flush=True)


# -------------------------
# HTTP
# -------------------------
def get_text(url: str) -> str:
    log(f"GET {url}")
    r = requests.get(
        url,
        headers={
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.9",
        },
        timeout=25,
    )
    log(f"  -> {r.status_code} ({len(r.content)} bytes)")
    r.raise_for_status()
    time.sleep(SLEEP_S)
    return r.text


# -------------------------
# Discovery from /latest pages
# -------------------------
def latest_pages() -> List[str]:
    urls = [LATEST]
    for i in range(2, MAX_LATEST_PAGES + 1):
        urls.append(f"{LATEST}/page/{i}")
    return urls

def extract_article_urls_from_listing(html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    out: List[str] = []
    seen = set()

    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if href.startswith("/articles/"):
            href = BASE + href
        if ARTICLE_RE.match(href):
            key = href.rstrip("/")
            if key not in seen:
                seen.add(key)
                out.append(href)

    return out

def dedupe_keep_order(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        k = u.rstrip("/")
        if k in seen:
            continue
        seen.add(k)
        out.append(u)
    return out


# -------------------------
# Parsing article page
# -------------------------
def safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None

def extract_json_ld(soup: BeautifulSoup) -> List[dict]:
    out = []
    for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
        data = safe_json_loads(s.get_text(strip=True))
        if not data:
            continue
        if isinstance(data, dict):
            out.append(data)
        elif isinstance(data, list):
            out.extend([x for x in data if isinstance(x, dict)])
    return out

def first_meta(soup: BeautifulSoup, *, prop: str = None, name: str = None) -> Optional[str]:
    if prop:
        tag = soup.find("meta", attrs={"property": prop})
        if tag and tag.get("content"):
            return tag["content"].strip()
    if name:
        tag = soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            return tag["content"].strip()
    return None

def parse_date(dt_str: str) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone(PARIS_TZ)
    except Exception:
        return None

def parse_sifted_article(url: str, html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")

    title = None
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(" ", strip=True)

    excerpt = first_meta(soup, prop="og:description") or first_meta(soup, name="description")

    page_text = soup.get_text(" ", strip=True)
    is_pro = bool(re.search(r"\bPro\b", page_text))

    authors: List[str] = []
    section = None
    published_dt: Optional[datetime] = None

    for obj in extract_json_ld(soup):
        if "@graph" in obj and isinstance(obj["@graph"], list):
            for node in obj["@graph"]:
                if isinstance(node, dict) and node.get("@type") in ("NewsArticle", "Article"):
                    if not published_dt and node.get("datePublished"):
                        published_dt = parse_date(str(node.get("datePublished")))

        if obj.get("@type") in ("NewsArticle", "Article", "ReportageNewsArticle"):
            if not title and obj.get("headline"):
                title = str(obj["headline"])
            if not excerpt and obj.get("description"):
                excerpt = str(obj["description"])
            if not published_dt and obj.get("datePublished"):
                published_dt = parse_date(str(obj.get("datePublished")))
            if obj.get("articleSection"):
                section = str(obj.get("articleSection"))

            auth = obj.get("author")
            if isinstance(auth, dict) and auth.get("name"):
                authors.append(str(auth["name"]))
            elif isinstance(auth, list):
                for a in auth:
                    if isinstance(a, dict) and a.get("name"):
                        authors.append(str(a["name"]))

    tags: List[str] = []
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if href.startswith("/sector/") or href.startswith("/topic/") or href.startswith("/countries/"):
            t = a.get_text(" ", strip=True)
            if t and t not in tags:
                tags.append(t)

    authors = [a.strip() for a in authors if a and a.strip()]
    authors = list(dict.fromkeys(authors))

    return {
        "source": "sifted",
        "url": url,
        "title": title,
        "published_date": published_dt.isoformat() if published_dt else None,
        "authors": authors,
        "section": section,
        "tags": tags,
        "is_pro": is_pro,
        "excerpt": excerpt,
    }


# -------------------------
# Main
# -------------------------
def main():
    # reset error log each run
    with open(ERROR_LOG, "w", encoding="utf-8") as f:
        f.write("")

    cutoff = datetime.now(tz=PARIS_TZ) - timedelta(days=DAYS)
    log(f"Cutoff for last {DAYS} days: {cutoff.isoformat()}")

    # 1) discover candidate article URLs
    all_urls: List[str] = []
    for i, lp in enumerate(latest_pages(), start=1):
        try:
            log(f"\n[LISTING {i}/{MAX_LATEST_PAGES}] {lp}")
            html = get_text(lp)
            urls = extract_article_urls_from_listing(html)
            log(f"Found {len(urls)} article URLs on this page")
            all_urls.extend(urls)
        except Exception as e:
            log_error(f"Listing page failed: {lp} :: {repr(e)}")

    urls = dedupe_keep_order(all_urls)
    log(f"\nTotal unique article URLs discovered: {len(urls)}")

    # 2) fetch + parse, filter last 7 days
    items: List[Dict[str, Any]] = []
    kept = 0
    filtered_old = 0
    no_date = 0

    for idx, url in enumerate(urls, start=1):
        try:
            log(f"\n[ARTICLE {idx}/{len(urls)}] {url}")
            html = get_text(url)
            item = parse_sifted_article(url, html)

            pdate = item.get("published_date")
            if pdate:
                dt = parse_date(pdate)
                log(f"Parsed date: {dt.isoformat() if dt else pdate}")
                if dt and dt < cutoff:
                    filtered_old += 1
                    log("-> SKIP (older than cutoff)")
                    continue
            else:
                no_date += 1
                log("Parsed date: None (keeping)")

            kept += 1
            log(f"-> KEEP | Pro={item.get('is_pro')} | title={repr(item.get('title'))[:120]}")
            items.append(item)

        except Exception as e:
            log_error(f"Article failed: {url} :: {repr(e)}")

    # 3) save
    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    log("\n====================")
    log(f"Done. Kept: {kept}")
    log(f"Filtered (old): {filtered_old}")
    log(f"Missing date (kept): {no_date}")
    log(f"Wrote: {OUTFILE}")
    log(f"Errors (if any): {ERROR_LOG}")
    log("====================\n")


if __name__ == "__main__":
    main()
