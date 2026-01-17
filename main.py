import csv
import io
import json
import math
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------
# Paths (handle both `data/` and `Data/` because Linux is case-sensitive)
# ---------------------------------------------------------------------
DATA_DIR_CANDIDATES = [Path("data"), Path("Data"), Path(".")]

def _first_existing(*paths: Path) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

# Prefer data/ or Data/ if present
DATA_DIR = _first_existing(*DATA_DIR_CANDIDATES) or Path("data")
HISTORY_DIR = DATA_DIR / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# Prefer JSON file inside data folders, fallback to repo root
SIFTED_JSON = _first_existing(
    DATA_DIR / "sifted_last7d.json",
    Path("sifted_last7d.json"),
)

mcp = FastMCP("origin-investor")

# -------------------------
# Text helpers
# -------------------------
WORD_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9\-\.]+")

STOP = {
    "the","a","an","and","or","to","of","in","on","for","with","as","at","by","from","into","about",
    "is","are","was","were","be","been","being","it","its","this","that","these","those","their","they",
    "you","we","our","your","but","not","can","could","should","would","may","might","will","just","more",
    "less","than","over","under","after","before","new","says","said",
}


def _safe_lower(x: Any) -> str:
    return (str(x) if x is not None else "").strip().lower()


def _tokens(text: str) -> List[str]:
    toks = [_safe_lower(t) for t in WORD_RE.findall(text or "")]
    # keep dots/dashes tokens (domains, product names), but drop stop words
    out = [t for t in toks if len(t) >= 3 and t not in STOP]
    return out


def _text_of_article(it: Dict[str, Any]) -> str:
    return " ".join([
        it.get("title") or "",
        it.get("excerpt") or "",
        " ".join(it.get("tags") or []),
        it.get("section") or "",
        it.get("url") or "",
    ]).strip()


def _load_sifted_items() -> List[Dict[str, Any]]:
    if not SIFTED_JSON or not SIFTED_JSON.exists():
        return []
    return json.loads(SIFTED_JSON.read_text(encoding="utf-8"))


def _parse_iso(dt: Optional[str]) -> Optional[datetime]:
    if not dt:
        return None
    try:
        # Handles Z as UTC
        return datetime.fromisoformat(dt.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _week_key(dt: datetime) -> str:
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n != len(ys) or n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return cov / math.sqrt(vx * vy)


def _zscore(values: List[float], x: float) -> float:
    if not values:
        return 0.0
    m = sum(values) / len(values)
    v = sum((a - m) ** 2 for a in values) / max(1, (len(values) - 1))
    s = math.sqrt(v)
    if s == 0:
        return 0.0
    return (x - m) / s


def _decay_weight(published_utc: Optional[datetime], half_life_days: float = 7.0) -> float:
    """Exponential decay so very recent articles matter more."""
    if not published_utc:
        return 0.5  # unknown date => medium weight
    age_days = (datetime.now(timezone.utc) - published_utc).total_seconds() / 86400.0
    # weight = 0.5^(age/half_life)
    return 0.5 ** (age_days / max(0.001, half_life_days))


# -------------------------
# Portfolio parsing
# -------------------------
@dataclass
class Startup:
    name: str
    domain: str = ""
    linkedin: str = ""
    raw: Dict[str, Any] = None

    def mention_keys(self) -> List[str]:
        keys: List[str] = []
        n = (self.name or "").strip()
        if n:
            keys.append(n)
        d = (self.domain or "").strip()
        if d:
            # allow full domain and root token
            keys.append(d)
            root = d.replace("https://", "").replace("http://", "").split("/")[0]
            keys.append(root)
            keys.append(root.replace("www.", ""))
            # brand-ish token from domain
            base = root.replace("www.", "").split(":")[0]
            base = base.split(".")[0]
            if base and base not in keys:
                keys.append(base)
        return [k for k in dict.fromkeys(keys) if k]


def _infer_field(row: Dict[str, str], candidates: List[str]) -> str:
    # match by normalized header
    norm = {re.sub(r"[^a-z0-9]", "", k.lower()): k for k in row.keys()}
    for c in candidates:
        ck = re.sub(r"[^a-z0-9]", "", c.lower())
        if ck in norm:
            return (row.get(norm[ck]) or "").strip()
    return ""


def parse_portfolio_csv(csv_text: str) -> List[Startup]:
    """Parse any CRM export: tries to auto-detect name/domain/linkedin columns."""
    f = io.StringIO(csv_text)
    sniffer = csv.Sniffer()
    sample = csv_text[:4096]
    try:
        dialect = sniffer.sniff(sample)
    except Exception:
        dialect = csv.excel

    reader = csv.DictReader(f, dialect=dialect)
    out: List[Startup] = []
    for row in reader:
        name = _infer_field(row, ["name", "startup", "company", "company name", "startup name"]) or ""
        domain = _infer_field(row, ["domain", "website", "url", "company domain"]) or ""
        linkedin = _infer_field(row, ["linkedin", "linkedin page", "linkedin url"]) or ""

        # fallback: if first column looks like a name
        if not name:
            for k in row.keys():
                if row.get(k) and len(str(row.get(k))) >= 2:
                    name = str(row.get(k)).strip()
                    break

        if not name:
            continue

        out.append(Startup(name=name, domain=domain, linkedin=linkedin, raw=row))

    # dedupe by name
    seen = set()
    dedup: List[Startup] = []
    for s in out:
        key = s.name.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(s)
    return dedup


# -------------------------
# Scoring
# -------------------------
@dataclass
class StartupScore:
    name: str
    domain: str
    mention_count: int
    weighted_mentions: float
    trend_overlap: int
    outlier_z: float
    score: float
    evidence_urls: List[str]


def _article_mentions(article_text_l: str, keys_l: List[str]) -> bool:
    # Fast containment check
    for k in keys_l:
        if k and k in article_text_l:
            return True
    return False


def rank_startups_from_portfolio(portfolio: List[Startup], items: List[Dict[str, Any]], top_k: int = 10) -> List[StartupScore]:
    if not portfolio:
        return []

    # global trends for this JSON
    all_tokens = []
    for it in items:
        all_tokens.extend(_tokens(_text_of_article(it)))
    # top 30 terms as “trend vocabulary” for overlap scoring (cheap MVP)
    freq: Dict[str, int] = {}
    for t in all_tokens:
        freq[t] = freq.get(t, 0) + 1
    trend_vocab = set([t for t, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:30]])

    # compute per-startup mention stats
    raw_scores: List[Tuple[Startup, int, float, int, List[str]]] = []

    for s in portfolio:
        keys = s.mention_keys()
        keys_l = [k.lower() for k in keys]

        mention_count = 0
        weighted = 0.0
        overlap = 0
        ev_urls: List[str] = []

        for it in items:
            txt = _text_of_article(it)
            txt_l = txt.lower()
            if not _article_mentions(txt_l, keys_l):
                continue

            mention_count += 1
            dt = _parse_iso(it.get("published_date"))
            w = _decay_weight(dt)
            weighted += w

            # overlap: how many trend terms appear in that article
            toks = set(_tokens(txt))
            overlap += len(toks.intersection(trend_vocab))

            if it.get("url"):
                ev_urls.append(it["url"])

        raw_scores.append((s, mention_count, weighted, overlap, ev_urls[:6]))

    # outlier z-score on weighted_mentions
    weighted_list = [w for _, _, w, _, _ in raw_scores]

    results: List[StartupScore] = []
    for s, mc, w, ov, ev in raw_scores:
        z = _zscore(weighted_list, w)
        # Simple combined score: prioritize (a) weighted mentions, (b) outlierness, (c) trend overlap
        score = (2.0 * w) + (1.5 * max(0.0, z)) + (0.05 * ov)
        results.append(
            StartupScore(
                name=s.name,
                domain=s.domain,
                mention_count=mc,
                weighted_mentions=round(w, 4),
                trend_overlap=ov,
                outlier_z=round(z, 4),
                score=round(score, 4),
                evidence_urls=ev,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[: max(1, top_k)]


# -------------------------
# Weekly snapshot + correlations (optional but useful)
# -------------------------
@mcp.tool()
def snapshot_week() -> Dict[str, Any]:
    """Save a weekly snapshot of trend vocab counts so you can correlate across weeks later."""
    items = _load_sifted_items()
    if not items:
        return {"error": "Missing or empty sifted_last7d.json"}

    # decide week by 'now'
    wk = _week_key(datetime.now(timezone.utc))

    # trend counts
    freq: Dict[str, int] = {}
    for it in items:
        for t in _tokens(_text_of_article(it)):
            freq[t] = freq.get(t, 0) + 1

    outp = HISTORY_DIR / f"trends_{wk}.json"
    outp.write_text(json.dumps({"week": wk, "counts": freq}, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"ok": True, "week": wk, "path": str(outp)}


@mcp.tool()
def correlate_terms(term_a: str, term_b: str) -> Dict[str, Any]:
    """Correlation of two terms across saved weekly snapshots (needs multiple weeks)."""
    files = sorted(HISTORY_DIR.glob("trends_*.json"))
    if len(files) < 4:
        return {"error": "Need >= 4 weekly snapshots. Run snapshot_week() weekly (or generate mock history)."}

    weeks: List[str] = []
    xs: List[float] = []
    ys: List[float] = []

    a = _safe_lower(term_a)
    b = _safe_lower(term_b)

    for p in files:
        obj = json.loads(p.read_text(encoding="utf-8"))
        weeks.append(obj.get("week") or p.stem.replace("trends_", ""))
        counts = obj.get("counts") or {}
        xs.append(float(counts.get(a, 0)))
        ys.append(float(counts.get(b, 0)))

    r = _pearson(xs, ys)
    return {"term_a": a, "term_b": b, "weeks": weeks, "series_a": xs, "series_b": ys, "corr": r}


# -------------------------
# MCP tools for ChatGPT Developer Mode
# -------------------------
@mcp.tool()
def ping() -> Dict[str, Any]:
    return {"ok": True, "data_dir": str(DATA_DIR), "has_sifted_json": bool(SIFTED_JSON and SIFTED_JSON.exists())}


@mcp.tool()
def list_articles(query: Optional[str] = None, limit: int = 20, pro_only: Optional[bool] = None) -> List[Dict[str, Any]]:
    """Filter and return articles from the Sifted JSON."""
    items = _load_sifted_items()
    if not items:
        return [{"error": "Missing or empty sifted_last7d.json"}]

    out: List[Dict[str, Any]] = []
    for it in items:
        if pro_only is not None and bool(it.get("is_pro")) != pro_only:
            continue
        txt = _text_of_article(it).lower()
        if query and query.lower() not in txt:
            continue
        out.append({k: it.get(k) for k in ["source","url","title","published_date","authors","section","tags","is_pro","excerpt"]})
        if len(out) >= max(1, int(limit)):
            break
    return out


@mcp.tool()
def compute_trends(top_n: int = 12) -> Dict[str, Any]:
    """Top tokens + bigrams for the currently loaded JSON (single-week)."""
    items = _load_sifted_items()
    if not items:
        return {"error": "Missing or empty sifted_last7d.json"}

    uni: Dict[str, int] = {}
    bi: Dict[str, int] = {}

    for it in items:
        toks = _tokens(_text_of_article(it))
        for t in toks:
            uni[t] = uni.get(t, 0) + 1
        for i in range(len(toks) - 1):
            bg = f"{toks[i]} {toks[i+1]}"
            bi[bg] = bi.get(bg, 0) + 1

    top_kw = sorted(uni.items(), key=lambda x: x[1], reverse=True)[: max(1, int(top_n))]
    top_bg = sorted(bi.items(), key=lambda x: x[1], reverse=True)[: max(1, int(top_n))]

    return {
        "source": "sifted",
        "n_items": len(items),
        "top_keywords": [{"term": k, "count": c} for k, c in top_kw],
        "top_bigrams": [{"term": k, "count": c} for k, c in top_bg],
    }


@mcp.tool()
def rank_portfolio(csv_text: str, top_k: int = 10) -> Dict[str, Any]:
    """Main MVP: parse a CRM CSV and rank startups based on news mentions + trend alignment + outlierness."""
    items = _load_sifted_items()
    if not items:
        return {"error": "Missing or empty sifted_last7d.json"}

    portfolio = parse_portfolio_csv(csv_text)
    if not portfolio:
        return {"error": "Could not parse any startups from the provided CSV text."}

    ranked = rank_startups_from_portfolio(portfolio, items, top_k=top_k)

    return {
        "n_startups": len(portfolio),
        "n_articles": len(items),
        "top_k": top_k,
        "ranking": [asdict(r) for r in ranked],
        "how_to_use": "Provide your CRM export as raw CSV text (including header). The tool ranks by recent mentions (decayed), trend overlap, and positive outlier z-score.",
    }


app = mcp.streamable_http_app()


if __name__ == "__main__":
    # --- Alpic transport detection hint ---
    # Alpic scans source for mcp.run(transport="http"). Keep it present.
    if False:
        mcp.run(transport="http")

    import uvicorn

    port = int(os.environ.get("PORT") or os.environ.get("ALPIC_PORT") or "8000")
    print(f"Starting MCP HTTP on 0.0.0.0:{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
