import json
import os
import re
import math
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import random

from mcp.server.fastmcp import FastMCP

DATA_PATH = Path("data/sifted_last7d.json")
HISTORY_DIR = Path("data/history")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

mcp = FastMCP("origin-trends")

# --- tiny NLP helpers (pure Python) ---
STOP = {
    "the","a","an","and","or","to","of","in","on","for","with","as","at","by","from","into","about",
    "is","are","was","were","be","been","being","it","its","this","that","these","those","their","they",
    "you","we","our","your","but","not","can","could","should","would","may","might","will","just","more",
    "less","than","over","under","after","before","new","says","said"
}

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]+")

@mcp.tool()
def ping() -> Dict[str, Any]:
    return {"ok": True}


def _text_of(it: Dict[str, Any]) -> str:
    return " ".join([
        it.get("title") or "",
        it.get("excerpt") or "",
        " ".join(it.get("tags") or []),
        it.get("section") or "",
    ]).strip()

def _tokens(text: str) -> List[str]:
    toks = [t.lower() for t in WORD_RE.findall(text)]
    toks = [t for t in toks if t not in STOP and len(t) >= 3]
    return toks

def _bigrams(toks: List[str]) -> List[str]:
    return [f"{toks[i]} {toks[i+1]}" for i in range(len(toks)-1)]

def _count(items: List[Dict[str, Any]]) -> Tuple[Dict[str,int], Dict[str,int]]:
    uni: Dict[str,int] = {}
    bi: Dict[str,int] = {}
    for it in items:
        toks = _tokens(_text_of(it))
        for t in toks:
            uni[t] = uni.get(t, 0) + 1
        for b in _bigrams(toks):
            bi[b] = bi.get(b, 0) + 1
    return uni, bi

def _top(d: Dict[str,int], n: int) -> List[Tuple[str,int]]:
    return sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]

def _week_key(dt: datetime) -> str:
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"

def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n != len(ys) or n < 3:
        return float("nan")
    mx = sum(xs)/n
    my = sum(ys)/n
    vx = sum((x-mx)**2 for x in xs)
    vy = sum((y-my)**2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    cov = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    return cov / math.sqrt(vx*vy)

def _load_items(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))

def _history_files() -> List[Path]:
    return sorted(HISTORY_DIR.glob("week_*.json"))

# --- MCP tools ---
@mcp.tool()
def list_articles(query: Optional[str] = None, limit: int = 20, pro_only: Optional[bool] = None) -> List[Dict[str, Any]]:
    """Filter and return articles from data/sifted_last7d.json."""
    if not DATA_PATH.exists():
        return [{"error": f"Missing {DATA_PATH}"}]

    items = _load_items(DATA_PATH)

    out = []
    for it in items:
        if pro_only is not None and bool(it.get("is_pro")) != pro_only:
            continue
        txt = _text_of(it).lower()
        if query and query.lower() not in txt:
            continue
        out.append({k: it.get(k) for k in ["source","url","title","published_date","authors","section","tags","is_pro","excerpt"]})
        if len(out) >= limit:
            break
    return out

@mcp.tool()
def compute_trends(top_n: int = 12) -> Dict[str, Any]:
    """Compute weekly keyword + bigram trends from the current JSON (no time series)."""
    if not DATA_PATH.exists():
        return {"error": f"Missing {DATA_PATH}. Commit it or generate mock history."}

    items = _load_items(DATA_PATH)
    uni, bi = _count(items)

    return {
        "source": "sifted",
        "n_items": len(items),
        "top_keywords": [{"term": k, "count": c} for k, c in _top(uni, top_n)],
        "top_bigrams": [{"term": k, "count": c} for k, c in _top(bi, top_n)],
        "note": "This is single-week theme extraction. For correlations you need multiple weeks -> use generate_mock_history()."
    }

@mcp.tool()
def generate_mock_history(n_weeks: int = 12, seed: int = 7) -> Dict[str, Any]:
    """
    Create data/history/week_YYYY-WW.json files from your base week,
    injecting rotating 'spike' terms so correlations are meaningful.
    """
    rng = random.Random(seed)

    if not DATA_PATH.exists():
        return {"error": f"Missing {DATA_PATH}. Commit it to the repo or put mock base data in data/."}

    base = _load_items(DATA_PATH)
    if not base:
        return {"error": "Base JSON is empty."}

    spikes = [
        "defence tech", "ai chips", "data centres", "secondary", "series b",
        "m&a", "regulation", "climate", "fintech", "robotics"
    ]

    start = datetime.utcnow() - timedelta(weeks=n_weeks)
    created = []

    for i in range(n_weeks):
        dt = start + timedelta(weeks=i)
        wk = _week_key(dt)

        k = rng.randint(max(8, len(base)//3), max(12, len(base)))
        sampled = [rng.choice(base).copy() for _ in range(k)]

        spike = spikes[i % len(spikes)]
        for it in sampled[: max(2, k//5)]:
            it["excerpt"] = ((it.get("excerpt") or "") + f" | {spike}").strip()
            it["published_date"] = dt.isoformat()

        outp = HISTORY_DIR / f"week_{wk}.json"
        outp.write_text(json.dumps(sampled, ensure_ascii=False, indent=2), encoding="utf-8")
        created.append(outp.name)

    return {"ok": True, "weeks_created": n_weeks, "files_example": created[:3]}

@mcp.tool()
def correlate_trends(top_terms: int = 20, top_pairs: int = 12) -> Dict[str, Any]:
    """
    Build weekly keyword counts from data/history and return top correlated keyword pairs.
    """
    files = _history_files()
    if len(files) < 4:
        return {"error": "Need >= 4 history weeks. Run generate_mock_history(n_weeks=12) first."}

    weeks = [p.stem.replace("week_", "") for p in files]

    # build per-week top keywords
    week_counts: List[Dict[str,int]] = []
    for p in files:
        items = _load_items(p)
        uni, _ = _count(items)
        week_counts.append(dict(_top(uni, top_terms)))

    # collect candidate terms
    vocab = sorted(set(t for wc in week_counts for t in wc.keys()))

    # time series per term
    series: Dict[str, List[float]] = {}
    for term in vocab:
        series[term] = [float(wc.get(term, 0)) for wc in week_counts]

    # compute correlations
    pairs = []
    for i in range(len(vocab)):
        for j in range(i+1, len(vocab)):
            a, b = vocab[i], vocab[j]
            r = _pearson(series[a], series[b])
            if not math.isnan(r):
                pairs.append((abs(r), r, a, b))

    pairs.sort(reverse=True, key=lambda x: x[0])
    pairs = pairs[:top_pairs]

    return {
        "weeks": weeks,
        "n_weeks": len(weeks),
        "top_terms_per_week": top_terms,
        "top_correlations": [
            {"corr": r, "term_a": a, "term_b": b}
            for _, r, a, b in pairs
        ],
        "note": "This is keyword-level correlation (demo). Next step is correlating clusters + portfolio/company mention series."
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    mcp.run(transport="http", host="0.0.0.0", port=port)

