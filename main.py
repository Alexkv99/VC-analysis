import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------
# Config
# -------------------------
DATA_PATH = Path("data/sifted_last7d.json")

# -------------------------
# FastAPI + MCP
# -------------------------
app = FastAPI()
mcp = FastMCP("origin-trends")

@app.get("/health")
def health():
    return {"ok": True, "data_exists": DATA_PATH.exists()}

# -------------------------
# Helpers
# -------------------------
def load_items() -> List[Dict[str, Any]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Put your JSON there (data/sifted_last7d.json)."
        )

    items = json.loads(DATA_PATH.read_text(encoding="utf-8"))

    # Build a single unified text field for trend detection
    for it in items:
        it["_text"] = " ".join(
            [
                it.get("title") or "",
                it.get("excerpt") or "",
                " ".join(it.get("tags") or []),
                it.get("section") or "",
            ]
        ).strip()

    return items

def _top_terms_per_cluster(
    X, vectorizer: TfidfVectorizer, labels: np.ndarray, top_n: int = 10
) -> List[Dict[str, Any]]:
    terms = np.array(vectorizer.get_feature_names_out())
    clusters: List[Dict[str, Any]] = []

    for cid in sorted(set(labels.tolist())):
        idx = np.where(labels == cid)[0]
        centroid = X[idx].mean(axis=0).A1  # mean tf-idf vector for the cluster
        top_idx = np.argsort(-centroid)[:top_n]
        keywords = terms[top_idx].tolist()

        clusters.append(
            {
                "cluster_id": int(cid),
                "keywords": keywords,
                "size": int(len(idx)),
                "indices": idx.tolist(),
            }
        )

    # Sort clusters by size descending
    clusters.sort(key=lambda c: c["size"], reverse=True)
    return clusters

# -------------------------
# MCP Tools
# -------------------------
@mcp.tool()
def list_articles(
    query: Optional[str] = None,
    limit: int = 20,
    pro_only: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    Return articles from the JSON with optional text filtering.
    - query: substring match on (title+excerpt+tags+section)
    - pro_only: if True, keep only Pro; if False keep only non-Pro; if None keep all
    """
    items = load_items()

    def keep(it: Dict[str, Any]) -> bool:
        if pro_only is not None and bool(it.get("is_pro")) != pro_only:
            return False
        if query:
            q = query.lower()
            return q in (it.get("_text") or "").lower()
        return True

    out: List[Dict[str, Any]] = []
    for it in items:
        if keep(it):
            out.append(
                {
                    "source": it.get("source"),
                    "url": it.get("url"),
                    "title": it.get("title"),
                    "published_date": it.get("published_date"),
                    "authors": it.get("authors"),
                    "section": it.get("section"),
                    "tags": it.get("tags"),
                    "is_pro": it.get("is_pro"),
                    "excerpt": it.get("excerpt"),
                }
            )
        if len(out) >= limit:
            break

    return out


@mcp.tool()
def compute_trends(k: int = 8, examples_per_cluster: int = 5) -> Dict[str, Any]:
    """
    Cluster articles into k themes and return cluster keywords + example headlines.
    MVP uses TF-IDF + KMeans.
    """
    items = load_items()
    texts = [it["_text"] for it in items if it.get("_text")]

    if len(texts) < max(5, k):
        return {
            "error": "Not enough articles/text to cluster",
            "n_items": len(items),
            "requested_k": k,
        }

    vectorizer = TfidfVectorizer(
        max_features=7000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X = vectorizer.fit_transform(texts)

    km = KMeans(n_clusters=k, random_state=7, n_init="auto")
    labels = km.fit_predict(X)

    clusters = _top_terms_per_cluster(X, vectorizer, labels, top_n=12)

    # Attach examples
    out_clusters: List[Dict[str, Any]] = []
    for c in clusters:
        idxs = c["indices"]
        examples = []
        for i in idxs[:examples_per_cluster]:
            it = items[i]
            examples.append(
                {
                    "title": it.get("title"),
                    "url": it.get("url"),
                    "is_pro": it.get("is_pro"),
                    "tags": it.get("tags"),
                    "section": it.get("section"),
                }
            )

        out_clusters.append(
            {
                "cluster_id": c["cluster_id"],
                "size": c["size"],
                "keywords": c["keywords"],
                "examples": examples,
            }
        )

    return {
        "source": "sifted",
        "k": k,
        "n_items": len(items),
        "clusters": out_clusters,
    }


@mcp.tool()
def weekly_brief(k: int = 8) -> Dict[str, Any]:
    """
    Returns a ready-to-narrate structure for a weekly brief.
    (ChatGPT can turn this into a VC-style recap.)
    """
    r = compute_trends(k=k, examples_per_cluster=4)
    if "error" in r:
        return r

    top = r["clusters"][:5]
    themes = []
    for c in top:
        themes.append(
            {
                "theme_hint": ", ".join(c["keywords"][:5]),
                "signal": f"{c['size']} articles",
                "examples": c["examples"][:2],
            }
        )

    return {
        "headline": "Sifted — Weekly market themes (MVP)",
        "n_articles": r["n_items"],
        "top_themes": themes,
        "notes": [
            "This MVP clusters by TF-IDF (title+excerpt+tags). Next: embeddings + better entity extraction.",
            "Your current JSON may have published_date=null; add dates later for momentum/correlations.",
        ],
    }


# -------------------------
# Mount MCP HTTP app
# -------------------------
app.mount("/mcp", mcp.streamable_http_app())
