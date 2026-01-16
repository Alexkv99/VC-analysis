import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from mcp.server.fastmcp import FastMCP
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = Path("data/sifted_last7d.json")
mcp = FastMCP("origin-trends")

def load_items() -> List[Dict[str, Any]]:
    items = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    for it in items:
        it["_text"] = " ".join([
            it.get("title") or "",
            it.get("excerpt") or "",
            " ".join(it.get("tags") or []),
            it.get("section") or "",
        ]).strip()
    return items

def _top_terms_per_cluster(X, vectorizer, labels, top_n: int = 10):
    terms = np.array(vectorizer.get_feature_names_out())
    clusters = []
    for cid in sorted(set(labels.tolist())):
        idx = np.where(labels == cid)[0]
        centroid = X[idx].mean(axis=0).A1
        top = terms[np.argsort(-centroid)[:top_n]].tolist()
        clusters.append({"cluster_id": int(cid), "keywords": top, "size": int(len(idx)), "indices": idx.tolist()})
    clusters.sort(key=lambda c: c["size"], reverse=True)
    return clusters

@mcp.tool()
def list_articles(query: Optional[str] = None, limit: int = 20, pro_only: Optional[bool] = None):
    items = load_items()
    out = []
    for it in items:
        if pro_only is not None and bool(it.get("is_pro")) != pro_only:
            continue
        if query and query.lower() not in (it.get("_text") or "").lower():
            continue
        out.append({k: it.get(k) for k in ["source","url","title","published_date","authors","section","tags","is_pro","excerpt"]})
        if len(out) >= limit:
            break
    return out

@mcp.tool()
def compute_trends(k: int = 8, examples_per_cluster: int = 4):
    items = load_items()
    texts = [it["_text"] for it in items if it.get("_text")]
    if len(texts) < max(5, k):
        return {"error": "Not enough text to cluster", "n_items": len(items), "requested_k": k}

    vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1,2), stop_words="english")
    X = vectorizer.fit_transform(texts)

    km = KMeans(n_clusters=k, random_state=7, n_init="auto")
    labels = km.fit_predict(X)

    clusters = _top_terms_per_cluster(X, vectorizer, labels, top_n=12)
    out_clusters = []
    for c in clusters:
        ex = []
        for i in c["indices"][:examples_per_cluster]:
            it = items[i]
            ex.append({"title": it.get("title"), "url": it.get("url"), "is_pro": it.get("is_pro")})
        out_clusters.append({"cluster_id": c["cluster_id"], "size": c["size"], "keywords": c["keywords"], "examples": ex})

    return {"source": "sifted", "k": k, "n_items": len(items), "clusters": out_clusters}

@mcp.tool()
def weekly_brief(k: int = 8):
    r = compute_trends(k=k, examples_per_cluster=4)
    if "error" in r:
        return r
    top = r["clusters"][:5]
    return {
        "headline": "Sifted — Weekly market themes (MVP)",
        "n_articles": r["n_items"],
        "top_themes": [
            {"theme_hint": ", ".join(c["keywords"][:5]), "signal": f"{c['size']} articles", "examples": c["examples"][:2]}
            for c in top
        ],
        "notes": [
            "MVP uses TF-IDF + KMeans on title+excerpt+tags.",
            "Add published_date to enable momentum + correlations over time."
        ],
    }

# ✅ THIS is what uvicorn will serve (MCP at the root)
app = mcp.streamable_http_app()
