"""Loading the saved Phase 1 outputs with clear, non-crashing error reporting.

Every loader returns the object (or None) and records any problem in an
`issues` list, so the dashboard can show a helpful message on the page that
needs the file instead of crashing.
"""
import json
import os

import pandas as pd
import scipy.sparse as sp

from . import config as C

# file -> (folder, columns that must be present)
CSV_FILES = {
    "train_interactions": ("data", "train_interactions.csv.gz",
                           ["visitorid", "itemid", "weight", "n_events", "max_event_weight", "last_timestamp"]),
    "test_ground_truth": ("data", "test_ground_truth.csv", ["visitorid", "itemid"]),
    "item_popularity": ("data", "item_popularity.csv", ["itemid", "n_users", "total_weight"]),
    "evaluation_summary": ("results", "evaluation_summary.csv",
                           ["method", "n_users", "precision@5", "recall@5", "f1@5", "ndcg@5", "avg_time_ms"]),
    "per_user_evaluation": ("results", "per_user_evaluation.csv",
                            ["visitorid", "method", "precision@5", "recall@5", "f1@5", "ndcg@5", "time_ms"]),
    "recommendations": ("results", "recommendations.csv", ["visitorid", "method", "rank", "itemid", "score", "is_hit"]),
    "scalability_results": ("results", "scalability_results.csv",
                            ["fraction", "dataset_pct", "users", "products", "interactions", "graph_nodes",
                             "graph_build_time_s", "graph_memory_mb"]),
    "test_results": ("results", "test_results.csv", ["test", "status"]),
}
JSON_FILES = {
    "graph_stats": ("results", "graph_stats.json"),
    "dataset_overview": ("results", "dataset_overview.json"),
    "run_config": ("results", "config.json"),
    "cf_index": ("models", "cf_index.json"),
}
TEXT_FILES = {"interpretation": ("results", "interpretation.md")}
GRAPH_FILE = ("models", "bipartite_graph.pkl")
SIMILARITY_FILE = ("models", "item_similarity.npz")


def path_for(folder, name, outputs_dir=None):
    return os.path.join(outputs_dir or C.OUTPUTS_DIR, folder, name)


def _issue(issues, key, path, message):
    issues.append({"key": key, "file": os.path.relpath(path, os.path.dirname(C.OUTPUTS_DIR)), "message": message})


def load_csv(key, outputs_dir=None, issues=None):
    folder, name, required = CSV_FILES[key]
    path = path_for(folder, name, outputs_dir)
    issues = issues if issues is not None else []
    if not os.path.exists(path):
        _issue(issues, key, path, "file not found")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        _issue(issues, key, path, f"could not be read ({type(e).__name__}: {e})")
        return None
    missing = [c for c in required if c not in df.columns]
    if missing:
        _issue(issues, key, path, f"missing columns {missing}")
        return None
    if df.empty:
        _issue(issues, key, path, "file is empty")
        return None
    return df


def load_json(key, outputs_dir=None, issues=None):
    folder, name = JSON_FILES[key]
    path = path_for(folder, name, outputs_dir)
    issues = issues if issues is not None else []
    if not os.path.exists(path):
        _issue(issues, key, path, "file not found")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        _issue(issues, key, path, f"could not be read ({type(e).__name__}: {e})")
        return None


def load_text(key, outputs_dir=None, issues=None):
    folder, name = TEXT_FILES[key]
    path = path_for(folder, name, outputs_dir)
    issues = issues if issues is not None else []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        _issue(issues, key, path, f"could not be read ({e.strerror or e})")
        return None


def load_similarity(outputs_dir=None, issues=None):
    path = path_for(*SIMILARITY_FILE, outputs_dir)
    issues = issues if issues is not None else []
    try:
        return sp.load_npz(path)
    except Exception as e:
        _issue(issues, "item_similarity", path, f"could not be loaded ({type(e).__name__}: {e})")
        return None


def load_results(outputs_dir=None):
    """All saved results used by the comparison / scalability / home pages."""
    issues = []
    res = {}
    for key in ["evaluation_summary", "per_user_evaluation", "recommendations", "scalability_results",
                "test_results", "test_ground_truth", "item_popularity"]:
        res[key] = load_csv(key, outputs_dir, issues)
    for key in ["graph_stats", "dataset_overview", "run_config"]:
        res[key] = load_json(key, outputs_dir, issues)
    res["interpretation"] = load_text("interpretation", outputs_dir, issues)
    res["settings"] = C.load_run_config(path_for("results", "config.json", outputs_dir))
    res["issues"] = issues
    return res


def check_outputs(outputs_dir=None):
    """Existence check for every expected Phase 1 file (used for the sidebar status panel)."""
    outputs_dir = outputs_dir or C.OUTPUTS_DIR
    expected = [(f, n) for f, n, _ in CSV_FILES.values()] + list(JSON_FILES.values()) \
        + list(TEXT_FILES.values()) + [GRAPH_FILE, SIMILARITY_FILE]
    rows = []
    for folder, name in expected:
        p = path_for(folder, name, outputs_dir)
        rows.append({"file": f"{folder}/{name}", "exists": os.path.exists(p),
                     "size_mb": round(os.path.getsize(p) / 1e6, 2) if os.path.exists(p) else None})
    return pd.DataFrame(rows)
