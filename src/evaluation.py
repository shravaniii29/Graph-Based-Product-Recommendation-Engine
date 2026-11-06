"""Evaluation metrics and helpers.

Metric functions, `evaluate_recommender` and `bootstrap_ci` are ported unchanged
from Phase 1 notebook Section 8. `summarise` reproduces the comparison table and
`ci_overlap_table` / `recommendation_overlap` are read-only analyses of saved results.
"""
import math
import time

import numpy as np
import pandas as pd

from .config import N_BOOTSTRAP, SEED, TOP_K

METRICS = ["precision@5", "recall@5", "f1@5", "ndcg@5", "hit@5"]


def precision_at_k(recommended, relevant, k=TOP_K):
    return len(set(recommended[:k]) & relevant) / k


def recall_at_k(recommended, relevant, k=TOP_K):
    return len(set(recommended[:k]) & relevant) / len(relevant) if relevant else 0.0


def f1_at_k(recommended, relevant, k=TOP_K):
    p, r = precision_at_k(recommended, relevant, k), recall_at_k(recommended, relevant, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def ndcg_at_k(recommended, relevant, k=TOP_K):
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommender(model, eval_users, gt, k=TOP_K, progress_every=100):
    metric_rows, rec_rows = [], []
    for n, u in enumerate(eval_users, 1):
        t0 = time.perf_counter()
        recs = model.recommend(u, k)
        dt_ms = (time.perf_counter() - t0) * 1000
        items = [i for i, _ in recs]
        rel = gt[u]
        metric_rows.append({
            "visitorid": u, "method": model.name,
            "precision@5": precision_at_k(items, rel, k), "recall@5": recall_at_k(items, rel, k),
            "f1@5": f1_at_k(items, rel, k), "ndcg@5": ndcg_at_k(items, rel, k),
            "hit@5": float(len(set(items) & rel) > 0), "n_relevant": len(rel),
            "time_ms": dt_ms, "fallback": model.last_info.get("fallback", False),
            "n_padded": model.last_info.get("n_padded", 0)})
        for rank, (item, score) in enumerate(recs, 1):
            rec_rows.append({"visitorid": u, "method": model.name, "rank": rank, "itemid": item,
                             "score": score, "is_hit": item in rel})
        if progress_every and n % progress_every == 0:
            print(f"   {model.name}: {n}/{len(eval_users)} users")
    return pd.DataFrame(metric_rows), pd.DataFrame(rec_rows)


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, seed=SEED):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def summarise(per_user_eval, metrics=METRICS):
    """Comparison table (means, 95% bootstrap CIs, timing), same as the Phase 1 notebook."""
    rows = []
    for name, g in per_user_eval.groupby("method", sort=False):
        row = {"method": name, "n_users": len(g)}
        for mtr in metrics:
            row[mtr] = g[mtr].mean()
            lo, hi = bootstrap_ci(g[mtr])
            row[f"{mtr}_ci_low"], row[f"{mtr}_ci_high"] = lo, hi
        row["avg_time_ms"] = g["time_ms"].mean()
        row["median_time_ms"] = g["time_ms"].median()
        row["fallback_pct"] = 100 * g["fallback"].mean()
        row["padded_pct"] = 100 * (g["n_padded"] > 0).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def ci_overlap_table(summary, metric="ndcg@5", methods=None):
    """Pairwise comparison of methods on one metric using their 95% CIs."""
    s = summary.set_index("method")
    methods = methods or list(s.index)
    rows = []
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            lo_a, hi_a = s.loc[a, f"{metric}_ci_low"], s.loc[a, f"{metric}_ci_high"]
            lo_b, hi_b = s.loc[b, f"{metric}_ci_low"], s.loc[b, f"{metric}_ci_high"]
            overlap = not (lo_a > hi_b or hi_a < lo_b)
            better = a if s.loc[a, metric] >= s.loc[b, metric] else b
            rows.append({"comparison": f"{a} vs {b}", "higher": better,
                         "difference": abs(s.loc[a, metric] - s.loc[b, metric]),
                         "verdict": "CIs overlap: not clearly different" if overlap
                         else f"CIs separate: {better} is likely better"})
    return pd.DataFrame(rows)


def recommendation_overlap(recommendations, methods):
    """Average Jaccard overlap between the top-5 lists of each pair of methods."""
    lists = {m: g.groupby("visitorid")["itemid"].agg(set)
             for m, g in recommendations.groupby("method") if m in methods}
    mat = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for a in methods:
        for b in methods:
            if a not in lists or b not in lists:
                continue
            users = lists[a].index.intersection(lists[b].index)
            j = [len(lists[a][u] & lists[b][u]) / len(lists[a][u] | lists[b][u]) for u in users]
            mat.loc[a, b] = float(np.mean(j)) if j else np.nan
    return mat
