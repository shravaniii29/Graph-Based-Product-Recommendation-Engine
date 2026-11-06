"""Scalability analysis.

`run_scalability` is the fixed Phase 1 experiment (nested user samples drawn from
the filtered active-user population). Re-running it needs the raw events, so the
dashboard does NOT re-run it: it reads the saved `scalability_results.csv`.
`growth_summary` and `long_format` only reshape those saved results.
"""
import gc
import os
import time

import numpy as np
import pandas as pd
import psutil

from .baseline_recommender import ItemCFRecommender, popularity_ranking
from .config import N_SCALE_EVAL_USERS, PERSONALISED_METHODS, SCALE_FRACTIONS, SEED
from .data_preprocessing import aggregate_interactions, build_ground_truth
from .evaluation import evaluate_recommender
from .graph_builder import build_bipartite_graph
from .graph_recommender import CommonNeighborsRecommender, PersonalizedPageRankRecommender


def rss_mb():
    """Current memory used by this Python process (MB)."""
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def run_scalability(base_interactions, events_df, cutoff_ts, fractions=SCALE_FRACTIONS,
                    n_eval=N_SCALE_EVAL_USERS, verbose=True):
    """Nested user samples drawn from the users who passed filtering (Phase 1 Section 9)."""
    rng = np.random.default_rng(SEED)
    pool = np.sort(base_interactions["visitorid"].unique())
    shuffled = rng.permutation(pool)
    results = []
    for frac in fractions:
        if verbose:
            print(f"\n=== {int(frac*100)}% of active users ===")
        gc.collect()
        sample_users = set(shuffled[: max(1, int(round(frac * len(shuffled))))])
        ev = events_df[events_df["visitorid"].isin(sample_users)]
        tr_ev, te_ev = ev[ev["timestamp"] < cutoff_ts], ev[ev["timestamp"] >= cutoff_ts]

        t0 = time.perf_counter()
        _ = aggregate_interactions(tr_ev)                      # preprocessing cost for this sample
        prep_time = time.perf_counter() - t0
        inter = base_interactions[base_interactions["visitorid"].isin(sample_users)].reset_index(drop=True)
        if len(inter) == 0:
            if verbose:
                print("   no interactions for this sample; skipped")
            continue
        gt, _ = build_ground_truth(te_ev, inter)

        t0 = time.perf_counter(); cf = ItemCFRecommender().fit(inter); cf_time = time.perf_counter() - t0
        gc.collect(); m0 = rss_mb()
        t0 = time.perf_counter(); Gs = build_bipartite_graph(inter); g_time = time.perf_counter() - t0
        g_mem = rss_mb() - m0

        pop = popularity_ranking(inter)
        methods = [cf, CommonNeighborsRecommender(Gs, pop), PersonalizedPageRankRecommender(Gs, pop)]
        users_gt = np.array(sorted(gt.keys()))
        eval_users = sorted(np.random.default_rng(SEED).choice(users_gt, size=min(n_eval, len(users_gt)),
                                                                 replace=False).tolist()) if len(users_gt) else []
        row = {"fraction": frac, "dataset_pct": int(frac * 100),
               "raw_events": len(ev), "users": inter["visitorid"].nunique(), "products": inter["itemid"].nunique(),
               "interactions": len(inter), "graph_nodes": Gs.number_of_nodes(), "graph_edges": Gs.number_of_edges(),
               "preprocess_time_s": prep_time, "cf_build_time_s": cf_time, "cf_similarity_mb": cf.similarity_matrix_mb(),
               "graph_build_time_s": g_time, "graph_memory_mb": g_mem, "process_rss_mb": rss_mb(),
               "n_eval_users": len(eval_users)}
        for m in methods:
            if eval_users:
                mdf, _ = evaluate_recommender(m, eval_users, gt, progress_every=0)
                row[f"{m.name}|time_ms"] = mdf["time_ms"].mean()
                row[f"{m.name}|precision@5"] = mdf["precision@5"].mean()
                row[f"{m.name}|ndcg@5"] = mdf["ndcg@5"].mean()
            else:
                row[f"{m.name}|time_ms"] = row[f"{m.name}|precision@5"] = row[f"{m.name}|ndcg@5"] = np.nan
        results.append(row)
        if verbose:
            print(f"   users={row['users']:,} products={row['products']:,} edges={row['graph_edges']:,} | "
                  f"eval users={len(eval_users)} | graph build {g_time:.2f}s, {g_mem:+.0f} MB")
        del Gs, cf, methods, inter, ev, tr_ev, te_ev
        gc.collect()
    return pd.DataFrame(results)


def long_format(scal, measure, methods=PERSONALISED_METHODS):
    """Saved wide columns like 'Item-Item CF|time_ms' -> long rows (dataset_pct, method, value)."""
    cols = [f"{m}|{measure}" for m in methods if f"{m}|{measure}" in scal.columns]
    out = scal.melt(id_vars=["dataset_pct", "users", "interactions", "graph_nodes"], value_vars=cols,
                    var_name="method", value_name=measure)
    out["method"] = out["method"].str.split("|").str[0]
    return out


def growth_summary(scal, methods=PERSONALISED_METHODS):
    """How much each quantity grew from the smallest to the largest sample."""
    s = scal.sort_values("fraction")
    if len(s) < 2:
        return pd.DataFrame()
    a, b = s.iloc[0], s.iloc[-1]

    def ratio(col):
        return b[col] / a[col] if a[col] else np.nan

    rows = [
        {"quantity": "Users", "from": a["users"], "to": b["users"], "growth": ratio("users")},
        {"quantity": "Interactions (edges)", "from": a["interactions"], "to": b["interactions"], "growth": ratio("interactions")},
        {"quantity": "Graph nodes", "from": a["graph_nodes"], "to": b["graph_nodes"], "growth": ratio("graph_nodes")},
        {"quantity": "Graph build time (s)", "from": a["graph_build_time_s"], "to": b["graph_build_time_s"], "growth": ratio("graph_build_time_s")},
        {"quantity": "CF build time (s)", "from": a["cf_build_time_s"], "to": b["cf_build_time_s"], "growth": ratio("cf_build_time_s")},
    ]
    for m in methods:
        col = f"{m}|time_ms"
        if col in s.columns:
            rows.append({"quantity": f"{m} time per user (ms)", "from": a[col], "to": b[col], "growth": ratio(col)})
    return pd.DataFrame(rows)
