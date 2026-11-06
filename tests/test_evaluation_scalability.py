import numpy as np
import pandas as pd

from src import config as C
from src.evaluation import (METRICS, bootstrap_ci, ci_overlap_table, f1_at_k, ndcg_at_k, precision_at_k,
                            recall_at_k, recommendation_overlap, summarise)
from src.scalability import growth_summary, long_format, run_scalability
from tests.conftest import needs_outputs


def test_metrics_hand_computed():
    rec, rel = [10, 20, 30, 40, 50], {20, 99}
    assert precision_at_k(rec, rel) == 0.2 and recall_at_k(rec, rel) == 0.5
    assert abs(f1_at_k(rec, rel) - 2 * 0.2 * 0.5 / 0.7) < 1e-12
    assert abs(ndcg_at_k(rec, rel) - (1 / np.log2(3)) / (1 + 1 / np.log2(3))) < 1e-12
    assert ndcg_at_k([1, 2, 3, 4, 5], {1, 2}) == 1.0
    assert ndcg_at_k([9, 8, 7, 6, 5], {1}) == 0.0
    assert ndcg_at_k([9, 1, 7, 6, 5], {1}) < ndcg_at_k([1, 9, 7, 6, 5], {1})


def test_bootstrap_ci_contains_mean():
    vals = np.random.default_rng(0).random(200)
    lo, hi = bootstrap_ci(vals)
    assert lo < vals.mean() < hi


@needs_outputs
def test_saved_summary_recomputes_exactly(results):
    """The comparison table recomputed from per-user results equals the saved one."""
    again = summarise(results["per_user_evaluation"]).set_index("method")
    saved = results["evaluation_summary"].set_index("method")
    for m in saved.index:
        for col in METRICS + [f"{k}_ci_low" for k in METRICS] + [f"{k}_ci_high" for k in METRICS] + ["avg_time_ms"]:
            assert np.isclose(again.loc[m, col], saved.loc[m, col], rtol=1e-9, atol=1e-12), (m, col)


@needs_outputs
def test_saved_metrics_consistent_with_recommendations(results):
    pue, rec = results["per_user_evaluation"], results["recommendations"]
    hits = rec.groupby(["visitorid", "method"])["is_hit"].sum().reset_index()
    merged = hits.merge(pue, on=["visitorid", "method"])
    assert len(merged) == len(pue)
    assert np.allclose(merged["is_hit"] / 5, merged["precision@5"])
    assert pue.groupby("method")["visitorid"].nunique().nunique() == 1   # same users for each method
    for mtr in METRICS:
        assert pue[mtr].between(0, 1).all()


@needs_outputs
def test_comparison_helpers(results):
    t = ci_overlap_table(results["evaluation_summary"], "ndcg@5", C.PERSONALISED_METHODS)
    assert len(t) == 3 and t["verdict"].str.len().gt(0).all()
    mat = recommendation_overlap(results["recommendations"], C.ALL_METHODS)
    assert np.allclose(np.diag(mat.values.astype(float)), 1.0)
    assert ((mat.values >= 0) & (mat.values <= 1)).all()


@needs_outputs
def test_saved_scalability_results(results):
    s = results["scalability_results"].sort_values("fraction")
    assert s["fraction"].tolist() == C.SCALE_FRACTIONS
    for col in ["users", "interactions", "graph_nodes", "raw_events"]:
        assert s[col].is_monotonic_increasing, col
    assert (s["graph_build_time_s"] > 0).all()
    gs = results["graph_stats"]
    full = s.iloc[-1]
    assert full["interactions"] == gs["edges"] and full["graph_nodes"] == gs["total_nodes"]
    lf = long_format(s, "time_ms")
    assert set(lf["method"]) == set(C.PERSONALISED_METHODS) and len(lf) == 3 * len(s)
    g = growth_summary(s)
    assert (g["growth"] > 0).all()


def test_run_scalability_small(toy_interactions):
    """The scalability experiment runs end to end on a tiny synthetic dataset."""
    rng = np.random.default_rng(0)
    rows = []
    for u in range(60):
        for _ in range(12):
            rows.append((int(rng.integers(0, 1000)), u, int(rng.integers(0, 40)), 1))
    ev = pd.DataFrame(rows, columns=["timestamp", "visitorid", "itemid", "weight"])
    cutoff = 800
    from src.data_preprocessing import aggregate_interactions
    base = aggregate_interactions(ev[ev["timestamp"] < cutoff])
    out = run_scalability(base, ev, cutoff, fractions=[0.5, 1.0], n_eval=10, verbose=False)
    assert out["fraction"].tolist() == [0.5, 1.0]
    assert out["users"].is_monotonic_increasing
    assert out.iloc[-1]["interactions"] == len(base)
