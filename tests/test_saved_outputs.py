"""Tests against the real saved Phase 1 outputs: loading, consistency and reproduction."""
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import data_loader as DL
from src.engine import compare_with_saved
from src.graph_builder import load_graph
from tests.conftest import needs_outputs

pytestmark = needs_outputs


# ---------------------------------------------------------------- loading
def test_all_expected_files_present():
    status = DL.check_outputs()
    assert status["exists"].all(), status[~status["exists"]]


def test_results_load_without_issues(results):
    assert results["issues"] == []
    assert set(results["evaluation_summary"]["method"]) == set(C.ALL_METHODS)
    assert (results["test_results"]["status"] == "PASS").all()


def test_engine_uses_saved_artifacts(engine):
    assert engine.graph_source == "saved"
    assert engine.cf_source == "saved"
    assert engine.issues == []


def test_graph_matches_training_data(engine):
    G, inter = engine.G, engine.interactions
    assert G.number_of_edges() == len(inter)
    sample = inter.sample(300, random_state=0)
    for r in sample.itertuples():
        assert abs(G[f"U_{r.visitorid}"][f"P_{r.itemid}"]["weight"] - r.weight) < 1e-6


def test_saved_similarity_shape(engine):
    S = engine.cf.S
    assert S.shape == (len(engine.cf.item_ids),) * 2
    assert (S.diagonal() == 0).all()
    assert S.getnnz(axis=1).max() <= engine.settings["CF_NEIGHBORS"]


# ---------------------------------------------------------------- graceful failure
def _copy_outputs(tmp_path):
    dst = tmp_path / "outputs"
    shutil.copytree(C.OUTPUTS_DIR, dst)
    return str(dst)


def test_missing_and_corrupted_results(tmp_path):
    out = _copy_outputs(tmp_path)
    os.remove(os.path.join(out, "results", "scalability_results.csv"))
    with open(os.path.join(out, "results", "graph_stats.json"), "w") as f:
        f.write("{not json")
    with open(os.path.join(out, "results", "evaluation_summary.csv"), "w") as f:
        f.write("a,b\n1,2\n")
    res = DL.load_results(out)
    keys = {i["key"] for i in res["issues"]}
    assert {"scalability_results", "graph_stats", "evaluation_summary"} <= keys
    assert res["scalability_results"] is None and res["evaluation_summary"] is None
    assert res["per_user_evaluation"] is not None       # other files still load


def test_corrupted_graph_is_rebuilt(tmp_path, engine):
    bad = tmp_path / "graph.pkl"
    bad.write_bytes(b"this is not a pickle")
    G, source, msg = load_graph(str(bad), engine.interactions)
    assert source == "rebuilt" and G.number_of_edges() == engine.G.number_of_edges()
    G2, source2, _ = load_graph(str(tmp_path / "missing.pkl"), engine.interactions)
    assert source2 == "rebuilt"


def test_wrong_graph_is_rebuilt(tmp_path, engine):
    import networkx as nx
    p = tmp_path / "g.pkl"
    with open(p, "wb") as f:
        pickle.dump(nx.path_graph(3), f)
    _, source, _ = load_graph(str(p), engine.interactions)
    assert source == "rebuilt"


def test_engine_falls_back_when_models_missing(tmp_path):
    from src.engine import RecommendationEngine
    out = _copy_outputs(tmp_path)
    os.remove(os.path.join(out, "models", "bipartite_graph.pkl"))
    os.remove(os.path.join(out, "models", "item_similarity.npz"))
    e = RecommendationEngine(out)
    assert e.graph_source == "rebuilt" and e.cf_source == "refit"
    ref = RecommendationEngine()
    u = sorted(ref.ground_truth)[0]
    for m in [C.CF_NAME, C.CN_NAME]:
        assert e.recommend(m, u)[0]["itemid"].tolist() == ref.recommend(m, u)[0]["itemid"].tolist()


def test_engine_error_without_training_data(tmp_path):
    from src.engine import EngineLoadError, RecommendationEngine
    out = _copy_outputs(tmp_path)
    os.remove(os.path.join(out, "data", "train_interactions.csv.gz"))
    with pytest.raises(EngineLoadError):
        RecommendationEngine(out)


# ---------------------------------------------------------------- recommendations on real data
def test_real_users_valid_recommendations(engine):
    rng = np.random.default_rng(1)
    users = rng.choice(engine.train_users, 25, replace=False)
    catalog = set(engine.cf.item_ids.tolist())
    for u in users:
        seen = engine.seen_items(u)
        for m in C.ALL_METHODS:
            df, info = engine.recommend(m, u)
            items = df["itemid"].tolist()
            assert len(items) == C.TOP_K and len(set(items)) == C.TOP_K, (m, u)
            assert not set(items) & seen, (m, u, "recommended an already-seen product")
            assert set(items) <= catalog, (m, u)
            assert df["score"].is_monotonic_decreasing, (m, u)


def _reproduce(engine, results, methods, users):
    """Every live list must equal the saved one, except swaps between exactly tied scores."""
    rec = results["recommendations"]
    outcome = {"exact": 0, "tie": 0}
    for u in users:
        for m in methods:
            live = engine.recommend(m, u)[0]
            saved = rec[(rec["visitorid"] == u) & (rec["method"] == m)].sort_values("rank")
            verdict = compare_with_saved(live, saved)
            assert verdict != "different", (m, u, live["itemid"].tolist(), saved["itemid"].tolist())
            outcome[verdict] += 1
            gt = engine.ground_truth.get(u, set())
            assert (saved["is_hit"].astype(bool).tolist()
                    == [i in gt for i in saved["itemid"]]), (m, u, "hit flags differ")
    print(f"\nreproduction: {outcome}")
    return outcome


def test_compare_with_saved_logic():
    import pandas as pd
    a = pd.DataFrame({"itemid": [1, 2, 3], "score": [3.0, 2.0, 1.0]})
    assert compare_with_saved(a, a) == "exact"
    tie = pd.DataFrame({"itemid": [1, 2, 9], "score": [3.0, 2.0, 1.0]})      # boundary item swapped, same score
    assert compare_with_saved(a, tie) == "tie"
    wrong = pd.DataFrame({"itemid": [9, 2, 3], "score": [3.0, 2.0, 1.0]})    # top item differs, unique score
    assert compare_with_saved(a, wrong) == "different"
    other = pd.DataFrame({"itemid": [1, 2, 3], "score": [3.0, 2.5, 1.0]})
    assert compare_with_saved(a.assign(itemid=[1, 2, 4]), other) == "different"


def test_reproduces_phase1_fast_methods(engine, results):
    """Live CF / CN / Popularity == saved Phase 1 recommendations for all evaluated users."""
    users = sorted(results["recommendations"]["visitorid"].unique())
    _reproduce(engine, results, [C.CF_NAME, C.CN_NAME, C.POP_NAME], users)


def test_reproduces_phase1_pagerank(engine, results):
    """Live PageRank == saved results (40 users by default; set SDS_FULL_TEST=1 for all 300)."""
    users = sorted(results["recommendations"]["visitorid"].unique())
    if not os.environ.get("SDS_FULL_TEST"):
        users = users[::max(1, len(users) // 40)]
    _reproduce(engine, results, [C.PPR_NAME], users)


def test_cold_start_unknown_user(engine):
    unknown = int(engine.train_users.max()) + 10 ** 9
    for m in C.PERSONALISED_METHODS:
        df, info = engine.recommend(m, unknown)
        assert len(df) == C.TOP_K and info["fallback"]


def test_user_views(engine):
    u = sorted(engine.ground_truth)[0]
    h = engine.user_history(u)
    assert len(h) == engine.user_summary(u)["n_products"] >= C.MIN_USER_ITEMS
    assert set(h["strongest_event"]) <= {"view", "add to cart", "transaction"}
    sim = engine.similar_users(u, 5)
    assert len(sim) <= 5 and sim["shared_products"].is_monotonic_decreasing
    df, _ = engine.recommend(C.CF_NAME, u)
    ex = engine.cf_explanations(u, df["itemid"].tolist())
    assert set(ex["recommended_item"]) <= set(df["itemid"])
    assert set(ex["because_of_item"]) <= engine.seen_items(u)


def test_subgraph_with_recommendations(engine):
    u = sorted(engine.ground_truth)[0]
    recs = engine.recommend(C.PPR_NAME, u)[0]["itemid"].tolist()
    H, center, rec_nodes = engine.subgraph(u, 6, 3, recommended=recs)
    assert center in H and len(rec_nodes) == len(recs)
    for p in rec_nodes:
        assert p in H and H.degree(p) >= 1
    assert engine.subgraph(10 ** 12) is None
