"""Toy-example and structural tests for all recommenders (no saved outputs needed)."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.baseline_recommender import ItemCFRecommender, PopularityRecommender, popularity_ranking
from src.graph_builder import build_bipartite_graph, extract_subgraph, graph_statistics, user_node
from src.graph_recommender import CommonNeighborsRecommender, PersonalizedPageRankRecommender


def _models(inter):
    pop = popularity_ranking(inter)
    G = build_bipartite_graph(inter)
    return [ItemCFRecommender(n_neighbors=10).fit(inter), CommonNeighborsRecommender(G, pop),
            PersonalizedPageRankRecommender(G, pop), PopularityRecommender(inter)], G


def test_toy_example_charger_first(toy_interactions):
    models, _ = _models(toy_interactions)
    for m in models[:3]:                       # Asha (user 1) should get the Charger (103) first
        assert m.recommend(1, 1)[0][0] == 103, m.name


def test_cf_similarity_matches_sklearn(toy_interactions):
    cf = ItemCFRecommender(n_neighbors=10).fit(toy_interactions)
    R = toy_interactions.pivot_table(index="visitorid", columns="itemid", values="weight", fill_value=0)
    S_ref = cosine_similarity(R.T.values)
    np.fill_diagonal(S_ref, 0)
    assert np.allclose(cf.S.toarray(), S_ref, atol=1e-6)


def test_cn_scores_hand_computed(toy_interactions):
    models, G = _models(toy_interactions)
    cn = models[1]
    sims = dict(cn.similar_users(user_node(1)))
    assert sims == {"U_2": 2, "U_4": 1}
    recs = dict(cn.recommend(1, 2))
    assert recs[103] == 2 * 3 + 1 * 1          # CN(Ben)*3 + CN(Dev)*1
    assert recs[105] == 1 * 1


def test_ppr_is_distribution(toy_interactions):
    models, _ = _models(toy_interactions)
    pr = models[2].scores_for(user_node(1))
    assert abs(sum(pr.values()) - 1) < 1e-6


def test_exclusion_length_and_order(toy_interactions):
    models, _ = _models(toy_interactions)
    seen = set(toy_interactions.loc[toy_interactions["visitorid"] == 2, "itemid"])
    for m in models:
        recs = m.recommend(2, 2)
        items, scores = [i for i, _ in recs], [s for _, s in recs]
        assert len(items) == 2 and len(set(items)) == 2, m.name
        assert not set(items) & seen, m.name
        assert scores == sorted(scores, reverse=True), m.name


def test_cold_start_fallback(toy_interactions):
    models, _ = _models(toy_interactions)
    for m in models[:3]:
        recs = m.recommend(999, 3)
        assert len(recs) == 3 and m.last_info["fallback"], m.name


def test_graph_structure(toy_interactions):
    G = build_bipartite_graph(toy_interactions)
    stats, _, _ = graph_statistics(G)
    assert stats["is_bipartite"] and stats["user_nodes"] == 4 and stats["product_nodes"] == 5
    assert stats["edges"] == len(toy_interactions)
    assert G["U_1"]["P_101"]["weight"] == 5
    H = extract_subgraph(G, "U_1", max_items=1, max_users_per_item=1)
    assert "U_1" in H and "P_101" in H and H.number_of_nodes() == 3
