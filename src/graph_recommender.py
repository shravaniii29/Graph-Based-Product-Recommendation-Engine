"""Graph-based recommenders: Common Neighbors and Personalized PageRank.

Ported unchanged from Phase 1 notebook Section 7.
"""
import heapq
from collections import Counter, defaultdict

import networkx as nx

from .baseline_recommender import fill_with_popular
from .config import (CN_NAME, CN_TOP_USERS, MIN_HISTORY, PPR_ALPHA, PPR_MAX_ITER, PPR_NAME,
                     PPR_TOL, TOP_K)
from .graph_builder import node_id, user_node


class CommonNeighborsRecommender:
    name = CN_NAME

    def __init__(self, G, popular_items, top_users=CN_TOP_USERS, min_history=MIN_HISTORY):
        self.G, self.popular_items = G, popular_items
        self.top_users, self.min_history = top_users, min_history
        self.last_info = {}

    def similar_users(self, unode):
        """CN(u, v) for every user v that shares at least one product with u."""
        cn = Counter()
        for p in self.G[unode]:
            for v in self.G[p]:
                if v != unode:
                    cn[v] += 1
        # sort by count desc, then node id for deterministic ties
        return sorted(cn.items(), key=lambda x: (-x[1], x[0]))[: self.top_users]

    def recommend(self, user_id, k=TOP_K):
        self.last_info = {"fallback": False, "n_padded": 0}
        unode = user_node(user_id)
        if unode not in self.G or self.G.degree(unode) < self.min_history:
            self.last_info["fallback"] = True
            seen = {node_id(p) for p in self.G[unode]} if unode in self.G else set()
            return fill_with_popular([], self.popular_items, seen, k)
        seen_nodes = set(self.G[unode])
        scores = defaultdict(float)
        for v, c in self.similar_users(unode):
            for p, data in self.G[v].items():
                if p not in seen_nodes:
                    scores[p] += c * data["weight"]
        top = heapq.nsmallest(k, scores.items(), key=lambda x: (-x[1], x[0]))
        recs = [(node_id(p), float(s)) for p, s in top]
        if len(recs) < k:
            self.last_info["n_padded"] = k - len(recs)
            recs = fill_with_popular(recs, self.popular_items, {node_id(p) for p in seen_nodes}, k)
        return recs


class PersonalizedPageRankRecommender:
    name = PPR_NAME

    def __init__(self, G, popular_items, alpha=PPR_ALPHA, max_iter=PPR_MAX_ITER, tol=PPR_TOL,
                 min_history=MIN_HISTORY):
        self.G, self.popular_items = G, popular_items
        self.alpha, self.max_iter, self.tol, self.min_history = alpha, max_iter, tol, min_history
        self.product_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "product"]
        self.last_info = {}

    def scores_for(self, unode):
        """Full PPR vector personalised on one user node (NetworkX implementation)."""
        return nx.pagerank(self.G, alpha=self.alpha, personalization={unode: 1.0},
                           weight="weight", max_iter=self.max_iter, tol=self.tol)

    def recommend(self, user_id, k=TOP_K):
        self.last_info = {"fallback": False, "n_padded": 0}
        unode = user_node(user_id)
        if unode not in self.G or self.G.degree(unode) < self.min_history:
            self.last_info["fallback"] = True
            seen = {node_id(p) for p in self.G[unode]} if unode in self.G else set()
            return fill_with_popular([], self.popular_items, seen, k)
        seen_nodes = set(self.G[unode])
        try:
            pr = self.scores_for(unode)
        except nx.PowerIterationFailedConvergence:
            print(f"PPR did not converge for user {user_id}; using popularity fallback.")
            self.last_info["fallback"] = True
            return fill_with_popular([], self.popular_items, {node_id(p) for p in seen_nodes}, k)
        cand = ((pr[p], p) for p in self.product_nodes if p not in seen_nodes and pr[p] > 0)
        top = heapq.nlargest(k, cand)
        recs = [(node_id(p), float(s)) for s, p in top]
        if len(recs) < k:
            self.last_info["n_padded"] = k - len(recs)
            recs = fill_with_popular(recs, self.popular_items, {node_id(p) for p in seen_nodes}, k)
        return recs
