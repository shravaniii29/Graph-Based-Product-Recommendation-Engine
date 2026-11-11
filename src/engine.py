"""RecommendationEngine: the one object the dashboard talks to.

It loads the saved Phase 1 artefacts once and exposes simple methods:
user history, live recommendations from all methods, explanations and subgraphs.
No algorithm logic lives here; it only calls the validated recommenders.
"""
import time

import numpy as np
import pandas as pd

from . import config as C
from . import data_loader as DL
from .baseline_recommender import ItemCFRecommender, PopularityRecommender, popularity_ranking
from .data_preprocessing import ground_truth_from_frame
from .graph_builder import extract_subgraph, load_graph, node_id, product_node, user_node
from .graph_recommender import CommonNeighborsRecommender, PersonalizedPageRankRecommender

EVENT_NAMES = {1: "view", 3: "add to cart", 5: "transaction"}


class EngineLoadError(RuntimeError):
    """Raised when the data needed for live recommendations cannot be loaded."""


class RecommendationEngine:
    def __init__(self, outputs_dir=None):
        self.outputs_dir = outputs_dir or C.OUTPUTS_DIR
        self.issues, self.notes = [], []
        self.settings = C.load_run_config(DL.path_for("results", "config.json", self.outputs_dir))
        s = self.settings

        t0 = time.perf_counter()
        inter = DL.load_csv("train_interactions", self.outputs_dir, self.issues)
        if inter is None:
            raise EngineLoadError("train_interactions.csv.gz is missing or unreadable, so live "
                                  "recommendations are not possible. " + _issue_text(self.issues))
        self.interactions = inter
        self.train_users = np.sort(inter["visitorid"].unique())
        self.train_user_set = set(self.train_users.tolist())

        gt_df = DL.load_csv("test_ground_truth", self.outputs_dir, self.issues)
        self.ground_truth = ground_truth_from_frame(gt_df) if gt_df is not None else {}

        pop_df = DL.load_csv("item_popularity", self.outputs_dir, self.issues)
        if pop_df is None:
            pop_df = (inter.groupby("itemid").agg(n_users=("visitorid", "nunique"), total_weight=("weight", "sum"))
                      .reset_index())
        self.item_popularity = pop_df.set_index("itemid")["n_users"].to_dict()
        self.popular_items = popularity_ranking(inter)

        # ---- graph (saved pickle, or rebuilt from the same interactions) ----
        self.G, self.graph_source, msg = load_graph(DL.path_for(*DL.GRAPH_FILE, self.outputs_dir), inter)
        self.notes.append(msg)

        # ---- item-item CF from the saved similarity matrix ----
        cf_kwargs = dict(n_neighbors=s["CF_NEIGHBORS"], min_history=s["MIN_HISTORY"])
        S = DL.load_similarity(self.outputs_dir, self.issues)
        idx = DL.load_json("cf_index", self.outputs_dir, self.issues)
        try:
            if S is None or idx is None:
                raise ValueError("saved similarity matrix or index missing")
            self.cf = ItemCFRecommender.from_saved(inter, S, idx["user_ids"], idx["item_ids"], **cf_kwargs)
            self.cf_source = "saved"
            self.notes.append("Loaded the saved item-item similarity matrix.")
        except Exception as e:
            self.cf = ItemCFRecommender(**cf_kwargs).fit(inter)
            self.cf_source = "refit"
            self.notes.append(f"Saved CF model unusable ({e}); recomputed it from the training data.")

        self.pop = PopularityRecommender(inter)
        self.cn = CommonNeighborsRecommender(self.G, self.popular_items, top_users=s["CN_TOP_USERS"],
                                             min_history=s["MIN_HISTORY"])
        self.ppr = PersonalizedPageRankRecommender(self.G, self.popular_items, alpha=s["PPR_ALPHA"],
                                                   max_iter=s["PPR_MAX_ITER"], tol=s["PPR_TOL"],
                                                   min_history=s["MIN_HISTORY"])
        self.models = {m.name: m for m in [self.cf, self.cn, self.ppr, self.pop]}
        self.load_time_s = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------
    def has_user(self, user_id):
        return int(user_id) in self.train_user_set

    def user_history(self, user_id):
        h = self.interactions[self.interactions["visitorid"] == int(user_id)].copy()
        if h.empty:
            return h
        h["strongest_event"] = h["max_event_weight"].map(EVENT_NAMES).fillna("other")
        h["last_interaction"] = pd.to_datetime(h["last_timestamp"], unit="ms")
        h["product_popularity"] = h["itemid"].map(self.item_popularity)
        h = h.sort_values(["weight", "last_timestamp"], ascending=[False, False])
        return h[["itemid", "weight", "raw_weight", "n_events", "strongest_event", "last_interaction",
                  "product_popularity"]].reset_index(drop=True)

    def user_summary(self, user_id):
        h = self.user_history(user_id)
        gt = self.ground_truth.get(int(user_id), set())
        return {
            "n_products": len(h),
            "n_events": int(h["n_events"].sum()) if len(h) else 0,
            "total_weight": float(h["weight"].sum()) if len(h) else 0.0,
            "n_purchased": int((h["strongest_event"] == "transaction").sum()) if len(h) else 0,
            "n_carted": int((h["strongest_event"] == "add to cart").sum()) if len(h) else 0,
            "first": h["last_interaction"].min() if len(h) else None,
            "last": h["last_interaction"].max() if len(h) else None,
            "n_future_items": len(gt),
            "graph_degree": self.G.degree(user_node(int(user_id))) if user_node(int(user_id)) in self.G else 0,
        }

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    def recommend(self, method, user_id, k=None):
        """Live top-k from one method, as a table, plus timing and fallback info."""
        k = k or self.settings["TOP_K"]
        model = self.models[method]
        user_id = int(user_id)
        t0 = time.perf_counter()
        recs = model.recommend(user_id, k)
        dt_ms = (time.perf_counter() - t0) * 1000
        gt = self.ground_truth.get(user_id, set())
        df = pd.DataFrame(recs, columns=["itemid", "score"])
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        df["product_popularity"] = df["itemid"].map(self.item_popularity)
        df["in_future_interactions"] = df["itemid"].isin(gt)
        info = dict(model.last_info)
        info["time_ms"] = dt_ms
        info["has_ground_truth"] = bool(gt)
        return df, info

    def recommend_all(self, user_id, methods=None):
        methods = methods or C.ALL_METHODS
        return {m: self.recommend(m, user_id) for m in methods}

    def seen_items(self, user_id):
        return set(self.interactions.loc[self.interactions["visitorid"] == int(user_id), "itemid"].tolist())

    def cf_explanations(self, user_id, item_ids, top_n=3):
        rows = []
        for it in item_ids:
            for hist_item, w, sim, contrib in self.cf.explain(int(user_id), int(it), top_n=top_n):
                rows.append({"recommended_item": int(it), "because_of_item": hist_item, "user_weight": w,
                             "cosine_similarity": sim, "contribution": contrib})
        return pd.DataFrame(rows)

    def similar_users(self, user_id, n=10):
        unode = user_node(int(user_id))
        if unode not in self.G:
            return pd.DataFrame(columns=["user", "shared_products", "their_products"])
        sims = self.cn.similar_users(unode)[:n]
        return pd.DataFrame([{"user": node_id(v), "shared_products": c, "their_products": self.G.degree(v)}
                             for v, c in sims])

    # ------------------------------------------------------------------
    # Graph views
    # ------------------------------------------------------------------
    def subgraph(self, user_id, max_items=8, max_users_per_item=4, recommended=None):
        """2-hop neighbourhood of the user, optionally with recommended products attached.

        Recommended products are added with their edges to any user already in the
        subgraph, so the picture shows how the graph reaches them.
        """
        center = user_node(int(user_id))
        if center not in self.G:
            return None
        H = extract_subgraph(self.G, center, max_items=max_items, max_users_per_item=max_users_per_item)
        rec_nodes = []
        for it in recommended or []:
            p = product_node(int(it))
            if p not in self.G:
                continue
            rec_nodes.append(p)
            if p not in H:
                users_here = [u for u in H if H.nodes[u]["node_type"] == "user" and self.G.has_edge(u, p)]
                if not users_here:   # bring in the strongest user linking it to the picture
                    linkers = sorted(self.G[p], key=lambda u: -self.G[p][u]["weight"])[:1]
                    users_here = linkers
                H.add_node(p, **self.G.nodes[p])
                for u in users_here:
                    if u not in H:
                        H.add_node(u, **self.G.nodes[u])
                    H.add_edge(u, p, weight=self.G[u][p]["weight"])
        return H, center, rec_nodes


def compare_with_saved(live, saved, rtol=1e-5):
    """Compare a live top-k with the saved Phase 1 top-k.

    Returns "exact" (same products, same order), "tie" (same scores; products differ
    only where several products have exactly the same score, so either choice is
    equally correct) or "different".

    Why ties happen: CF selects the top-k with numpy's argpartition, whose choice
    between equal values can differ between NumPy versions and runtime environments.
    """
    li, si = [int(x) for x in live["itemid"]], [int(x) for x in saved["itemid"]]
    if li == si:
        return "exact"
    ls, ss = np.asarray(live["score"], float), np.asarray(saved["score"], float)
    if len(ls) != len(ss) or not np.allclose(ls, ss, rtol=rtol, atol=1e-12):
        return "different"
    for a, b, s in zip(li, si, ls):
        if a != b and np.isclose(ls, s, rtol=rtol, atol=1e-12).sum() < 2:
            # a differing position whose score is unique within the list could still be a tie with
            # an item just outside the top-k; accept it only at the boundary score
            if not np.isclose(s, ls[-1], rtol=rtol, atol=1e-12):
                return "different"
    return "tie"


def _issue_text(issues):
    return " ".join(f"[{i['file']}: {i['message']}]" for i in issues)
