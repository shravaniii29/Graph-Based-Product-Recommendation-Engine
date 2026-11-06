"""Baseline: Item-Item Collaborative Filtering (+ popularity reference and helpers).

Ported unchanged from Phase 1 notebook Section 5. Two additions that do not
change any recommendation:
  * `ItemCFRecommender.from_saved(...)` rebuilds the fitted model from the saved
    similarity matrix, so the dashboard does not recompute cosine similarity.
  * `ItemCFRecommender.explain(...)` shows which history items produced a score.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from .config import CF_NAME, CF_NEIGHBORS, MIN_HISTORY, POP_NAME, TOP_K


def popularity_ranking(interactions):
    """Products ordered by number of distinct users in TRAIN (ties broken by itemid for determinism)."""
    pop = interactions.groupby("itemid")["visitorid"].nunique().reset_index(name="n_users")
    pop = pop.sort_values(["n_users", "itemid"], ascending=[False, True])
    return pop["itemid"].to_numpy()


def fill_with_popular(recs, popular_items, exclude, k):
    """Top up a recommendation list with popular unseen items until it has k entries."""
    have = {i for i, _ in recs} | set(exclude)
    for item in popular_items:
        if len(recs) >= k:
            break
        if item not in have:
            recs.append((int(item), 0.0))
            have.add(item)
    return recs


class PopularityRecommender:
    """Non-personalised reference: recommend the most popular unseen products."""
    name = POP_NAME

    def __init__(self, interactions):
        self.popular_items = popularity_ranking(interactions)
        self.seen = interactions.groupby("visitorid")["itemid"].agg(set).to_dict()
        self.last_info = {}

    def recommend(self, user_id, k=TOP_K):
        seen = self.seen.get(user_id, set())
        self.last_info = {"fallback": False, "n_padded": 0}
        return fill_with_popular([], self.popular_items, seen, k)


class ItemCFRecommender:
    """Item-based CF with cosine similarity on the weighted user-item matrix (top-N neighbours per item)."""
    name = CF_NAME

    def __init__(self, n_neighbors=CF_NEIGHBORS, chunk_size=1000, min_history=MIN_HISTORY):
        self.n_neighbors = n_neighbors
        self.chunk_size = chunk_size
        self.min_history = min_history
        self.last_info = {}

    # ---------------- fitting (identical to Phase 1) ----------------
    def _build_matrix(self, interactions, user_ids=None, item_ids=None):
        self.user_ids = np.sort(interactions["visitorid"].unique()) if user_ids is None else np.asarray(user_ids)
        self.item_ids = np.sort(interactions["itemid"].unique()) if item_ids is None else np.asarray(item_ids)
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.item_index = {it: j for j, it in enumerate(self.item_ids)}
        n_u, n_i = len(self.user_ids), len(self.item_ids)
        rows = interactions["visitorid"].map(self.user_index).to_numpy()
        cols = interactions["itemid"].map(self.item_index).to_numpy()
        vals = interactions["weight"].to_numpy(dtype=np.float32)
        self.R = sp.csr_matrix((vals, (rows, cols)), shape=(n_u, n_i), dtype=np.float32)

    def fit(self, interactions):
        # 1) + 2) index maps and sparse user-item matrix R (users x items)
        self._build_matrix(interactions)
        n_i = len(self.item_ids)

        # 3) cosine similarity = dot product of L2-normalised item columns, computed chunk by chunk
        Rn = normalize(self.R.tocsc(), norm="l2", axis=0).tocsr()   # users x items, unit-length columns
        Rn_T = Rn.T.tocsr()                                         # items x users
        k = min(self.n_neighbors, n_i - 1)
        data, indices, indptr = [], [], [0]
        for start in range(0, n_i, self.chunk_size):
            end = min(start + self.chunk_size, n_i)
            block = (Rn_T[start:end] @ Rn).toarray()                # (chunk x items) cosine similarities
            block[np.arange(end - start), np.arange(start, end)] = 0.0   # no self-similarity
            if k > 0:
                top = np.argpartition(-block, k - 1, axis=1)[:, :k]
                top_vals = np.take_along_axis(block, top, axis=1)
            for r in range(end - start):
                if k > 0:
                    keep = top_vals[r] > 0
                    order = np.argsort(top[r][keep])
                    indices.extend(top[r][keep][order]); data.extend(top_vals[r][keep][order])
                indptr.append(len(indices))
        self.S = sp.csr_matrix((np.array(data, dtype=np.float32), np.array(indices, dtype=np.int32),
                                np.array(indptr, dtype=np.int64)), shape=(n_i, n_i))
        self.popular_items = popularity_ranking(interactions)
        return self

    @classmethod
    def from_saved(cls, interactions, similarity, user_ids, item_ids, **kwargs):
        """Rebuild a fitted model from Phase 1 artefacts (no similarity recomputation).

        `similarity` is the saved item_similarity.npz matrix; `user_ids` / `item_ids`
        come from cf_index.json and fix the row/column order of R and S.
        """
        model = cls(**kwargs)
        model._build_matrix(interactions, user_ids=user_ids, item_ids=item_ids)
        n_i = len(model.item_ids)
        if similarity.shape != (n_i, n_i):
            raise ValueError(f"similarity matrix shape {similarity.shape} does not match {n_i} items")
        if model.R.shape[0] != len(model.user_ids) or interactions["visitorid"].map(model.user_index).isna().any():
            raise ValueError("cf_index user ids do not match the training interactions")
        model.S = similarity.tocsr().astype(np.float32)
        model.popular_items = popularity_ranking(interactions)
        return model

    def similarity_matrix_mb(self):
        return (self.S.data.nbytes + self.S.indices.nbytes + self.S.indptr.nbytes) / 1e6

    # ---------------- recommending (identical to Phase 1) ----------------
    def recommend(self, user_id, k=TOP_K):
        self.last_info = {"fallback": False, "n_padded": 0}
        if user_id not in self.user_index:
            self.last_info["fallback"] = True
            return fill_with_popular([], self.popular_items, set(), k)
        u_row = self.R[self.user_index[user_id]]
        seen_idx = u_row.indices
        seen_items = set(self.item_ids[seen_idx])
        if len(seen_idx) < self.min_history:            # insufficient history -> popularity
            self.last_info["fallback"] = True
            return fill_with_popular([], self.popular_items, seen_items, k)

        scores = np.asarray((u_row @ self.S).todense()).ravel()   # sum_i w_ui * sim(i, j)
        scores[seen_idx] = -np.inf                                  # exclude seen products
        n_pos = int((scores > 0).sum())
        take = min(k, n_pos)
        recs = []
        if take > 0:
            top = np.argpartition(-scores, take - 1)[:take]
            top = top[np.argsort(-scores[top], kind="stable")]
            recs = [(int(self.item_ids[j]), float(scores[j])) for j in top]
        if len(recs) < k:
            self.last_info["n_padded"] = k - len(recs)
            recs = fill_with_popular(recs, self.popular_items, seen_items, k)
        return recs

    # ---------------- explanation (dashboard addition) ----------------
    def explain(self, user_id, item_id, top_n=3):
        """History items that contributed most to `item_id`'s score: w_ui * sim(i, item).

        Returns a list of (history_itemid, user_weight, similarity, contribution).
        """
        if user_id not in self.user_index or item_id not in self.item_index:
            return []
        u_row = self.R[self.user_index[user_id]]
        col = self.S[:, self.item_index[item_id]].toarray().ravel()
        out = []
        for j, w in zip(u_row.indices, u_row.data):
            s = float(col[j])
            if s > 0:
                out.append((int(self.item_ids[j]), float(w), s, float(w) * s))
        out.sort(key=lambda x: -x[3])
        return out[:top_n]
