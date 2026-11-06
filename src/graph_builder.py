"""Weighted bipartite user-product graph: building, statistics, loading, subgraphs.

Ported from Phase 1 notebook Section 6. `load_graph` adds a safe loader: if the
saved pickle is missing or cannot be read (for example a different NetworkX or
NumPy version), the graph is rebuilt from the saved training interactions with
exactly the same builder (about one second).
"""
import pickle

import networkx as nx
import numpy as np
import pandas as pd


def user_node(uid):
    return f"U_{uid}"


def product_node(iid):
    return f"P_{iid}"


def node_id(node):
    return int(node[2:])


def build_bipartite_graph(interactions):
    """Weighted, undirected bipartite user-product graph."""
    G = nx.Graph()
    u_nodes = ("U_" + interactions["visitorid"].astype(str)).to_numpy()
    p_nodes = ("P_" + interactions["itemid"].astype(str)).to_numpy()
    G.add_nodes_from(pd.unique(u_nodes), bipartite=0, node_type="user")
    G.add_nodes_from(pd.unique(p_nodes), bipartite=1, node_type="product")
    G.add_weighted_edges_from(zip(u_nodes, p_nodes, interactions["weight"].astype(float).to_numpy()))
    return G


def graph_statistics(G):
    users = [n for n, d in G.nodes(data=True) if d["node_type"] == "user"]
    prods = [n for n, d in G.nodes(data=True) if d["node_type"] == "product"]
    u_deg = np.array([G.degree(n) for n in users])
    p_deg = np.array([G.degree(n) for n in prods])
    comps = list(nx.connected_components(G))
    largest = max(len(c) for c in comps) if comps else 0
    w = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    return {
        "user_nodes": len(users),
        "product_nodes": len(prods),
        "total_nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "bipartite_density": G.number_of_edges() / (len(users) * len(prods)) if users and prods else 0.0,
        "networkx_density": nx.density(G),
        "avg_user_degree": float(u_deg.mean()), "median_user_degree": float(np.median(u_deg)), "max_user_degree": int(u_deg.max()),
        "avg_product_degree": float(p_deg.mean()), "median_product_degree": float(np.median(p_deg)), "max_product_degree": int(p_deg.max()),
        "connected_components": len(comps),
        "largest_component_nodes": largest,
        "largest_component_pct": 100 * largest / G.number_of_nodes(),
        "avg_edge_weight": float(w.mean()),
        "is_bipartite": nx.is_bipartite(G),
    }, u_deg, p_deg


def degree_arrays(G):
    """User and product degree arrays (cheap; used by the dashboard instead of full statistics)."""
    u_deg, p_deg = [], []
    for n, d in G.nodes(data=True):
        (u_deg if d.get("node_type") == "user" else p_deg).append(G.degree(n))
    return np.array(u_deg), np.array(p_deg)


def extract_subgraph(G, center, max_items=8, max_users_per_item=4):
    """Center user + their strongest products + top co-users of each product."""
    items = sorted(G[center], key=lambda p: -G[center][p]["weight"])[:max_items]
    nodes = {center, *items}
    for p in items:
        others = sorted((u for u in G[p] if u != center), key=lambda u: -G[p][u]["weight"])[:max_users_per_item]
        nodes.update(others)
    return G.subgraph(nodes).copy()


def validate_graph(G, interactions=None):
    """Sanity checks on a loaded graph. Returns a list of problems (empty = OK)."""
    problems = []
    if not isinstance(G, nx.Graph):
        return [f"object is a {type(G).__name__}, not a NetworkX graph"]
    if G.number_of_nodes() == 0:
        problems.append("graph has no nodes")
    types = {d.get("node_type") for _, d in G.nodes(data=True)}
    if not types <= {"user", "product"}:
        problems.append(f"unexpected node types {types}")
    if interactions is not None:
        if G.number_of_edges() != len(interactions):
            problems.append(f"{G.number_of_edges():,} edges but {len(interactions):,} training interactions")
        n_users = interactions["visitorid"].nunique()
        n_user_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "user")
        if n_user_nodes != n_users:
            problems.append(f"{n_user_nodes:,} user nodes but {n_users:,} training users")
    return problems


def load_graph(path, interactions=None):
    """Load the saved graph; rebuild it from `interactions` if loading or validation fails.

    Returns (G, source, message) where source is "saved" or "rebuilt".
    """
    try:
        with open(path, "rb") as f:
            G = pickle.load(f)
        problems = validate_graph(G, interactions)
        if not problems:
            return G, "saved", "Loaded the saved Phase 1 graph."
        reason = "; ".join(problems)
    except Exception as e:  # missing file, corrupted pickle, version mismatch...
        reason = f"{type(e).__name__}: {e}"
    if interactions is None:
        raise RuntimeError(f"Could not load graph ({reason}) and no interactions were given to rebuild it.")
    G = build_bipartite_graph(interactions)
    return G, "rebuilt", f"Saved graph unusable ({reason}). Rebuilt it from train_interactions.csv.gz."
