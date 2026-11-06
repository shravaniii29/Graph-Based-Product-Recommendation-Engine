"""Graph-Based Product Recommendation Engine: Streamlit dashboard (Phase 2).

Run from the project root:
    streamlit run dashboard/app.py

Everything shown comes from the saved Phase 1 outputs (outputs/) or is computed
live by the validated recommenders in src/. Nothing is hardcoded.
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import streamlit as st

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DASHBOARD_DIR)
for p in (PROJECT_ROOT, DASHBOARD_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import charts  # noqa: E402
import theme as T  # noqa: E402
from ui_helpers import build_user_options, default_index  # noqa: E402
from src import config as C  # noqa: E402
from src import data_loader as DL  # noqa: E402
from src.engine import RecommendationEngine, compare_with_saved  # noqa: E402
from src.evaluation import ci_overlap_table, recommendation_overlap  # noqa: E402
from src.graph_builder import degree_arrays  # noqa: E402
from src.scalability import growth_summary, long_format  # noqa: E402

PAGES = [
    "🏠  Home",
    "👤  User Recommendations",
    "📘  Baseline: Item-Item CF",
    "🕸️  Graph Recommendations",
    "📊  Algorithm Comparison",
    "🔗  Graph Visualization",
    "📈  Scalability Analysis",
]
METRICS = ["precision@5", "recall@5", "f1@5", "ndcg@5"]

st.set_page_config(page_title="Graph Recommendation Engine", page_icon="🛍️", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(T.CSS, unsafe_allow_html=True)


# ======================================================================
# Streamlit-version-safe helpers
# ======================================================================
def _st_version():
    nums = re.findall(r"\d+", st.__version__)
    return tuple(int(x) for x in nums[:2]) if len(nums) >= 2 else (0, 0)


_NEW_WIDTH_API = _st_version() >= (1, 50)


def show_chart(fig, key):
    """Full-width Plotly chart with our own theme (works on old and new Streamlit)."""
    if _NEW_WIDTH_API:
        try:
            st.plotly_chart(fig, width="stretch", theme=None, key=key)
            return
        except TypeError:
            pass
    st.plotly_chart(fig, use_container_width=True, theme=None, key=key)


def show_table(df, column_config=None, height=None):
    kwargs = {"hide_index": True}
    if column_config:
        kwargs["column_config"] = column_config
    if height:
        kwargs["height"] = height
    if _NEW_WIDTH_API:
        try:
            st.dataframe(df, width="stretch", **kwargs)
            return
        except TypeError:
            pass
    st.dataframe(df, use_container_width=True, **kwargs)


def cards(items):
    """Row of KPI cards. items: list of (label, value, sub)."""
    cols = st.columns(len(items))
    for i, (col, item) in enumerate(zip(cols, items)):
        label, value, sub = (list(item) + [""])[:3]
        col.markdown(T.kpi(label, value, sub, T.CARD_ACCENTS[i % len(T.CARD_ACCENTS)]), unsafe_allow_html=True)


def page_header(title, subtitle):
    st.markdown(f'<div class="page-title">{T.esc(title)}</div><div class="page-sub">{T.esc(subtitle)}</div>',
                unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section">{T.esc(title)}</div>', unsafe_allow_html=True)


def note(text_html):
    st.markdown(f'<div class="note">{text_html}</div>', unsafe_allow_html=True)


def show_issues(issues, keys):
    for i in issues:
        if i["key"] in keys:
            st.error(f"**{i['file']}**: {i['message']}. Re-copy this file from the Phase 1 `outputs.zip` "
                     f"into the `outputs/` folder.")


# ======================================================================
# Cached loading (runs once per session; later page switches are instant)
# ======================================================================
@st.cache_resource(show_spinner="Loading the saved graph and models…")
def get_engine():
    return RecommendationEngine()


@st.cache_data(show_spinner=False)
def get_results():
    return DL.load_results()


@st.cache_data(show_spinner=False)
def get_file_status():
    return DL.check_outputs()


@st.cache_data(show_spinner="Generating recommendations…", max_entries=200)
def get_recs(user_id, method):
    return get_engine().recommend(method, user_id)


@st.cache_data(show_spinner=False)
def get_degrees():
    return degree_arrays(get_engine().G)


@st.cache_data(show_spinner=False)
def get_user_sizes():
    e = get_engine()
    return e.interactions.groupby("visitorid").size().to_dict()


def engine_or_stop():
    try:
        return get_engine()
    except Exception as e:  # missing/corrupted core files
        st.error(f"The recommendation engine could not start: {e}")
        st.info("Check that the `outputs/` folder contains `data/train_interactions.csv.gz` "
                "(see README → Troubleshooting).")
        st.stop()


# ======================================================================
# Sidebar: navigation + user selection (rendered on every page so the
# selected user is kept when switching pages)
# ======================================================================
def candidate_users(res, engine):
    pue = res.get("per_user_evaluation")
    if pue is not None:
        users = sorted(int(u) for u in pue["visitorid"].unique() if engine.has_user(u))
        if users:
            return users, "evaluated"
    gt_users = sorted(u for u in engine.ground_truth if engine.has_user(u))
    if gt_users:
        return gt_users[:300], "with test interactions"
    return [int(u) for u in engine.train_users[:300]], "training"


def sidebar(res):
    st.sidebar.markdown("### 🛍️ Graph Recommender")
    st.sidebar.caption("Retailrocket · Scalable Data Science mini project")
    page = st.sidebar.radio("Navigate", PAGES, key="nav", label_visibility="collapsed")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Selected user**")
    user_id = None
    try:
        engine = get_engine()
    except Exception:
        engine = None
    if engine is None:
        st.sidebar.warning("User selection unavailable: engine failed to load.")
    else:
        users, kind = candidate_users(res, engine)
        sizes = get_user_sizes()
        demo = (res.get("dataset_overview") or {}).get("demo_user")
        mode = st.sidebar.radio("Pick from", ["Evaluated users", "Enter a user ID"], key="user_mode",
                                horizontal=True, label_visibility="collapsed")
        # The selected user is kept in st.session_state["selected_user"], so it survives page changes and
        # switching the picker mode. Widget parameters stay constant (no index/value that follows the
        # selection), because changing them would make Streamlit treat it as a new widget and drop a click.
        previous = st.session_state.get("selected_user")
        if mode == "Evaluated users":
            labels, label_to_id, id_to_label = build_user_options(users, sizes, engine.ground_truth)
            if st.session_state.get("user_select") not in label_to_id:      # first run or after a mode switch
                start = previous if previous in id_to_label else users[default_index(users, [demo])]
                st.session_state["user_select"] = id_to_label[int(start)]
            chosen = st.sidebar.selectbox(f"{len(users)} {kind} users", labels, key="user_select")
            user_id = label_to_id[chosen]
        else:
            if "user_id_input" not in st.session_state:                     # first run or after a mode switch
                st.session_state["user_id_input"] = int(previous if previous is not None
                                                        else (demo if demo else users[0]))
            user_id = int(st.sidebar.number_input("Visitor ID", min_value=0, step=1, key="user_id_input"))
            if not engine.has_user(user_id):
                st.sidebar.warning("Not in the training data: every method will fall back to popular "
                                   "products (cold start).")
        st.session_state["selected_user"] = user_id
        if engine.has_user(user_id):
            st.sidebar.caption(f"{sizes.get(user_id, 0)} products in history · "
                               f"{len(engine.ground_truth.get(user_id, ()))} new products in the test period")

    st.sidebar.markdown("---")
    with st.sidebar.expander("Data status"):
        status = get_file_status()
        ok = int(status["exists"].sum())
        st.write(f"Phase 1 files found: **{ok}/{len(status)}**")
        tr = res.get("test_results")
        if tr is not None:
            st.write(f"Phase 1 tests: **{(tr['status'] == 'PASS').sum()}/{len(tr)} passed**")
        if engine is not None:
            st.write(f"Graph: {engine.graph_source} · CF model: {engine.cf_source} · "
                     f"loaded in {engine.load_time_s:.1f}s")
        missing = status[~status["exists"]]["file"].tolist()
        if missing:
            st.warning("Missing: " + ", ".join(missing))
    return page, user_id


# ======================================================================
# Page 1: Home
# ======================================================================
def page_home(res):
    st.markdown('<div class="hero"><h1>Graph-Based Product Recommendation Engine</h1>'
                '<p>Comparing graph-based recommenders (Common Neighbors, Personalized PageRank) with traditional '
                'Item-Item Collaborative Filtering on the Retailrocket e-commerce dataset.</p></div>',
                unsafe_allow_html=True)
    ov = res.get("dataset_overview")
    if ov is None:
        show_issues(res["issues"], {"dataset_overview"})
        return
    raw, clean, train = ov.get("raw", {}), ov.get("cleaning", {}), ov.get("clean_train_summary", {})

    section("Raw dataset (events.csv)")
    cards([("Interactions", T.fmt_int(raw.get("total_interactions")), "view · add to cart · transaction"),
           ("Visitors", T.fmt_int(raw.get("unique_users")), "unique visitorid"),
           ("Products", T.fmt_int(raw.get("unique_products")), "unique itemid"),
           ("Time span", f"{raw.get('days_covered', '–')} days",
            f"{str(raw.get('first_event', ''))[:10]} → {str(raw.get('last_event', ''))[:10]}"),
           ("Duplicates removed", T.fmt_int(clean.get("removed_duplicates")), "exact repeated events")])

    section("Clean training data used by every method")
    sparsity = train.get("matrix_sparsity_pct")
    cards([("Users", T.fmt_int(train.get("train_users")), "≥ 5 products, ≤ 500 (bots removed)"),
           ("Products", T.fmt_int(train.get("train_products")), "≥ 5 distinct users"),
           ("User-product pairs", T.fmt_int(train.get("train_interactions")), "= graph edges"),
           ("Matrix sparsity", f"{sparsity:.2f}%" if sparsity is not None else "–", "of the user × product matrix"),
           ("Evaluable test users", T.fmt_int(train.get("test_users_with_new_relevant_items")),
            f"after cutoff {str(ov.get('cutoff_datetime', ''))[:10]}")])

    c1, c2 = st.columns([1.05, 1])
    with c1:
        section("Interaction types")
        ev = ov.get("event_distribution")
        if ev:
            show_chart(charts.event_distribution(ev), "home_events")
    with c2:
        section("From raw visitors to evaluable users")
        stages = [("Raw visitors", raw.get("unique_users")),
                  ("Users in clean training set", train.get("train_users")),
                  ("Test users seen in training", train.get("test_users_known_in_train")),
                  ("Users with new products to predict", train.get("test_users_with_new_relevant_items"))]
        stages = [(a, b) for a, b in stages if b is not None]
        if stages:
            show_chart(charts.data_funnel(stages), "home_funnel")

    section("Interaction weights")
    s = res.get("settings", {})
    w = s.get("EVENT_WEIGHTS", C.EVENT_WEIGHTS)
    weights = pd.DataFrame([
        {"Event": "view", "Weight": w.get("view"), "Why": "Weakest signal: browsing, often exploratory."},
        {"Event": "add to cart", "Weight": w.get("addtocart"), "Why": "Clear purchase intent; much rarer than views."},
        {"Event": "transaction", "Weight": w.get("transaction"), "Why": "Strongest signal: the user paid."},
    ])
    show_table(weights)
    st.caption(f"Repeated interactions with the same product are summed and capped at "
               f"{s.get('MAX_EDGE_WEIGHT', C.MAX_EDGE_WEIGHT)}.")

    section("Pipeline")
    steps = [("Clean", "drop duplicates, convert timestamps"), ("Weight", "view 1 · cart 3 · buy 5"),
             ("Split by time", f"first {int(100*s.get('TRAIN_FRACTION', 0.8))}% train · rest test"),
             ("Filter", "k-core: ≥5 products/user, ≥5 users/product"),
             ("Model", "CF matrix · bipartite graph"), ("Evaluate", "P@5 · R@5 · F1@5 · NDCG@5")]
    st.markdown('<div class="steps">' + "".join(
        f'<div class="step"><div class="n">{i}</div><b>{T.esc(a)}</b><span>{T.esc(b)}</span></div>'
        for i, (a, b) in enumerate(steps, 1)) + "</div>", unsafe_allow_html=True)

    img = os.path.join(C.VIS_DIR, "01_dataset_inspection.png")
    if os.path.exists(img):
        with st.expander("Phase 1 dataset inspection chart (events per day, user activity)"):
            st.image(img)


# ======================================================================
# Page 2: User Recommendations
# ======================================================================
def rec_card_html(method, df, info):
    items = []
    for r in df.itertuples():
        hit = ' <span class="hit">✓ interacted later</span>' if r.in_future_interactions else ""
        items.append(f"<li>item {r.itemid}{hit}</li>")
    meta = f"{info['time_ms']:.1f} ms"
    if info.get("fallback"):
        meta += " · popularity fallback"
    elif info.get("n_padded"):
        meta += f" · {info['n_padded']} topped up with popular items"
    return (f'<div class="rec-card" style="background:{T.METHOD_SOFT.get(method)}">'
            f'<div class="title">{T.esc(T.SHORT_NAMES.get(method, method))}</div>'
            f'<div class="meta">{T.esc(meta)}</div><ol>{"".join(items)}</ol></div>')


def saved_match(res, user_id, method, live_df):
    rec = res.get("recommendations")
    if rec is None:
        return None
    saved = rec[(rec["visitorid"] == user_id) & (rec["method"] == method)].sort_values("rank")
    if saved.empty:
        return None
    return compare_with_saved(live_df, saved)


def page_user(res, user_id):
    engine = engine_or_stop()
    page_header("User Recommendations", "Interaction history of the selected user and the top-5 from every method.")
    if user_id is None:
        st.info("Select a user in the sidebar.")
        return
    if not engine.has_user(user_id):
        st.warning(f"User {user_id} has no training history. All methods fall back to the most popular "
                   f"products. This is the cold-start case.")
    else:
        s = engine.user_summary(user_id)
        cards([("User", user_id, f"graph node U_{user_id}"),
               ("Products interacted", s["n_products"], f"{s['n_events']} raw events"),
               ("Purchased / carted", f"{s['n_purchased']} / {s['n_carted']}", "strongest action per product"),
               ("Active period", f"{s['first']:%d %b} → {s['last']:%d %b %Y}" if s["first"] is not None else "–",
                "training period"),
               ("Future products", s["n_future_items"], "new items after the cutoff (test)")])

    section("Top-5 recommendations from each method")
    if engine.ground_truth.get(user_id):
        st.caption("✓ marks a product this user really interacted with after the training cutoff.")
    else:
        st.caption("This user has no new products in the test period, so hits cannot be checked.")
    cols = st.columns(4)
    results = {}
    for col, m in zip(cols, C.ALL_METHODS):
        df, info = get_recs(user_id, m)
        results[m] = (df, info)
        col.markdown(rec_card_html(m, df, info), unsafe_allow_html=True)

    checks = {m: saved_match(res, user_id, m, results[m][0]) for m in C.ALL_METHODS}
    checked = {m: v for m, v in checks.items() if v is not None}
    if checked:
        bad = [T.SHORT_NAMES[m] for m, v in checked.items() if v == "different"]
        ties = [T.SHORT_NAMES[m] for m, v in checked.items() if v == "tie"]
        if not bad:
            msg = (f"Live results match the saved Phase 1 recommendations for this user "
                   f"({len(checked)}/{len(checked)} methods).")
            if ties:
                msg += (f" For {', '.join(ties)}, two products have exactly the same score, and the one picked "
                        f"for the last slot differs from Phase 1. Both choices are equally correct.")
            st.success(msg)
        else:
            st.warning(f"Live results differ from the saved Phase 1 output for: {', '.join(bad)}. "
                       f"This can happen if the graph was rebuilt instead of loaded (see Data status).")

    if engine.has_user(user_id):
        hist = engine.user_history(user_id)
        section("Interaction history (training period)")
        c1, c2 = st.columns([1.6, 1])
        with c1:
            show_chart(charts.history_bars(hist), "user_hist")
        with c2:
            show_chart(charts.event_mix(hist), "user_mix")
        show_table(hist.rename(columns={"itemid": "Product", "weight": "Weight", "raw_weight": "Raw weight",
                                        "n_events": "Events", "strongest_event": "Strongest action",
                                        "last_interaction": "Last interaction",
                                        "product_popularity": "Product popularity (users)"}),
                   height=min(420, 38 + 35 * len(hist)))


# ======================================================================
# Page 3: Baseline Item-Item CF
# ======================================================================
def rec_table(df, method):
    out = df.rename(columns={"rank": "Rank", "itemid": "Product", "score": "Score",
                             "product_popularity": "Popularity (users)",
                             "in_future_interactions": "Interacted later?"})
    mx = float(out["Score"].max()) if len(out) and out["Score"].max() > 0 else 1.0
    cfg = {"Score": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=mx,
                                                    format="%.5f" if mx < 0.01 else "%.3f"),
           "Interacted later?": st.column_config.CheckboxColumn("Interacted later?")}
    show_table(out, column_config=cfg)


def page_baseline(res, user_id):
    engine = engine_or_stop()
    page_header("Baseline: Item-Item Collaborative Filtering",
                "“People who interacted with this product also interacted with that one.”")
    note("<b>How it works.</b> Each product is a vector of user interactions (a column of the user × product "
         "matrix). Two products are similar when the <b>same users</b> interacted with both: "
         "<i>cosine similarity</i> = cos(angle between the vectors). A product's score for a user is "
         "Σ (user's weight on a history item × similarity to that item). Already-seen products are excluded "
         f"and the top {engine.settings['TOP_K']} are returned. For scalability only the "
         f"{engine.settings['CF_NEIGHBORS']} most similar products per item are kept.")
    S = engine.cf.S
    cards([("Products in matrix", T.fmt_int(S.shape[0]), "rows/columns of the similarity matrix"),
           ("Stored similarities", T.fmt_int(S.nnz), f"≤ {engine.settings['CF_NEIGHBORS']} neighbours per item"),
           ("Similarity matrix", f"{engine.cf.similarity_matrix_mb():.1f} MB", "sparse (CSR)"),
           ("Model source", "saved Phase 1" if engine.cf_source == "saved" else "recomputed", "item_similarity.npz")])

    if user_id is None:
        return
    df, info = get_recs(user_id, C.CF_NAME)
    section(f"Recommendations for user {user_id}")
    if info.get("fallback"):
        st.info(f"Fewer than {engine.settings['MIN_HISTORY']} products in this user's history (or unknown user), "
                f"so CF falls back to the most popular unseen products.")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        rec_table(df, C.CF_NAME)
        st.caption(f"Generated live in {info['time_ms']:.2f} ms. Previously interacted products are excluded.")
    with c2:
        show_chart(charts.rec_scores(df, C.CF_NAME), "cf_scores")

    if not info.get("fallback") and engine.has_user(user_id):
        section("Why these products? (strongest contributions)")
        ex = engine.cf_explanations(user_id, df["itemid"].tolist())
        if ex.empty:
            st.caption("No positive similarity contributions (list was filled with popular items).")
        else:
            show_table(ex.rename(columns={"recommended_item": "Recommended", "because_of_item": "Because you used",
                                          "user_weight": "Your weight", "cosine_similarity": "Cosine similarity",
                                          "contribution": "Contribution (weight × similarity)"}),
                       column_config={"Cosine similarity": st.column_config.NumberColumn(format="%.3f"),
                                      "Contribution (weight × similarity)": st.column_config.NumberColumn(format="%.3f")})


# ======================================================================
# Page 4: Graph recommendations
# ======================================================================
def page_graph_recs(res, user_id):
    engine = engine_or_stop()
    page_header("Graph-Based Recommendations",
                "Common Neighbors and Personalized PageRank on the weighted user–product bipartite graph.")
    if user_id is None:
        return
    tab_cn, tab_ppr, tab_cmp = st.tabs(["Common Neighbors", "Personalized PageRank", "Side by side"])

    with tab_cn:
        note("<b>Common Neighbors.</b> Two users are similar when they share products: CN(u, v) = number of "
             f"products both interacted with. We take the {engine.settings['CN_TOP_USERS']} most similar users "
             "and score each unseen product by Σ CN(u, v) × weight(v, product). "
             "<i>“Find people whose basket looks like yours and show what else they liked.”</i>")
        df, info = get_recs(user_id, C.CN_NAME)
        if info.get("fallback"):
            st.info("Insufficient history: popularity fallback.")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            rec_table(df, C.CN_NAME)
            st.caption(f"Generated live in {info['time_ms']:.1f} ms.")
        with c2:
            sim = engine.similar_users(user_id, n=10)
            if len(sim):
                show_chart(charts.similar_users_bar(sim), "cn_sim")
            else:
                st.caption("No users share a product with this user.")

    with tab_ppr:
        note(f"<b>Personalized PageRank.</b> A random walker moves along edges (heavier edges more likely) and, "
             f"with probability {1 - engine.settings['PPR_ALPHA']:.2f}, jumps back to the selected user. "
             "The share of time spent on each product is its score, so products reachable through "
             "<b>many paths of any length</b> rank high. Scores are probabilities (all nodes sum to 1).")
        df, info = get_recs(user_id, C.PPR_NAME)
        if info.get("fallback"):
            st.info("Insufficient history: popularity fallback.")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            rec_table(df, C.PPR_NAME)
            st.caption(f"Generated live in {info['time_ms']:.0f} ms. PageRank iterates over the whole graph "
                       f"({engine.G.number_of_nodes():,} nodes) for every user, which is why it is the slowest.")
        with c2:
            show_chart(charts.rec_scores(df, C.PPR_NAME), "ppr_scores")

    with tab_cmp:
        lists = {m: get_recs(user_id, m)[0] for m in C.PERSONALISED_METHODS}
        all_items = []
        for m in C.PERSONALISED_METHODS:
            for it in lists[m]["itemid"]:
                if it not in all_items:
                    all_items.append(int(it))
        gt = engine.ground_truth.get(user_id, set())
        rows = []
        for it in all_items:
            row = {"Product": it}
            for m in C.PERSONALISED_METHODS:
                ranks = lists[m].loc[lists[m]["itemid"] == it, "rank"]
                row[T.SHORT_NAMES[m]] = f"#{int(ranks.iloc[0])}" if len(ranks) else ""
            row["Recommended by"] = sum(1 for m in C.PERSONALISED_METHODS if row[T.SHORT_NAMES[m]])
            row["Interacted later?"] = it in gt
            rows.append(row)
        table = pd.DataFrame(rows).sort_values(["Recommended by", "Product"], ascending=[False, True])
        show_table(table, column_config={"Interacted later?": st.column_config.CheckboxColumn()})
        st.caption("Rank of each product in each method's top-5. Products recommended by several methods are "
                   "supported by different kinds of evidence.")


# ======================================================================
# Page 5: Algorithm comparison
# ======================================================================
def page_comparison(res):
    page_header("Algorithm Comparison", "Saved Phase 1 evaluation on a chronological train/test split. "
                                        "Same users and same split for every method.")
    summ, pue = res.get("evaluation_summary"), res.get("per_user_evaluation")
    if summ is None:
        show_issues(res["issues"], {"evaluation_summary"})
        return
    methods = [m for m in C.ALL_METHODS if m in set(summ["method"])]
    s = summ.set_index("method")
    personal = [m for m in C.PERSONALISED_METHODS if m in s.index]

    def best(metric, lowest=False):
        col = s.loc[personal, metric]
        m = col.idxmin() if lowest else col.idxmax()
        return T.SHORT_NAMES[m], col[m]

    n_users = int(s["n_users"].iloc[0])
    b_p, b_r, b_n, b_t = best("precision@5"), best("recall@5"), best("ndcg@5"), best("avg_time_ms", lowest=True)
    cards([("Users evaluated", n_users, "reproducible sample"),
           ("Best Precision@5", f"{b_p[1]:.4f}", b_p[0]), ("Best Recall@5", f"{b_r[1]:.4f}", b_r[0]),
           ("Best NDCG@5", f"{b_n[1]:.4f}", b_n[0]), ("Fastest", f"{b_t[1]:.2f} ms", b_t[0])])

    section("Recommendation quality @5")
    show_chart(charts.metric_bars(summ, METRICS, methods), "cmp_bars")
    st.caption("Error bars are 95% bootstrap confidence intervals. Absolute values are low because each user "
               "touches a handful of products out of ~12,000. The comparison between methods is what matters.")

    section("Is the difference real?")
    c1, c2 = st.columns([1, 1.1])
    with c1:
        metric = st.selectbox("Metric", METRICS, index=3, key="cmp_metric")
        show_chart(charts.metric_ci_dots(summ, metric, methods), "cmp_ci")
    with c2:
        st.write("")
        st.write("")
        if f"{metric}_ci_low" in summ.columns:
            show_table(ci_overlap_table(summ, metric, personal).rename(columns={
                "comparison": "Comparison", "higher": "Higher", "difference": "Difference", "verdict": "Verdict"}),
                column_config={"Difference": st.column_config.NumberColumn(format="%.4f")})
        st.caption("If two confidence intervals overlap, the sample is too small to say one method is better.")

    section("Execution time")
    c1, c2 = st.columns(2)
    with c1:
        show_chart(charts.time_bars(summ, methods), "cmp_time")
    with c2:
        if pue is not None:
            show_chart(charts.time_boxes(pue, methods), "cmp_time_box")

    section("Full comparison table")
    cols = ["method", "n_users"] + METRICS + ["hit@5", "avg_time_ms", "median_time_ms", "fallback_pct", "padded_pct"]
    tbl = summ[[c for c in cols if c in summ.columns]].copy()
    show_table(tbl.rename(columns={"method": "Method", "n_users": "Users", "hit@5": "Hit rate@5",
                                   "avg_time_ms": "Avg ms", "median_time_ms": "Median ms",
                                   "fallback_pct": "Fallback %", "padded_pct": "Padded %"}),
               column_config={c: st.column_config.NumberColumn(format="%.4f") for c in METRICS + ["Hit rate@5"]})

    c1, c2 = st.columns(2)
    rec = res.get("recommendations")
    with c1:
        if pue is not None:
            show_chart(charts.hit_distribution(pue, methods), "cmp_hits")
    with c2:
        if rec is not None:
            show_chart(charts.overlap_heatmap(recommendation_overlap(rec, methods)), "cmp_overlap")
    if res.get("interpretation"):
        section("Interpretation (generated from the Phase 1 numbers)")
        note(res["interpretation"].replace("**", "").replace("\n", "<br>"))


# ======================================================================
# Page 6: Graph visualization
# ======================================================================
def page_graph_viz(res, user_id):
    engine = engine_or_stop()
    page_header("Graph Visualization", "Interactive view of the user–product bipartite graph around the selected user.")
    gs = res.get("graph_stats") or {}
    G = engine.G
    u_deg, p_deg = get_degrees()
    cards([("User nodes", T.fmt_int(len(u_deg)), "U_<visitorid>"),
           ("Product nodes", T.fmt_int(len(p_deg)), "P_<itemid>"),
           ("Edges", T.fmt_int(gs.get("edges", G.number_of_edges())), "weighted interactions"),
           ("Bipartite density", f"{gs['bipartite_density']*100:.3f}%" if "bipartite_density" in gs else "–",
            "edges / (users × products)"),
           ("Largest component", f"{gs['largest_component_pct']:.1f}%" if "largest_component_pct" in gs else "–",
            f"{gs.get('connected_components', '–')} components")])

    if user_id is None or not engine.has_user(user_id):
        st.info("Select a user who is in the training data to draw their neighbourhood.")
    else:
        section(f"Neighbourhood of user {user_id}")
        c1, c2, c3 = st.columns(3)
        max_items = c1.slider("Products of the user", 3, 20, 8, key="gv_items")
        max_users = c2.slider("Other users per product", 1, 10, 4, key="gv_users")
        highlight = c3.selectbox("Highlight recommendations from",
                                 ["Personalized PageRank", "Common Neighbors", "Item-Item CF", "None"], key="gv_method")
        recs = [] if highlight == "None" else get_recs(user_id, highlight)[0]["itemid"].tolist()
        H, center, rec_nodes = engine.subgraph(user_id, max_items, max_users, recommended=recs)
        show_chart(charts.network(H, center, rec_nodes, G.degree), "gv_network")
        st.caption(f"{H.number_of_nodes()} nodes, {H.number_of_edges()} edges shown out of "
                   f"{G.number_of_nodes():,} / {G.number_of_edges():,}. Purple: selected user. Blue: other users. "
                   f"Orange squares: products. Green stars: recommended products. Edge width = interaction weight. "
                   f"Each recommended product is linked to a user in the picture who used it, so you can trace "
                   f"the path the graph took.")

    section("Degree distribution")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        show_chart(charts.degree_scatter(u_deg, p_deg), "gv_degree")
    with c2:
        note("A roughly straight line on log-log axes means a <b>heavy-tailed</b> distribution: a few hub "
             "products and users have very many connections, and most have few.<br><br>"
             f"Users: median degree <b>{np.median(u_deg):.0f}</b>, max <b>{u_deg.max()}</b>.<br>"
             f"Products: median degree <b>{np.median(p_deg):.0f}</b>, max <b>{p_deg.max()}</b>.<br><br>"
             "Hubs matter: one very popular product links a user to hundreds of others, which makes "
             "Common Neighbors slower and pushes PageRank scores towards popular items.")


# ======================================================================
# Page 7: Scalability
# ======================================================================
def page_scalability(res):
    page_header("Scalability Analysis", "Pipeline re-run on 10%, 30%, 60% and 100% of the active users "
                                        "(nested, reproducible samples). Saved Phase 1 measurements.")
    scal = res.get("scalability_results")
    if scal is None:
        show_issues(res["issues"], {"scalability_results"})
        return
    scal = scal.sort_values("fraction")
    g = growth_summary(scal).set_index("quantity") if len(scal) >= 2 else pd.DataFrame()

    def gr(q):
        return f"×{g.loc[q, 'growth']:.1f}" if q in g.index and pd.notna(g.loc[q, "growth"]) else "–"

    lo, hi = int(scal["dataset_pct"].iloc[0]), int(scal["dataset_pct"].iloc[-1])
    cards([("Edges", gr("Interactions (edges)"), f"{lo}% → {hi}% of users"),
           ("Graph build time", gr("Graph build time (s)"), f"{scal['graph_build_time_s'].iloc[-1]:.2f} s at {hi}%"),
           ("PPR time / user", gr(f"{C.PPR_NAME} time per user (ms)"),
            f"{scal[f'{C.PPR_NAME}|time_ms'].iloc[-1]:.0f} ms at {hi}%" if f"{C.PPR_NAME}|time_ms" in scal else ""),
           ("CN time / user", gr(f"{C.CN_NAME} time per user (ms)"),
            f"{scal[f'{C.CN_NAME}|time_ms'].iloc[-1]:.1f} ms at {hi}%" if f"{C.CN_NAME}|time_ms" in scal else ""),
           ("Graph memory", f"{scal['graph_memory_mb'].iloc[-1]:.0f} MB",
            f"≈ {1e6*scal['graph_memory_mb'].iloc[-1]/max(scal['graph_edges'].iloc[-1],1):.0f} bytes/edge at {hi}%")])

    c1, c2 = st.columns(2)
    with c1:
        show_chart(charts.scal_lines(long_format(scal, "time_ms"), "interactions", "time_ms",
                                     "interactions (edges)", "avg ms per user (log)",
                                     "Dataset size vs recommendation time", log_y=True), "sc_time")
    with c2:
        show_chart(charts.scal_pair(scal, "graph_nodes",
                                    [("graph_memory_mb", "NetworkX graph (RSS change)", "#E8A0A0", "solid"),
                                     ("cf_similarity_mb", "CF similarity matrix", T.METHOD_COLORS[C.CF_NAME], "dash")],
                                    "graph nodes", "MB", "Graph nodes vs memory"), "sc_mem")
    c1, c2 = st.columns(2)
    with c1:
        show_chart(charts.scal_lines(long_format(scal, "ndcg@5"), "dataset_pct", "ndcg@5", "% of active users",
                                     "NDCG@5", "Dataset size vs recommendation quality"), "sc_quality")
    with c2:
        show_chart(charts.scal_pair(scal, "interactions",
                                    [("graph_build_time_s", "Graph construction", T.METHOD_COLORS[C.PPR_NAME], "solid"),
                                     ("cf_build_time_s", "CF similarity build", T.METHOD_COLORS[C.CF_NAME], "dash")],
                                    "interactions (edges)", "seconds", "Build time vs dataset size"), "sc_build")

    notes = []
    if (scal["graph_memory_mb"] <= 0).any():
        pcts = scal.loc[scal["graph_memory_mb"] <= 0, "dataset_pct"].tolist()
        notes.append(f"Memory at {pcts}% shows 0 MB because Python reused memory it had already allocated, so "
                     f"psutil saw no increase. Memory figures are approximate.")
    if "n_eval_users" in scal and (scal["n_eval_users"] < scal["n_eval_users"].max()).any():
        small = scal.loc[scal["n_eval_users"] < scal["n_eval_users"].max(), ["dataset_pct", "n_eval_users"]]
        notes.append("Quality at small sizes is noisy: " + ", ".join(
            f"{int(r.dataset_pct)}% had only {int(r.n_eval_users)} users to evaluate" for r in small.itertuples())
            + ". Read the quality chart as a trend.")
    for n_ in notes:
        st.caption("ℹ️ " + n_)

    section("Measurements")
    cols = ["dataset_pct", "users", "products", "interactions", "graph_nodes", "graph_build_time_s",
            "cf_build_time_s", "graph_memory_mb", "cf_similarity_mb", "n_eval_users"] + \
           [f"{m}|time_ms" for m in C.PERSONALISED_METHODS] + [f"{m}|ndcg@5" for m in C.PERSONALISED_METHODS]
    show_table(scal[[c for c in cols if c in scal.columns]])
    if len(g):
        with st.expander("Growth from smallest to largest sample"):
            show_table(g.reset_index(), column_config={"growth": st.column_config.NumberColumn(format="×%.1f")})

    section("Limitations of NetworkX (in-memory, single machine)")
    note("• <b>In memory:</b> every node and edge is a Python object in dictionaries, so memory grows linearly "
         "with edges but with a large constant (compare the graph with the compact CF sparse matrix).<br>"
         "• <b>Single-threaded, pure Python:</b> graph building and Common Neighbors run on one CPU core.<br>"
         "• <b>PageRank per query = O(edges × iterations):</b> NetworkX runs power iteration over the whole graph "
         "for every user, so its time grows with the graph (steepest line above).<br>"
         "• <b>Not distributed:</b> this project runs on one machine. Larger data would need Spark GraphFrames, "
         "a graph database, approximate/push-based PPR or C-based graph libraries. These are future work, "
         "not part of this implementation.")


# ======================================================================
# Main
# ======================================================================
def main():
    res = get_results()
    page, user_id = sidebar(res)
    if page == PAGES[0]:
        page_home(res)
    elif page == PAGES[1]:
        page_user(res, user_id)
    elif page == PAGES[2]:
        page_baseline(res, user_id)
    elif page == PAGES[3]:
        page_graph_recs(res, user_id)
    elif page == PAGES[4]:
        page_comparison(res)
    elif page == PAGES[5]:
        page_graph_viz(res, user_id)
    elif page == PAGES[6]:
        page_scalability(res)


if __name__ == "__main__":
    main()
