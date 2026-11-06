"""Data cleaning, interaction weighting, chronological split and filtering.

Ported from Phase 1 notebook Section 4. Only needed to re-run the pipeline on
raw `events.csv`; the dashboard itself loads the saved, already-processed data.
"""
import numpy as np
import pandas as pd

from .config import (EVENT_WEIGHTS, MAX_EDGE_WEIGHT, MAX_USER_ITEMS, MIN_ITEM_USERS,
                     MIN_USER_ITEMS, REQUIRED_COLUMNS, TRAIN_FRACTION)


def detect_timestamp_unit(ts):
    """Retailrocket uses Unix milliseconds; values above 1e12 are ms, otherwise seconds."""
    return "ms" if pd.Series(ts).max() > 1e12 else "s"


def clean_events(events_raw, event_weights=EVENT_WEIGHTS):
    """Normalise event names, drop unknown/missing/duplicate rows, add datetime and weight.

    Returns (clean_events_df, cleaning_log).
    """
    raw = events_raw.copy()
    raw.columns = [c.strip().lower() for c in raw.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"events data is missing required columns {missing}")
    ts_unit = detect_timestamp_unit(raw["timestamp"])

    events = raw[REQUIRED_COLUMNS].copy()
    n0 = len(events)
    events["event"] = events["event"].astype(str).str.strip().str.lower()
    events = events[events["event"].isin(event_weights.keys())]
    events = events.dropna(subset=REQUIRED_COLUMNS)
    n_after_na = len(events)
    events = events.drop_duplicates(subset=REQUIRED_COLUMNS)
    n_after_dupes = len(events)

    events["visitorid"] = events["visitorid"].astype(np.int64)
    events["itemid"] = events["itemid"].astype(np.int64)
    events["timestamp"] = events["timestamp"].astype(np.int64)
    if ts_unit == "s":
        events["timestamp"] = events["timestamp"] * 1000
    events["datetime"] = pd.to_datetime(events["timestamp"], unit="ms")
    events["weight"] = events["event"].map(event_weights).astype(np.int32)
    events["event"] = events["event"].astype("category")
    events = events.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    log = {"raw_rows": n0, "removed_unknown_or_missing": n0 - n_after_na,
           "removed_duplicates": n_after_na - n_after_dupes, "clean_rows": len(events)}
    return events, log


def chronological_split(events, train_fraction=TRAIN_FRACTION):
    """Earliest `train_fraction` of events (by time) = train, the rest = test."""
    cutoff_ts = int(events["timestamp"].quantile(train_fraction))
    train = events[events["timestamp"] < cutoff_ts].copy()
    test = events[events["timestamp"] >= cutoff_ts].copy()
    return train, test, cutoff_ts


def aggregate_interactions(ev, max_weight=MAX_EDGE_WEIGHT):
    """Collapse raw events into one row per (visitorid, itemid) with summed and capped weight."""
    agg = (ev.groupby(["visitorid", "itemid"], observed=True)
             .agg(raw_weight=("weight", "sum"),
                  n_events=("weight", "size"),
                  max_event_weight=("weight", "max"),
                  last_timestamp=("timestamp", "max"))
             .reset_index())
    agg["weight"] = agg["raw_weight"].clip(upper=max_weight).astype(np.float32)
    return agg


def filter_interactions(inter, min_user_items=MIN_USER_ITEMS, min_item_users=MIN_ITEM_USERS,
                        max_user_items=MAX_USER_ITEMS, max_rounds=50, verbose=True):
    """Remove bot-like users, then iteratively enforce minimum user/item activity (k-core style)."""
    log = []
    user_deg = inter.groupby("visitorid").size()
    bots = user_deg[user_deg > max_user_items].index
    inter = inter[~inter["visitorid"].isin(bots)]
    log.append(f"removed {len(bots):,} bot-like users (> {max_user_items} products)")
    for r in range(1, max_rounds + 1):
        n_before = len(inter)
        item_deg = inter.groupby("itemid")["visitorid"].nunique()
        inter = inter[inter["itemid"].isin(item_deg[item_deg >= min_item_users].index)]
        user_deg = inter.groupby("visitorid")["itemid"].nunique()
        inter = inter[inter["visitorid"].isin(user_deg[user_deg >= min_user_items].index)]
        if len(inter) == n_before:
            log.append(f"converged after {r} rounds")
            break
    else:
        log.append(f"stopped after {max_rounds} rounds (not fully converged)")
    if verbose:
        for line in log:
            print("  -", line)
    return inter.reset_index(drop=True)


def build_ground_truth(test_ev, train_inter):
    """Relevant items per user = test items that are in the train catalogue and new to that user."""
    train_users = set(train_inter["visitorid"].unique())
    train_items = set(train_inter["itemid"].unique())
    seen = train_inter.groupby("visitorid")["itemid"].agg(set)

    t = test_ev[test_ev["visitorid"].isin(train_users) & test_ev["itemid"].isin(train_items)]
    gt = {}
    for u, items in t.groupby("visitorid")["itemid"]:
        rel = set(items.unique()) - seen.get(u, set())
        if rel:
            gt[u] = rel
    stats = {
        "test_users_total": int(test_ev["visitorid"].nunique()),
        "test_users_known_in_train": int(test_ev.loc[test_ev["visitorid"].isin(train_users), "visitorid"].nunique()),
        "test_users_with_new_relevant_items": len(gt),
        "avg_relevant_items_per_user": float(np.mean([len(v) for v in gt.values()])) if gt else 0.0,
    }
    return gt, stats


def ground_truth_from_frame(df):
    """Saved test_ground_truth.csv (visitorid, itemid rows) -> {visitorid: set(itemids)}."""
    return {int(u): set(int(i) for i in g) for u, g in df.groupby("visitorid")["itemid"]}
