import pandas as pd
import pytest

from src.config import EVENT_WEIGHTS, MAX_EDGE_WEIGHT
from src.data_preprocessing import (aggregate_interactions, build_ground_truth, chronological_split,
                                    clean_events, filter_interactions, ground_truth_from_frame)


def test_clean_events_removes_bad_rows(raw_events):
    ev, log = clean_events(raw_events)
    assert log["removed_duplicates"] == 1
    assert log["removed_unknown_or_missing"] == 1
    assert set(ev["event"].astype(str)) <= set(EVENT_WEIGHTS)
    assert ev["timestamp"].is_monotonic_increasing
    for name, w in EVENT_WEIGHTS.items():
        assert (ev.loc[ev["event"] == name, "weight"] == w).all()


def test_clean_events_missing_column():
    with pytest.raises(ValueError):
        clean_events(pd.DataFrame({"timestamp": [1], "visitorid": [1], "itemid": [1]}))


def test_chronological_split_no_overlap(raw_events):
    ev, _ = clean_events(raw_events)
    train, test, cutoff = chronological_split(ev, 0.8)
    assert train["timestamp"].max() < cutoff <= test["timestamp"].min()
    assert len(train) + len(test) == len(ev)


def test_aggregation_sums_and_caps():
    ev = pd.DataFrame({"visitorid": [1] * 6 + [2], "itemid": [5] * 6 + [5],
                       "weight": [5, 5, 5, 3, 1, 1, 1], "timestamp": range(7)})
    agg = aggregate_interactions(ev)
    u1 = agg[agg["visitorid"] == 1].iloc[0]
    assert u1["raw_weight"] == 20 and u1["weight"] == MAX_EDGE_WEIGHT and u1["n_events"] == 6
    assert u1["max_event_weight"] == 5
    assert agg[agg["visitorid"] == 2].iloc[0]["weight"] == 1


def test_filter_thresholds(raw_events):
    ev, _ = clean_events(raw_events)
    agg = aggregate_interactions(ev)
    f = filter_interactions(agg, min_user_items=3, min_item_users=3, max_user_items=6, verbose=False)
    assert f.groupby("visitorid")["itemid"].nunique().between(3, 6).all()
    assert (f.groupby("itemid")["visitorid"].nunique() >= 3).all()


def test_ground_truth_only_new_known_items():
    train = pd.DataFrame({"visitorid": [1, 1, 2], "itemid": [10, 11, 10], "weight": [1.0, 1.0, 1.0]})
    test = pd.DataFrame({"visitorid": [1, 1, 1, 2, 3], "itemid": [10, 12, 11, 11, 10]})
    # 12 is unknown to train, 10/11 already seen by user 1, user 3 unknown
    test.loc[len(test)] = [1, 10]
    gt, stats = build_ground_truth(test, train)
    assert gt == {2: {11}}
    assert stats["test_users_with_new_relevant_items"] == 1


def test_ground_truth_from_frame():
    df = pd.DataFrame({"visitorid": [1, 1, 2], "itemid": [5, 6, 7]})
    assert ground_truth_from_frame(df) == {1: {5, 6}, 2: {7}}
