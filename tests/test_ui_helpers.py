"""Regression tests for the sidebar user picker (no Streamlit needed).

Bug fixed: the selectbox used raw integer IDs as options plus `format_func` for the
readable label, so the displayed option ("29802 · 14 products · 8 future") was not
a valid widget value. The options are now the labels themselves, mapped back to IDs.
"""
from dashboard.ui_helpers import build_user_options, default_index, user_label
from tests.conftest import needs_outputs


def test_label_format():
    assert user_label(29802, 14, 8) == "29802  ·  14 products  ·  8 future"


def test_options_round_trip():
    users = [29802, 14735, 32427]
    sizes = {29802: 14, 14735: 6}
    gt = {29802: {1, 2, 3}, 32427: {9}}
    labels, label_to_id, id_to_label = build_user_options(users, sizes, gt)
    assert len(labels) == len(set(labels)) == 3
    assert [label_to_id[lab] for lab in labels] == users          # every displayed option maps to its ID
    assert all(id_to_label[u] == lab for u, lab in zip(users, labels))
    assert labels[1] == "14735  ·  6 products  ·  0 future"       # missing counts shown as 0
    # what a test (or the browser) sends back is a displayed option -> valid value
    assert label_to_id[labels[0]] == 29802


def test_default_index():
    users = [5, 7, 9]
    assert default_index(users, [9]) == 2
    assert default_index(users, [None, 7]) == 1
    assert default_index(users, [42, None]) == 0


@needs_outputs
def test_options_for_real_evaluated_users(engine, results):
    users = sorted(int(u) for u in results["per_user_evaluation"]["visitorid"].unique())
    sizes = engine.interactions.groupby("visitorid").size().to_dict()
    labels, label_to_id, _ = build_user_options(users, sizes, engine.ground_truth)
    assert len(labels) == len(set(labels)) == len(users) == 300
    assert sorted(label_to_id.values()) == users
    lab = next(lab for lab in labels if lab.startswith("29802 "))
    assert lab == f"29802  ·  {sizes[29802]} products  ·  {len(engine.ground_truth[29802])} future"
