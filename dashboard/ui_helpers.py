"""Pure-Python helpers for the user picker (no Streamlit import, so they are unit-testable).

Why labels instead of format_func: the selectbox options ARE the readable labels
("29802 · 14 products · 8 future") and a dict maps each label back to its user ID.
With `format_func`, the widget value (int) and what is displayed (label) differ, which
made the selection hard to drive and test reliably (Streamlit's AppTest exposes the
displayed labels as `options`). With label options, the value and the display are the
same string, and the user ID is looked up explicitly.
"""


def user_label(user_id, n_products, n_future):
    return f"{int(user_id)}  ·  {int(n_products)} products  ·  {int(n_future)} future"


def build_user_options(users, sizes, ground_truth):
    """Return (labels, label_to_id, id_to_label) for the given user IDs (order preserved)."""
    labels, label_to_id, id_to_label = [], {}, {}
    for u in users:
        u = int(u)
        lab = user_label(u, sizes.get(u, 0), len(ground_truth.get(u, ())))
        labels.append(lab)
        label_to_id[lab] = u
        id_to_label[u] = lab
    return labels, label_to_id, id_to_label


def default_index(users, preferred):
    """Index of the first preferred user ID present in `users` (0 if none)."""
    users = [int(u) for u in users]
    for p in preferred:
        if p is not None and int(p) in users:
            return users.index(int(p))
    return 0
