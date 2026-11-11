"""Dashboard tests with Streamlit's built-in AppTest (runs the real app headlessly).

Requires `streamlit` to be installed. Run from the project root:
    python -m pytest tests/test_dashboard_app.py -v
"""
import os
import re

import pytest

from src import config as C
from tests.conftest import needs_outputs

st_testing = pytest.importorskip("streamlit.testing.v1", reason="streamlit not installed")
pytest.importorskip("plotly", reason="plotly not installed")
pytestmark = needs_outputs

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard", "app.py")


def _app():
    at = st_testing.AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_app_starts_on_home():
    at = _app()
    assert any("Graph-Based Product Recommendation Engine" in m.value for m in at.markdown)


@pytest.mark.parametrize("page_index", range(7))
def test_every_page_renders(page_index):
    at = _app()
    nav = at.radio(key="nav")
    nav.set_value(nav.options[page_index]).run()
    assert not at.exception, [e.value for e in at.exception]


def _card_items(at):
    """Product IDs listed in the four recommendation cards, in method order."""
    cards = [m.value for m in at.markdown if 'class="rec-card"' in m.value]
    return [[int(x) for x in re.findall(r"<li>item (\d+)", c)] for c in cards]


def _shows_user(at, user_id):
    return any(f'<div class="value">{user_id}</div>' in m.value for m in at.markdown)


def _expected_items(engine, user_id):
    return [engine.recommend(m, user_id)[0]["itemid"].astype(int).tolist() for m in C.ALL_METHODS]


def test_user_selection_changes_recommendations(engine):
    """Regression test: selecting a user by its displayed option must work and update every method."""
    at = _app()
    nav = at.radio(key="nav")
    nav.set_value(nav.options[1]).run()

    sel = at.selectbox(key="user_select")
    assert len(sel.options) == 300
    first_label = sel.value
    first_id = int(first_label.split()[0])
    assert _card_items(at) == _expected_items(engine, first_id)

    # pick another evaluated user whose recommendations differ
    target_label = next(lab for lab in sel.options
                        if lab != first_label and _expected_items(engine, int(lab.split()[0]))
                        != _expected_items(engine, first_id))
    target_id = int(target_label.split()[0])
    sel.set_value(target_label).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="user_select").value == target_label
    assert _shows_user(at, target_id)
    assert _card_items(at) == _expected_items(engine, target_id)          # recommendations really changed
    assert _card_items(at) != _expected_items(engine, first_id)

    # the selection is kept when switching pages
    at.radio(key="nav").set_value(nav.options[2]).run()
    assert not at.exception
    assert at.selectbox(key="user_select").value == target_label
    assert any(f"Recommendations for user {target_id}" in m.value for m in at.markdown)


def test_user_id_input_and_cold_start(engine):
    at = _app()
    at.radio(key="nav").set_value(at.radio(key="nav").options[1]).run()
    selected = int(at.selectbox(key="user_select").value.split()[0])

    at.radio(key="user_mode").set_value("Enter a user ID").run()
    assert not at.exception
    assert at.number_input(key="user_id_input").value == selected      # current user carried over

    typed = int(engine.train_users[3])                                   # any training user, not only evaluated ones
    at.number_input(key="user_id_input").set_value(typed).run()
    assert not at.exception
    assert _shows_user(at, typed)
    assert _card_items(at) == _expected_items(engine, typed)

    at.number_input(key="user_id_input").set_value(999_999_999).run()   # unknown user -> cold start
    assert not at.exception
    assert any("cold start" in w.value for w in at.warning)
    assert all(len(items) == C.TOP_K for items in _card_items(at))

    at.radio(key="user_mode").set_value("Evaluated users").run()         # back to the list: valid selection
    assert not at.exception
    assert at.selectbox(key="user_select").value in at.selectbox(key="user_select").options
