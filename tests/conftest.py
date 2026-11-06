import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import config as C  # noqa: E402

OUTPUTS_PRESENT = os.path.exists(os.path.join(C.OUTPUTS_DIR, "data", "train_interactions.csv.gz"))
needs_outputs = pytest.mark.skipif(not OUTPUTS_PRESENT, reason="Phase 1 outputs/ folder not found")


@pytest.fixture(scope="session")
def engine():
    if not OUTPUTS_PRESENT:
        pytest.skip("Phase 1 outputs/ folder not found")
    from src.engine import RecommendationEngine
    return RecommendationEngine()


@pytest.fixture(scope="session")
def results():
    if not OUTPUTS_PRESENT:
        pytest.skip("Phase 1 outputs/ folder not found")
    from src.data_loader import load_results
    return load_results()


@pytest.fixture
def toy_interactions():
    """The 4-user toy example from the Phase 1 notebook (Section 5)."""
    return pd.DataFrame([
        (1, 101, 5), (1, 102, 3),
        (2, 101, 1), (2, 102, 1), (2, 103, 3),
        (3, 104, 5), (3, 105, 3),
        (4, 101, 3), (4, 103, 1), (4, 105, 1),
    ], columns=["visitorid", "itemid", "weight"]).astype({"weight": "float32"})


@pytest.fixture
def raw_events():
    """Small raw events table in the Retailrocket format."""
    rows = []
    t = 1_433_000_000_000
    for u in range(1, 13):
        for k, item in enumerate(range(100, 108)):
            if (u + item) % 3 == 0:
                continue
            ev = "transaction" if (u * item) % 11 == 0 else ("addtocart" if (u + item) % 5 == 0 else "view")
            rows.append((t + u * 1000 + k * 10_000_000, u, ev, item, None))
    rows.append(rows[0])                       # an exact duplicate
    rows.append((t, 99, "hover", 100, None))   # an unknown event type
    return pd.DataFrame(rows, columns=["timestamp", "visitorid", "event", "itemid", "transactionid"])
