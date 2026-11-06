"""Project paths and algorithm settings.

Defaults are the values used in the validated Phase 1 run. When the saved
`outputs/results/config.json` exists, `load_run_config()` reads the exact
settings that produced the saved results so the dashboard always matches them.
"""
import json
import os

# ---------------------------------------------------------------
# Paths (everything is relative to the project root, so the project
# works wherever the SDS_MiniProj folder is placed)
# ---------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.environ.get("SDS_OUTPUTS_DIR", os.path.join(PROJECT_ROOT, "outputs"))
DATA_DIR = os.path.join(OUTPUTS_DIR, "data")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
VIS_DIR = os.path.join(OUTPUTS_DIR, "visualizations")

# ---------------------------------------------------------------
# Algorithm settings (same values as the Phase 1 notebook)
# ---------------------------------------------------------------
SEED = 42
EVENT_WEIGHTS = {"view": 1, "addtocart": 3, "transaction": 5}
MAX_EDGE_WEIGHT = 15
TRAIN_FRACTION = 0.80
MIN_USER_ITEMS = 5
MIN_ITEM_USERS = 5
MAX_USER_ITEMS = 500
TOP_K = 5
MIN_HISTORY = 2
CF_NEIGHBORS = 50
CN_TOP_USERS = 50
PPR_ALPHA = 0.85
PPR_MAX_ITER = 100
PPR_TOL = 1e-6
N_BOOTSTRAP = 1000
SCALE_FRACTIONS = [0.10, 0.30, 0.60, 1.00]
N_SCALE_EVAL_USERS = 200
REQUIRED_COLUMNS = ["timestamp", "visitorid", "event", "itemid"]

# Method names exactly as they appear in the saved results files
CF_NAME = "Item-Item CF"
CN_NAME = "Common Neighbors"
PPR_NAME = "Personalized PageRank"
POP_NAME = "Popularity (reference)"
PERSONALISED_METHODS = [CF_NAME, CN_NAME, PPR_NAME]
ALL_METHODS = PERSONALISED_METHODS + [POP_NAME]

# Keys of config.json that are algorithm settings (others are run-time variables)
_SETTING_KEYS = [
    "SEED", "EVENT_WEIGHTS", "MAX_EDGE_WEIGHT", "TRAIN_FRACTION", "MIN_USER_ITEMS",
    "MIN_ITEM_USERS", "MAX_USER_ITEMS", "TOP_K", "MIN_HISTORY", "CF_NEIGHBORS",
    "CN_TOP_USERS", "PPR_ALPHA", "PPR_MAX_ITER", "PPR_TOL", "N_EVAL_USERS",
    "N_BOOTSTRAP", "SCALE_FRACTIONS", "N_SCALE_EVAL_USERS", "CUTOFF_TS", "DEMO_USER",
]


def default_settings():
    g = globals()
    return {k: g[k] for k in _SETTING_KEYS if k in g}


def load_run_config(path=None):
    """Settings used for the saved Phase 1 run (falls back to the defaults above)."""
    settings = default_settings()
    path = path or os.path.join(RESULTS_DIR, "config.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        settings.update({k: saved[k] for k in _SETTING_KEYS if k in saved})
    except (OSError, json.JSONDecodeError):
        pass
    return settings
