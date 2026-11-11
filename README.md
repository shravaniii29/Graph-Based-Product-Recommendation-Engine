# Graph-Based Product Recommendation Engine
### Comparing Graph-Based and Traditional Recommendation Methods
Scalable Data Science mini project · Retailrocket e-commerce dataset

| Method | Type | Idea |
|---|---|---|
| Item-Item Collaborative Filtering | traditional baseline | cosine similarity between products in the user × product matrix |
| Common Neighbors | graph | users who share products are similar; recommend what they liked |
| Personalized PageRank | graph | random walk with restart from the user over the bipartite graph |

**Phase 1** (Google Colab, `notebooks/Graph_Recommendation_Phase1.ipynb`) cleaned the data, built the models, evaluated them and ran the scalability experiments: 17/17 tests passed on the real data.
**Phase 2** (this folder) turns that notebook into Python modules and a Streamlit dashboard that uses the **saved Phase 1 results**.

> **Licences in brief:** the original project code is released under the **MIT licence** ([`LICENSE`](LICENSE)). Everything in `outputs/` is derived from the Retailrocket dataset and stays under **CC BY-NC-SA 4.0** ([`DATA_LICENSE.md`](DATA_LICENSE.md)); the MIT licence does not apply to those files. See section 10.

---

## 1. Project structure

```
SDS_MiniProj/
├── outputs/                      ← Phase 1 outputs (from outputs.zip). Read-only: never modified.
│   ├── data/                       train_interactions.csv.gz, test_ground_truth.csv, item_popularity.csv, ...
│   ├── models/                     bipartite_graph.pkl, item_similarity.npz, cf_index.json
│   ├── results/                    evaluation_summary.csv, per_user_evaluation.csv, recommendations.csv,
│   │                               scalability_results.csv, graph_stats.json, dataset_overview.json, ...
│   └── visualizations/             Phase 1 PNG + HTML charts
├── src/                          ← backend (algorithms ported unchanged from the notebook)
│   ├── config.py                   paths + settings (reads outputs/results/config.json)
│   ├── data_preprocessing.py       cleaning, weights, chronological split, filtering, ground truth
│   ├── baseline_recommender.py     Item-Item CF + popularity reference
│   ├── graph_builder.py            bipartite graph build / stats / safe loading / subgraphs
│   ├── graph_recommender.py        Common Neighbors + Personalized PageRank
│   ├── evaluation.py               Precision/Recall/F1/NDCG@5, bootstrap CIs, comparison helpers
│   ├── scalability.py              scalability experiment + helpers for the saved results
│   ├── data_loader.py              loads saved files and reports missing/corrupted ones
│   └── engine.py                   RecommendationEngine: the single object the dashboard uses
├── dashboard/
│   ├── app.py                      Streamlit app (7 pages)
│   ├── charts.py                   Plotly figures
│   ├── ui_helpers.py               user-picker labels ↔ user IDs (testable without Streamlit)
│   └── theme.py                    pastel palette, CSS, metric cards
├── tests/                        ← pytest suite (backend + dashboard)
├── notebooks/                    ← Phase 1 notebook (reference)
├── .streamlit/config.toml        ← dashboard theme
├── requirements.txt
├── LICENSE                       ← MIT licence (original project code only)
├── DATA_LICENSE.md               ← CC BY-NC-SA 4.0 notice for the dataset-derived files in outputs/
├── .gitignore
└── README.md
```

Phase 1 charts stay in `outputs/visualizations/`. The dashboard draws its own interactive versions from the saved CSVs.

---

## 2. Setup on Windows (one time)

Open **PowerShell** (or the VS Code terminal: *Terminal → New Terminal*) and run:

```powershell
# 1. Go to the project folder (replace the path with where you cloned/extracted it;
#    keep the quotes if the path contains spaces)
cd "path\to\SDS_MiniProj"

# 2. Check Python (3.10 or newer; tested on 3.13 and 3.14, see "Python compatibility" below)
py --version

# 3. Create a virtual environment in the folder .venv
py -m venv .venv

# 4. Activate it (the prompt then starts with (.venv))
.\.venv\Scripts\Activate.ps1

# 5. Install the dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Python compatibility.** Use **Python 3.10 or newer**. The full test suite has been run on
**Python 3.14 (Windows, including the Streamlit dashboard tests)** and **Python 3.13 (Linux, backend tests)**.
Python 3.10–3.12 should work with the versions in `requirements.txt` but have not been tested; older versions are
not supported.

If step 4 says *"running scripts is disabled on this system"*, run this once and then repeat step 4:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
(Or use Command Prompt instead: `.venv\Scripts\activate.bat`.)

**VS Code:** *File → Open Folder → SDS_MiniProj*, then *Ctrl+Shift+P → Python: Select Interpreter →* the one in `.venv`.

---

## 3. Run the tests

```powershell
python -m pytest tests -v
```
Expected: **52 passed** (about 2–3 minutes). This includes `tests/test_dashboard_app.py`, which starts the real Streamlit app headlessly and opens every page.

Optional full check (re-computes PageRank for all 300 evaluated users, ~3 minutes):
```powershell
$env:SDS_FULL_TEST=1; python -m pytest tests/test_saved_outputs.py -v; Remove-Item Env:SDS_FULL_TEST
```

## 4. Launch the dashboard

```powershell
streamlit run dashboard/app.py
```
The browser opens at **http://localhost:8501**. Stop it with **Ctrl+C** in the terminal.
The first page load takes a few seconds while the saved graph and similarity matrix are loaded; after that pages switch instantly.

Next time you only need:
```powershell
cd "path\to\SDS_MiniProj"
.\.venv\Scripts\Activate.ps1
streamlit run dashboard/app.py
```

---

## 5. Dashboard pages

| Page | Shows | Data source |
|---|---|---|
| Home | dataset statistics, interaction types, filtering funnel, weights, pipeline | `dataset_overview.json`, `config.json` |
| User Recommendations | selected user's history + top-5 from all methods, ✓ for products the user really interacted with later, check against saved Phase 1 output | live engine + `recommendations.csv` |
| Baseline: Item-Item CF | CF top-5, scores, and *why* (history item × cosine similarity) | live, saved `item_similarity.npz` |
| Graph Recommendations | Common Neighbors (+ most similar users), Personalized PageRank, side-by-side | live, saved `bipartite_graph.pkl` |
| Algorithm Comparison | Precision/Recall/F1/NDCG@5 with 95% CIs, significance check, execution time, overlap between methods | `evaluation_summary.csv`, `per_user_evaluation.csv`, `recommendations.csv` |
| Graph Visualization | interactive 2-hop subgraph with recommended products highlighted, degree distribution | live graph, `graph_stats.json` |
| Scalability Analysis | time / memory / quality vs dataset size, growth factors, NetworkX limitations | `scalability_results.csv` |

**Selecting a user (sidebar):** choose one of the 300 *evaluated users* (the same users used in the Phase 1 evaluation, so hits can be checked), or switch to *Enter a user ID* to type any visitor ID. An ID that is not in the training data demonstrates the cold-start fallback.

---

## 6. Implementation decisions (for the viva)

1. **Algorithms are not changed.** `src/` contains the notebook code ported unchanged. A test re-runs every method for all 300 evaluated users and compares with the saved Phase 1 recommendations: **1,197 / 1,200 lists identical, the other 3 differ only between products with exactly equal scores** (see point 5).
2. **No expensive recomputation.** The dashboard loads the saved graph (`bipartite_graph.pkl`) and the saved CF similarity matrix (`item_similarity.npz` + `cf_index.json`). Only the per-user step runs live. CF and Common Neighbors take milliseconds; PageRank takes ~0.5 s because NetworkX iterates over the whole graph. Results are cached per user.
3. **Metrics are never recomputed for display.** Comparison and scalability pages read the saved CSVs. A test recomputes the comparison table from the per-user results and checks it equals the saved table exactly.
4. **Graceful failure.** Each file is loaded with a check. A missing or corrupted results file shows an error on the page that needs it, and other pages keep working. If the graph pickle cannot be loaded (e.g. different NumPy/NetworkX version), the graph is **rebuilt from `train_interactions.csv.gz`** with the same builder (~1 s). If the CF files are missing, CF is re-fitted (~2 s). The sidebar's *Data status* shows which happened.
5. **Ties.** CF chooses the top 5 with NumPy's `argpartition`. When two products have exactly the same score at the 5th place, which one is chosen can differ between NumPy versions. Both are equally correct, so the dashboard reports such a case as a match and explains the tie.
6. **Product names.** Retailrocket item IDs are anonymised, so products are shown by ID with their popularity (number of users).

---

## 7. Testing record

**Recorded Linux validation on the saved outputs (Python 3.13, pandas 3.0, NumPy 2.5, NetworkX 3.7, SciPy 1.18, scikit-learn 1.9):**

| Test | Result |
|---|---|
| Backend unit tests (preprocessing, CF, CN, PPR, graph, metrics, scalability) | passed |
| All 29 Phase 1 files load, no issues | passed |
| Saved graph matches training data (edges, weights) | passed |
| Live recommendations vs saved Phase 1 for all 300 users × 4 methods | 1,197 exact + 3 ties, PageRank 300/300 exact |
| Comparison table recomputed from per-user results equals saved table | passed |
| No recommendation is an already-seen product; 5 unique items; sorted scores | passed |
| Cold-start / unknown user fallback | passed |
| Missing / corrupted files → clear errors, graph rebuild, CF refit | passed |
| User picker: displayed options map back to the right user IDs (regression test) | passed |
| **Total pytest** | **42 passed, 1 skipped** (the skipped file needs Streamlit) |
| All 7 pages executed for 6 scenarios (different users, unknown user, widget options, broken files) using Streamlit/Plotly test doubles | 0 errors |
| **Initial Windows validation with Streamlit** | 46 passed, 1 failed: user-selection test (fixed, see below) |

**Recommended local verification:**
1. `python -m pytest tests -v`: confirms everything on your Python/NumPy versions, **including `test_dashboard_app.py`** (real Streamlit, every page).
2. `streamlit run dashboard/app.py` and visually check each page: charts render, tables readable, user selection works, graph is interactive.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `'py' is not recognized` | Install Python 3.13 or 3.14 from python.org and tick **"Add python.exe to PATH"**, or use `python` instead of `py`. |
| `running scripts is disabled` when activating | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, or use `.venv\Scripts\activate.bat` in Command Prompt. |
| `'streamlit' is not recognized` | The venv is not active (no `(.venv)` in the prompt). Activate it, or run `python -m streamlit run dashboard/app.py`. |
| `ModuleNotFoundError: No module named 'src'` | Run the command from the `SDS_MiniProj` folder, not from inside `dashboard/`. |
| pip fails building numpy/pandas/scipy | No prebuilt packages exist for your Python version. Use a tested version, e.g. `py -3.13 -m venv .venv` (or 3.14). Python 3.9 or older is not supported. |
| Sidebar says *Graph: rebuilt* | The saved pickle could not be loaded on your versions (usually NumPy < 2). Harmless: the graph was rebuilt from the same data. `pip install "numpy>=2"` to use the saved file. |
| Red error mentioning a file in `outputs/` | That file is missing or damaged. Re-extract it from the Phase 1 `outputs.zip` into the same place. |
| Port 8501 already in use | `streamlit run dashboard/app.py --server.port 8502` |
| Browser does not open | Open http://localhost:8501 manually. |
| Charts look unstyled / deprecation warning about `use_container_width` | Harmless. Update with `pip install -U streamlit`. |
| Personalized PageRank page is slow the first time | Expected (~0.5 s per user, whole-graph power iteration). Later visits for the same user are cached. |

To re-run Phase 1 from scratch you need the raw `events.csv` and the notebook in `notebooks/` (Google Colab).

---

## 9. Change log

**Fix: user selection test (`ValueError: list.index(x): x not in list`).**
The sidebar selectbox used raw integer user IDs as its options and `format_func` to display labels such as
`29802 · 14 products · 8 future`. Streamlit's `AppTest` exposes the *displayed labels* as `options`, so selecting
one passed a label where an integer ID was expected. Now the options **are** the labels, and
`dashboard/ui_helpers.py` maps each label back to its user ID. The selected user is kept in
`st.session_state["selected_user"]`, so it survives page changes and switching between *Evaluated users* and
*Enter a user ID*. No algorithm or Phase 1 output was changed.
New/updated tests: `tests/test_ui_helpers.py` (label ↔ ID mapping), and `tests/test_dashboard_app.py`
(selecting a user updates all four recommendation lists to exactly that user's results; selection persists across
pages; typed IDs, cold start and switching modes work).

---

## 10. Data source, attribution and licences

**Dataset.** *Retailrocket recommender system dataset*, published by **Retailrocket** on Kaggle:
<https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset>.
The licence shown on the Kaggle listing is **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0)**, <https://creativecommons.org/licenses/by-nc-sa/4.0/> (checked on the listing on 7 November 2026).
Visitor and item IDs in the dataset are anonymised by the publisher.

**The raw dataset is not included in this repository.** To reproduce Phase 1, download it from Kaggle yourself and
run the notebook in `notebooks/`.

| Part of the repository | Origin | Licence |
|---|---|---|
| `src/`, `dashboard/`, `tests/`, `notebooks/` (code, no stored outputs), `.streamlit/config.toml`, `requirements.txt`, `.gitignore`, this README | Original work for this project | **MIT** ([`LICENSE`](LICENSE)) |
| `outputs/data/`, `outputs/models/`, `outputs/results/`, `outputs/visualizations/` | Derived from the Retailrocket dataset | **CC BY-NC-SA 4.0**, **not** MIT ([`DATA_LICENSE.md`](DATA_LICENSE.md)) |

**Changes made to the dataset to produce `outputs/`:** duplicate events removed; events weighted (view 1, add to cart 3,
transaction 5) and aggregated per visitor-item pair (capped at 15); chronological 80/20 split; users and items filtered
(≥ 5 items per user, ≥ 5 users per item, users with > 500 items removed); a bipartite graph, an item-item similarity
matrix, recommendations, evaluation metrics, scalability measurements and charts were computed from the result.

Anyone reusing the files in `outputs/` must credit Retailrocket, use them only for non-commercial purposes and share
any adaptations under the same licence. Retailrocket does not endorse this project.
