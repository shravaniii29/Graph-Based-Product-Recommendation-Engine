# Licence notice for dataset-derived files

The MIT licence in [`LICENSE`](LICENSE) covers **only the original project code** of this repository.
It does **not** apply to the files listed below, which are derived from a third-party dataset and keep that
dataset's licence.

## Files covered by this notice

Everything inside the `outputs/` folder:

| Folder | Contents |
|---|---|
| `outputs/data/` | filtered and weighted visitor-item interactions, test ground truth, item popularity |
| `outputs/models/` | saved bipartite graph, item-item similarity matrix and its ID index |
| `outputs/results/` | recommendations, evaluation metrics, scalability measurements, statistics, run settings |
| `outputs/visualizations/` | charts generated from the above |

## Source and licence

- **Dataset:** Retailrocket recommender system dataset
- **Publisher:** Retailrocket
- **Source:** https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
- **Licence:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0),
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  (as shown on the Kaggle listing, checked on 7 November 2026)

The derived files above are shared under the same licence, **CC BY-NC-SA 4.0**. When reusing them you must give
appropriate credit to Retailrocket, use them for non-commercial purposes only, and distribute any adaptations under
the same licence.

## Changes made

Duplicate events were removed; events were weighted (view 1, add to cart 3, transaction 5) and aggregated per
visitor-item pair (capped at 15); the data was split chronologically (80% train, 20% test); users and items were
filtered (at least 5 items per user, at least 5 users per item, users with more than 500 items removed). The graph,
similarity matrix, recommendations, metrics, scalability measurements and charts were computed from this processed
data.

The raw dataset files are not included in this repository. Retailrocket does not endorse this project.
