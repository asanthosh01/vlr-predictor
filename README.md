# VALORANT Esports Match Predictor

An educational machine-learning application that estimates professional match outcomes using **only results from earlier dates**. The October 2026 update replaces missing-model startup failures, current-ranking joins, and hard-coded accuracy displays with a dated dataset, exported XGBoost model, and reproducible evaluation.

## Quick start

Python 3.11+ recommended. From the repository root:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. The committed artifacts and dataset make the interface runnable without collecting new data. When files are missing or incompatible, the application displays setup instructions rather than crashing. This is a local application; no public deployment is claimed.

## Reproduce the experiment

```bash
python -m src.evaluate --data data/raw/matches.csv --out artifacts
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Optional refresh (requests are spaced two seconds apart; site markup can change):

```bash
python -m src.collect_history --pages 60 --out data/raw/matches.csv
python -m src.evaluate
```

## Actual recorded results

The checked-in snapshot contains **2,983 unique completed matches** with stable match IDs, dates, and individual VLR source URLs. Collection timestamp and source are recorded in `data/raw/matches.provenance.json`; `artifacts/evaluation.json` includes the dataset SHA-256 and package versions.

| Chronological test method | Accuracy | Brier score (lower is better) |
| --- | ---: | ---: |
| XGBoost | 54.1% | 0.251 |
| Training-majority baseline | 51.2% | 0.252 |
| Elo baseline | 57.6% | 0.243 |

The test contains 340 matches, dated July 28–October 4, 2026. XGBoost's 95% Wilson accuracy interval is 48.8%–59.3%. **The model does not outperform Elo on this test.** This is a useful negative experimental result, not evidence of a production-ready forecasting edge. Earlier 68%, 69%, and 71.7% figures are not validated by this experiment and should not be used to describe it.

## Evaluation design

- Dates are divided into 60% training, 20% validation, and 20% test blocks. Matches on one date never cross splits.
- Features include historical Elo difference, smoothed win rates, match counts, recent five-match form, and head-to-head records. Present-day ranking snapshots are excluded.
- Every match uses earlier dates only. Same-day outcomes become available the following day, because exact completion times are not available.
- The classifier is trained once with fixed parameters. Earlier validation/test outcomes update history for later predictions, matching sequential use, but do not refit the model.
- Probabilities are raw, not calibrated. Accuracy, balanced accuracy, ROC AUC, Brier score, and majority/Elo baselines are saved, including the date ranges.

## Structure

- `src/collect_history.py`: dated collection, deduplication, provenance
- `src/chronological.py`: validation, historical features, day-isolated updates
- `src/evaluate.py`: splits, XGBoost training, baselines, evaluation export
- `app.py`, `templates/index.html`: local interface and input validation
- `artifacts/`: portable model JSON, metrics, feature rows
- `tests/`: leakage checks, date splitting, parser, model export and API tests

Older research scripts and the exploratory notebook are retained for history; the commands above are the supported workflow. The dataset is a dated public-results snapshot, not a live feed. Inconsistent team aliases, team roster changes, cross-region strength, and selection bias remain limitations. Stronger future work includes event-grouped backtesting, alias resolution, map/roster features, and calibration measured on a separate validation period.

## Development note

This project was developed with AI assistance. The reproducibility and evaluation update was added in October 2026; it does not establish what was completed during an earlier résumé date range.
