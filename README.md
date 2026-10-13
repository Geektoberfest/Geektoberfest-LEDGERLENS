# 🔎 LEDGERLENS

**Public Development Project Anomaly Detection**

LEDGERLENS is a Streamlit machine-learning prototype that uses exploratory analysis, engineered project metrics and Isolation Forest to flag unusual public development project behaviour for further human review.

## Live Demo

https://ledgerlens-e8hoghpl88mynxsm2jfnw2.streamlit.app/

## Problem

Public development projects produce information about project cost, spending, fund release, utilisation, progress and timelines. Unusual combinations of these values can help identify projects for further review.

LEDGERLENS is a **screening tool, not a fraud or corruption detector**. An anomaly means the project's behaviour is unusual relative to the portfolio; it does not prove wrongdoing.

## What the prototype does

- Overview and EDA dashboard
- Isolation Forest anomaly detection with selectable feature sets
- Review queue for flagged projects
- Project-level anomaly explanations (peer-aware, evidence-backed)
- Sector-aware peer comparisons
- Side-by-side baseline vs improved feature comparison tab

## ML Pipeline

```
Project data → Sanitization → Feature engineering → StandardScaler
    → Isolation Forest → Anomaly score → Human review
```

## Feature Sets

The app supports two feature sets selectable from the sidebar:

### ✨ Improved Feature Set (default — Issue #2)

| Feature | Formula | Signal |
|---|---|---|
| `cost_overrun` | `(actual_cost − est_cost) / est_cost` | Absolute budget overrun |
| `cost_overrun_peer_gap` | `cost_overrun − sector_median(cost_overrun)` | Overrun relative to sector peers |
| `fund_util_peer_gap` | `fund_utilisation − sector_median(fund_utilisation)` | Absorption lag vs sector peers |
| `budget_burn_vs_progress` | `(funds_used / est_cost) − (progress_pct / 100)` | True budget spend ahead of delivery |
| `schedule_slippage` | `(progress_pct / 100) − (elapsed_days / planned_days)` | Unclipped schedule delay |
| `idle_funds_ratio` | `(funds_released − funds_used) / est_cost` | Parked / unabsorbed public funds |

### 📋 Baseline Feature Set (original prototype)

| Feature | Signal |
|---|---|
| `cost_overrun` | Absolute budget overrun |
| `fund_utilisation` | Tranche absorption rate |
| `funds_released_ratio` | Funds released relative to estimate |
| `time_elapsed` | Elapsed fraction of planned duration |
| `progress_vs_time` | Progress lag (clipped at deadline) |
| `spend_vs_progress` | Tranche spend vs progress (lifecycle-conflated) |

For a detailed comparison of both feature sets — including correlation analysis,
anomaly detection results, and limitations — see [`docs/feature_engineering_report.md`](docs/feature_engineering_report.md).

## Dataset

The repository contains a reproducible **260-project synthetic starter dataset**. It is designed to simulate public development project behaviour and includes sectors, districts, estimated and actual cost, funds released and utilised, planned and elapsed days, and progress. It must not be presented as official government data.

## Tech Stack

Python • Pandas • NumPy • scikit-learn • Isolation Forest • Streamlit • GitHub

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Run tests

```bash
python -m pytest tests/ -v
# 33 tests: 3 original smoke tests + 30 feature-engineering tests
```

## Competition Workflow

The base repository is intentionally a working baseline. Participants should:

1. Pick one open GitHub Issue.
2. Clone the repository and reproduce the baseline.
3. Make the improvement on their own branch.
4. Compare the result with the baseline.
5. Open a Pull Request linked to the issue.
6. Explain what changed, why it helps, and how it was evaluated.

### Open Challenges

- **Easy:** [Audit anomaly results and establish evaluation baseline](https://github.com/Geektoberfest/Geektoberfest-LEDGERLENS/issues/1)
- **Medium:** [Improve project feature engineering](https://github.com/Geektoberfest/Geektoberfest-LEDGERLENS/issues/2)
- **Medium:** [Improve anomaly explanations with evidence](https://github.com/Geektoberfest/Geektoberfest-LEDGERLENS/issues/3)
- **Hard:** [Compare anomaly-detection methods](https://github.com/Geektoberfest/Geektoberfest-LEDGERLENS/issues/4)

Anomalies are candidates for human review, not proof of fraud or misconduct. Since the starter dataset is synthetic, contributors should explain what their evaluation can and cannot establish.

## Project Structure

```
LEDGERLENS/
├── app.py                          # Streamlit UI (feature-set toggle, comparison tab)
├── model.py                        # Data loading, feature engineering, Isolation Forest
├── sample_projects.csv             # 260-project synthetic dataset
├── requirements.txt
├── CONTRIBUTING.md
├── PRESENTATION.md
├── LICENSE
├── .gitignore
├── docs/
│   ├── baseline_report.md          # Issue #1 audit baseline results
│   └── feature_engineering_report.md  # Issue #2 feature comparison report
└── tests/
    ├── test_model.py               # 3 original smoke tests
    └── test_feature_engineering.py # 30 feature-engineering tests (Issue #2)
```
