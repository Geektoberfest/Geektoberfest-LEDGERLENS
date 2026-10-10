# 🔎 LEDGERLENS

**Public Development Project Anomaly Detection**

LEDGERLENS is a Streamlit machine-learning prototype that uses exploratory analysis, engineered project metrics and Isolation Forest to flag unusual public development project behaviour for further human review.

## Live Demo

https://ledgerlens-e8hogpl88mynxsm2jfnw2.streamlit.app/

## Problem

Public development projects produce information about project cost, spending, fund release, utilisation, progress and timelines. Unusual combinations of these values can help identify projects for further review.

LEDGERLENS is a **screening tool, not a fraud or corruption detector**. An anomaly means the project's behaviour is unusual relative to the portfolio; it does not prove wrongdoing.

## What the prototype does

- Overview and EDA dashboard
- Isolation Forest anomaly detection
- Review queue for flagged projects
- Project-level anomaly explanations
- Sector-aware peer comparisons

## ML Pipeline

Project data → Feature engineering → Standardization → Isolation Forest → Anomaly score → Human review

## Baseline Features

- Cost overrun
- Fund utilisation
- Funds released ratio
- Time elapsed
- Progress vs time
- Spending vs progress

## Dataset

The repository contains a reproducible **260-project synthetic starter dataset**. It is designed to simulate public development project behaviour and includes sectors, districts, estimated and actual cost, funds released and utilised, planned and elapsed days, and progress. It must not be presented as official government data.

## Tech Stack

Python • Pandas • NumPy • scikit-learn • Isolation Forest • Streamlit • GitHub

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
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

- **Easy:** [Audit anomaly results and establish evaluation baseline](https://github.com/radhikadhingra820-hue/LEDGERLENS/issues/1)
- **Medium:** [Improve project feature engineering](https://github.com/radhikadhingra820-hue/LEDGERLENS/issues/2)
- **Medium:** [Improve anomaly explanations with evidence](https://github.com/radhikadhingra820-hue/LEDGERLENS/issues/3)
- **Hard:** [Compare anomaly-detection methods](https://github.com/radhikadhingra820-hue/LEDGERLENS/issues/4)

Anomalies are candidates for human review, not proof of fraud or misconduct. Since the starter dataset is synthetic, contributors should explain what their evaluation can and cannot establish.

## Project Structure

```
LEDGERLENS/
├── app.py
├── model.py
├── sample_projects.csv
├── requirements.txt
├── CONTRIBUTING.md
├── PRESENTATION.md
├── LICENSE
├── .gitignore
└── tests/
    └── test_model.py
```
