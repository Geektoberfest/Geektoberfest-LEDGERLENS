# LEDGERLENS — Slide Outline

## 1. Title
### LEDGERLENS
Public Development Project Anomaly Detection

## 2. Problem
Public development projects can show unusual relationships between spending, cost, utilisation, progress and timelines.

## 3. Solution
LEDGERLENS uses engineered project indicators and Isolation Forest to flag unusual projects for further human review.

## 4. ML Pipeline
Project data → Feature engineering → Standardization → Isolation Forest → Anomaly score

## 5. Prototype
Show:
- EDA dashboard
- Anomaly review queue
- Project details
- Sector-aware explanations
- Spending vs progress visualization

## 6. Baseline
- 260-project synthetic starter dataset
- Six engineered features
- Isolation Forest
- Rule-based sector peer explanations

## 7. Competition Challenge
Participants improve the baseline through difficulty-labelled GitHub Issues and Pull Requests:
- **Easy:** Audit anomaly results and establish evaluation baseline
- **Medium:** Improve project feature engineering
- **Medium:** Improve anomaly explanations with evidence
- **Hard:** Compare anomaly-detection methods

## 8. Tech Stack
Python • Pandas • NumPy • scikit-learn • Streamlit • GitHub

## 9. Important Limitation
An anomaly is not proof of fraud or corruption. LEDGERLENS is a screening prototype for prioritising projects for human review.
