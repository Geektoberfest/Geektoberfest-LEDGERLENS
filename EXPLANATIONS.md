# Evidence-Based Anomaly Explanations (Issue #3)

Each project in the review queue now shows the numbers behind its explanation: the observed value, the peer median, the size and direction of the difference, and a robust z-score ranking. The explanations are statistical comparisons. They show what is unusual and why a reviewer should look, not whether anything improper happened.

## What a reviewer sees

In the **Project Details** tab:

1. A notice that the explanations are statistical comparisons, not evidence of misconduct.
2. The peer group used for comparison, with its size. The project itself is excluded from its own peers.
3. A list of warnings, one per signal that stands out against its peers.
4. An evidence table with all six signals ranked by how unusual they are, showing the project value, peer median, robust z-score, and whether the signal counts as a warning.

## Signals

The same six features as the baseline (`FEATURES` in `model.py`):

| Signal | What it measures | Warning direction |
|---|---|---|
| Cost overrun | (actual − estimated cost) / estimated cost | High |
| Share of released funds spent | funds used / funds released | High |
| Share of estimated cost released | funds released / estimated cost | Context only |
| Share of planned time elapsed | elapsed days / planned days | High |
| Physical progress relative to time elapsed | progress − min(time elapsed, 1) | Low (behind schedule) |
| Fund use relative to physical progress | fund utilisation − progress | High (spending ahead of progress) |

"Share of estimated cost released" is shown in the table for context but never produces a warning, because a low release is not clearly a warning sign on its own.

## Peer groups

Peers are other projects in the same **sector and size band**, where the size band is a tercile of estimated cost within the sector (Small, Medium, Large). Each band has about 14–21 projects.

If a band has fewer than 8 peers, the comparison falls back to all projects in the sector. The explanation names the group it used and how many projects it contains.

## How a signal becomes a warning

For each signal, the project is compared with its peer median. A signal is a **warning** when all three hold:

1. It is in the warning direction (see table above).
2. Its robust z-score is at least 2 in absolute value.
3. Its gap from the peer median is at least the signal's minimum gap:
   - 10 percentage points for cost overrun and share of funds spent
   - 15 percentage points for time elapsed, progress vs time, and spend vs progress

**Robust z-score:** (value − peer median) / (1.4826 × median absolute deviation of peers). The median absolute deviation resists outliers, so one extreme peer does not inflate the spread. If the peers have no spread at all, the z-score is 0.

Warnings are ranked by the absolute robust z-score, so the most unusual signal appears first.

If no signal is a warning and the project is flagged by Isolation Forest, the explanation says that no single signal stands out but the combination is unusual relative to the portfolio. If the project is not flagged and has no warnings, it shows "No signal stands out against comparable projects."

## Examples

All values come from the baseline run on `sample_projects.csv`. Reproduce them with the code in the last section.

### Flagged, with several warnings: PRJ-2010, Substation #11 (Power, Large)

- Physical progress relative to time elapsed is −55 points, against a peer median of 0 points for Power projects of large size (13 projects).
- Fund use relative to physical progress is 60 points, against a peer median of −6 points.
- Cost overrun is 30%, 20 points above the peer median of 10%.

### Flagged, with warnings: PRJ-2025, District hospital wing #26 (Health, Large)

- Physical progress relative to time elapsed is −36 points, against a peer median of −3 points for Health projects of large size (15 projects).
- Cost overrun is 30%, 23 points above the peer median of 7%.
- Fund use relative to physical progress is 65 points, against a peer median of 3 points.

### Flagged, but no warning: PRJ-2027, Bridge repair #28 (Roads, Large)

The model flags this project, but no single signal passes all three warning tests. The explanation says so and points to the combination of metrics. Isolation Forest looks at all six features together, so a project can be flagged without any single signal being extreme.

### Not flagged, but with a warning: PRJ-2034, Trauma unit #35 (Health, Medium)

- Share of planned time elapsed is 117%, 61 points above the peer median of 56% for Health projects of medium size (15 projects).

This project is not in the review queue, but its schedule overrun is shown to the reviewer. The per-project explanation and the model flag are separate checks.

### Not flagged, no warning: PRJ-2000, Substation #1 (Power)

No signal passes the warning tests, so the Details tab shows "No signal stands out against comparable projects."

## Limitations

- **Synthetic data.** The dataset is synthetic. Peer medians and warnings describe the synthetic portfolio, not real public projects.
- **Statistical, not causal.** A warning says a project differs from its peers. It does not say why, and there may be legitimate reasons such as scope changes, price escalation, or reporting delays.
- **Small peer groups.** Each size band has 13–20 peers once the project itself is excluded, so medians and spreads are noisy. The fallback to sector-level peers is used only when a band has fewer than 8.
- **Hand-set thresholds.** The z-score cut-off of 2, the minimum gaps, and the size bands are judgement calls. They have not been tuned against labels, because none exist.
- **Multiple comparisons.** Six signals are tested per project, so some warnings will appear by chance.
- **Signals are not independent.** Cost overrun and spending relative to progress can move together, so one underlying issue may appear as several warnings.
- **Warnings and flags are different checks.** The warnings are univariate (one signal at a time). The flag is multivariate (Isolation Forest). As PRJ-2027 and PRJ-2034 show, they can disagree.
- **Size is measured by estimated cost.** Size bands do not adjust for inflation or differences across years.
- **Not evidence of misconduct.** Every explanation is a candidate for human review. Nothing in this view establishes fraud or corruption.

## Reproducing the examples

```bash
python -c "
from model import run_anomaly_detection, explain_project, build_evidence
data=run_anomaly_detection()['data']
for pid in ['PRJ-2010','PRJ-2025','PRJ-2027','PRJ-2034','PRJ-2000']:
    row=data[data['id']==pid].iloc[0]
    print(pid, row['name'], row['sector'], row['size_band'], row['anomaly'])
    for reason in explain_project(row,data): print('  ', reason)
"
```

Tests for the evidence logic live in `tests/test_explanations.py`.
