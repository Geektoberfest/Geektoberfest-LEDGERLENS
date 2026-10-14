# Project Feature Engineering (Issue #2)

**Question:** Do better features make unusual public-project cost, spending, fund-utilisation, timeline and progress patterns easier to identify?

**Short answer:** Not with the evidence available. Three feature sets were compared under the same detector and evaluation setup. Neither revised set improves detection of planted anomalies, and the revised sets flag almost the same projects as the baseline. The baseline six features stay as the default. The new features are implemented and documented so the result can be checked and extended.

Reproduce with:

```bash
python evaluate_features.py
python -m pytest tests/test_features.py
```

## Baseline review

The six baseline indicators (`model.py`, `FEATURES`):

| Feature | Definition | Review |
|---|---|---|
| `cost_overrun` | (actual − estimated cost) / estimated cost | Measures cost only; ignores how much work is done. |
| `fund_utilisation` | funds used / funds released, capped to [0, 1] | Spending against releases, not against progress. |
| `funds_released_ratio` | funds released / estimated cost | Context for how much funding has arrived. |
| `time_elapsed` | elapsed days / planned days | Can exceed 1 for overrunning projects (66 projects here). |
| `progress_vs_time` | progress − min(time elapsed, 1) | A difference, so overruns are capped at 1. |
| `spend_vs_progress` | fund utilisation − progress | Spending relative to physical progress. |

Two gaps stand out. First, cost and timeline are never combined with progress, so a project that is 10% over budget with 90% of the work done looks the same as one with 10% of the work done. Second, every feature is compared with the whole portfolio, although costs and timelines differ across sectors.

## New features

Each new feature has a definition and a reason for including it.

| Feature | Definition | Rationale |
|---|---|---|
| `schedule_performance` | progress / time elapsed, capped to [0, 3] | Work completed per unit of time used. Below 1 means behind schedule. Unlike `progress_vs_time`, it scales with the size of the schedule. |
| `cost_performance` | (progress × estimated cost) / actual cost, capped to [0, 3] | Earned-value cost efficiency: the value of work done divided by money spent. Below 1 means work is costing more than planned. It combines cost and progress, which the baseline never does. |
| `spend_pace` | fund utilisation / time elapsed, capped to [0, 3] | How fast funds are being used relative to schedule. High values mean spending is running ahead of the timeline. |
| `peer_cost_overrun_z` | robust z-score of `cost_overrun` within the sector | Puts overruns on a common scale across sectors whose typical overruns differ. |
| `peer_spend_gap_z` | robust z-score of `spend_vs_progress` within the sector | The same idea for spending relative to progress. |

The robust z-score is (value − sector median) / (1.4826 × sector median absolute deviation), falling back to the sector standard deviation when the deviation is zero.

Feature sets compared:

- **baseline:** the six baseline features
- **ratios:** baseline + `schedule_performance`, `cost_performance`, `spend_pace`
- **ratios+peers:** ratios + `peer_cost_overrun_z`, `peer_spend_gap_z`

## Data handling

The dataset is clean in most respects. Still, the code checks for each problem and counts every correction. The counts are printed by `evaluate_features.py`.

| Issue | Handling | Count on the starter data |
|---|---|---|
| Missing values | Reported. Missing values are not filled with another project's value. | 0 |
| Non-positive denominators (estimated cost, actual cost, funds released, planned days) | Set to missing, then the affected feature gets a neutral value: 0 for deviations, 1 for performance ratios. | 0 |
| Zero elapsed time | Performance ratios that divide by time get the neutral value 1. | 0 |
| Progress in percent instead of fraction | Detected and rescaled. | Not triggered |
| Progress outside [0, 1] | Clipped and counted. | 0 |
| Funds used above funds released | Clipped to funds released and counted. | 0 |
| Elapsed time beyond planned time | Kept. It is a real overrun, and `time_elapsed` > 1 is meaningful. | 66 |
| Extreme values | Each feature is winsorised at its 1st and 99th percentiles before scaling. | Applied to all sets |

The baseline `load_data` silently fills missing values with 0 and rescales progress with a heuristic (the maximum progress is at most 1.5). `features.py` makes both steps explicit and counted. On the starter data the baseline features match `model.py` exactly, which is checked in `tests/test_features.py`.

## Evaluation setup

All three feature sets are run through the same pipeline:

- Winsorising at the 1st and 99th percentiles, then `StandardScaler`, fitted on the data being scored.
- `IsolationForest` with 250 trees, contamination 0.12, and random state 42, unless a seed is stated.

Because the dataset has no ground-truth labels, two measures are used.

1. **Stability on the real data:** agreement between seeds (10 seeds) and between 30 random 80% subsamples. Each subsample refits the whole pipeline and is compared with the full-data flags for the same projects.
2. **Planted-anomaly recall:** known anomaly patterns are injected into copies of the data, and each feature set is scored on how many planted projects it flags. The planted labels are used only for scoring. They never reach the features or the detector. There are 5 trials, each planting 20 projects.

   - *Schedule slip:* elapsed days = 1.6 × planned days.
   - *Cost overrun:* actual cost = 1.5 × estimated cost.
   - *Spend ahead of progress:* progress halved, and funds used = 95% of funds released.

   Recall is the share of planted projects flagged. The unplanted flag rate shows how often clean projects are flagged.

Leakage: the features use only the project's own values and sector medians. Sector medians include the project itself, which is a small effect given sector sizes of 43–62. No labels, planted or otherwise, are used to build any feature or to fit the detector.

## Results

### Stability on the real data

| Feature set | Flagged | Seed agreement (Jaccard) | Subsample agreement (Jaccard) | Subsample std |
|---|---|---|---|---|
| baseline | 32 | 0.972 | 0.925 | 0.043 |
| ratios | 32 | 0.945 | 0.913 | 0.039 |
| ratios+peers | 32 | 0.930 | 0.904 | 0.041 |

Adding features reduces agreement slightly. The differences are small, but the baseline is at least as stable as the revised sets.

### Planted-anomaly recall

Mean over 5 trials of 20 planted projects each. One standard error is roughly ±0.04–0.05.

| Pattern | Feature set | Recall | Unplanted flag rate |
|---|---|---|---|
| Schedule slip | baseline | 0.35 | 0.10 |
| Schedule slip | ratios | 0.35 | 0.10 |
| Schedule slip | ratios+peers | 0.30 | 0.11 |
| Cost overrun | baseline | 0.27 | 0.11 |
| Cost overrun | ratios | 0.22 | 0.12 |
| Cost overrun | ratios+peers | 0.27 | 0.11 |
| Spend ahead of progress | baseline | 0.17 | 0.12 |
| Spend ahead of progress | ratios | 0.19 | 0.12 |
| Spend ahead of progress | ratios+peers | 0.23 | 0.11 |

The revised sets do not reliably improve recall. The one pattern where ratios+peers looks better, spend ahead of progress (0.23 vs 0.17), is about one standard error and is the pattern the new features were designed around. Recall is low for all sets, so Isolation Forest at 12% contamination misses most planted projects with these effect sizes.

### Example projects

The two revised sets differ on only one project in each direction (31 projects are flagged by both).

- **PRJ-2176 (Roads), flagged only by ratios+peers:** cost overrun 5%, time elapsed 29%, progress minus time −9 points, cost performance 0.19, schedule performance 0.69, sector cost-overrun z-score −0.06. The new flag comes from the earned-value ratio. The cost overrun is near zero, so the signal is that very little progress has been made for the money spent.
- **PRJ-2027 (Roads), flagged only by baseline:** cost overrun 3%, time elapsed 63%, progress minus time −15 points, cost performance 0.46, schedule performance 0.75, sector cost-overrun z-score −0.47. Under the revised features, the earned-value ratio is no longer unusual enough to flag it.

The changes are real but small, and they go in both directions. They do not show a clear improvement.

## Limitations

- **Synthetic data.** The starter dataset is synthetic. Both the real-data stability and the planted patterns describe the synthetic portfolio, not public projects in practice.
- **Planted patterns are our own.** They are chosen by us, so they measure sensitivity to the effects we chose to plant. A real fraud or misconduct pattern could look different, and a result on these patterns does not show that a feature set works on real cases.
- **Redundant features.** Several new features are almost duplicates of baseline features. Correlations on the starter data: `peer_cost_overrun_z` with `cost_overrun` 0.99, `peer_spend_gap_z` with `spend_vs_progress` 0.99, `schedule_performance` with `progress_vs_time` 0.92, and `cost_performance` with `spend_vs_progress` −0.95. Adding them mostly rescales existing signals, which explains why the sets agree so closely.
- **Small sectors.** Sector medians and spreads come from 43–62 projects, so robust z-scores are noisy, and the self-inclusion effect is not negligible for the smallest sector.
- **Noise in the benchmark.** Five trials of 20 planted projects give a standard error of about ±0.05 on recall, so only large differences are detectable.
- **Clipping choices.** Capping ratios at 3 and winsorising at the 1st and 99th percentiles are judgement calls. The `spend_pace` cap is reached by at least 1% of projects.
- **Not evidence of misconduct.** Flags from any feature set are candidates for human review. Nothing here shows that a flagged project involves fraud or corruption.

## What to try next

- Remove the redundant peer features and test whether a decorrelated set does better.
- Add a benchmark with milder planted effects, so recall is not near the floor.
- Evaluate on real public-project data with reliable labels, when available.
