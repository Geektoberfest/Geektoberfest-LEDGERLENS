# LEDGERLENS baseline audit (Issue #1)

> LEDGERLENS runs on a **synthetic** 260-project sample, not official government records. An anomaly is a signal for human review, **not** proof of fraud, corruption or misuse.

Regenerate this report with `python baseline_audit.py` (deterministic; no timestamps).

## 1. Baseline configuration

- Model: `IsolationForest(n_estimators=250, contamination=0.12, random_state=42)`
- Preprocessing: `StandardScaler` fitted on the 6 engineered features (no other preprocessing)
- Features: `cost_overrun`, `fund_utilisation`, `funds_released_ratio`, `time_elapsed`, `progress_vs_time`, `spend_vs_progress`
- Detector code is unchanged by this audit.

## 2. Anomaly count and rate

- Projects scored: **260**
- Flagged as anomalies: **32** (**12.3%**)
- This rate is **set by the `contamination=0.12` parameter**, not discovered from the data: the forest is told to flag roughly the lowest-scoring 12%. It should not be read as an estimate of how many projects are truly problematic.

Flagged projects per sector:

| sector | projects | flagged | rate |
|---|---|---|---|
| Roads | 62 | 10 | 0.161 |
| Health | 48 | 7 | 0.146 |
| Power | 43 | 5 | 0.116 |
| Water | 61 | 7 | 0.115 |
| Schools | 46 | 3 | 0.065 |

## 3. What distinguishes flagged from unflagged projects

Median feature values (unscaled):

| feature | flagged_median | unflagged_median | difference |
|---|---|---|---|
| cost_overrun | 0.105 | 0.053 | 0.052 |
| fund_utilisation | 0.907 | 0.899 | 0.008 |
| funds_released_ratio | 0.913 | 0.814 | 0.099 |
| time_elapsed | 0.779 | 0.729 | 0.050 |
| progress_vs_time | -0.333 | 0 | -0.333 |
| spend_vs_progress | 0.658 | 0.171 | 0.487 |

Largest median gaps between flagged and unflagged projects: `spend_vs_progress` and `progress_vs_time`. Medians describe the typical flagged project; a few of the most extreme cases are driven by a different feature (see below).

Primary driver per flagged project (feature furthest from the portfolio mean, with direction):

| primary_feature | direction | flagged_projects |
|---|---|---|
| progress_vs_time | low | 15 |
| cost_overrun | high | 5 |
| spend_vs_progress | high | 5 |
| fund_utilisation | low | 4 |
| funds_released_ratio | low | 2 |
| time_elapsed | high | 1 |

## 4. Representative cases

`top_unusual_features` lists the two standardized features furthest from the portfolio mean. Isolation Forest has no built-in attribution, so this is a descriptive proxy, not the model's reasoning.

**Most anomalous (lowest score)**

| id | sector | anomaly_score | top_unusual_features |
|---|---|---|---|
| PRJ-2074 | Health | -0.192 | fund_utilisation (-8.8σ), spend_vs_progress (-3.5σ) |
| PRJ-2177 | Water | -0.148 | time_elapsed (+3.9σ), progress_vs_time (-3.5σ) |
| PRJ-2090 | Water | -0.145 | fund_utilisation (-7.7σ), funds_released_ratio (-1.7σ) |
| PRJ-2170 | Schools | -0.131 | progress_vs_time (-4.1σ), time_elapsed (+2.8σ) |
| PRJ-2159 | Water | -0.118 | progress_vs_time (-3.9σ), time_elapsed (+3.2σ) |

**Borderline flagged (just inside the 12% cut)**

| id | sector | anomaly_score | top_unusual_features |
|---|---|---|---|
| PRJ-2027 | Roads | -0.008 | fund_utilisation (-1.6σ), funds_released_ratio (-1.3σ) |
| PRJ-2130 | Power | -0.007 | spend_vs_progress (+1.5σ), cost_overrun (+1.5σ) |
| PRJ-2237 | Power | -0.002 | funds_released_ratio (-2.3σ), spend_vs_progress (+1.7σ) |

**Borderline unflagged (just outside the cut)**

| id | sector | anomaly_score | top_unusual_features |
|---|---|---|---|
| PRJ-2112 | Roads | 0.018 | funds_released_ratio (-2.2σ), time_elapsed (-1.6σ) |
| PRJ-2146 | Roads | 0.020 | spend_vs_progress (+1.4σ), fund_utilisation (+1.4σ) |
| PRJ-2037 | Water | 0.026 | time_elapsed (-1.4σ), spend_vs_progress (+1.2σ) |

**Most typical (highest score)**

| id | sector | anomaly_score | top_unusual_features |
|---|---|---|---|
| PRJ-2223 | Roads | 0.141 | spend_vs_progress (-1.1σ), funds_released_ratio (+1.0σ) |
| PRJ-2115 | Health | 0.141 | spend_vs_progress (-1.0σ), funds_released_ratio (+1.0σ) |
| PRJ-2032 | Power | 0.142 | spend_vs_progress (-1.2σ), funds_released_ratio (+1.0σ) |

## 5. Reproducibility and stability

- With the fixed seed (`random_state=42`) results are identical on every run (`tests/test_model.py` asserts exactly 32 flags).
- Re-running with 10 different seeds flags exactly 32 projects each time. Mean Jaccard overlap with the baseline flag set is **0.99** (minimum 0.94).
- **30** projects are flagged under every seed; **33** are flagged under at least one. The remaining 3 are borderline and depend on the seed.

Sensitivity to the `contamination` setting (seed 42):

| contamination | flagged | overlap_with_baseline |
|---|---|---|
| 0.050 | 13 | 13 |
| 0.080 | 21 | 21 |
| 0.120 | 32 | 32 |
| 0.150 | 39 | 32 |

## 6. Data observations

- 66 projects have run past their planned duration (max elapsed/planned = 1.97); `progress_vs_time` clips this at 1.0, so overruns beyond the plan are not distinguished.
- `spend_vs_progress` > 0.2 (spending ahead of progress) for 136 projects; < -0.2 (progress ahead of spending) for 1. Spending ahead of progress is therefore common in this sample, so it is a weaker signal on its own than the extremes.
- 5 projects have a cost overrun above 50%; 48 projects report 100% progress.
- `load_data()` fills missing or non-numeric values with 0. The sample has no missing values, so this has no effect here, but on real data it could silently create misleading zeros.

## 7. Ground truth

`sample_projects.csv` has **no ground-truth or planted-anomaly marker** (columns: project_id, project_name, sector, district, estimated_cost, actual_cost, funds_released, funds_utilised, planned_days, elapsed_days, progress). Planted anomalies therefore cannot be checked, and **no accuracy, precision or recall is reported**, in line with CONTRIBUTING.md.

`python baseline_audit.py --labels labels.csv` (columns `id,label`) evaluates flags against labels if they become available. Labels are read only after the model is fitted.

## 8. Limitations

- **No reliable labels:** without ground truth, anomaly detection cannot be scored objectively; 'flagged' only means statistically unusual relative to this portfolio.
- **Detected outliers are not confirmed misuse:** an unusual project can be legitimate (for example a project with a justified scope change), and a problematic project can look ordinary.
- **Synthetic data:** the sample was generated to simulate project behaviour; findings do not describe any real government project or district.
- **Rate is a parameter:** the 12.3% flag rate follows from `contamination=0.12`.
- **Borderline cases are unstable:** projects near the cut-off change with the seed and contamination value, so the review queue should be read as a ranking, not a verdict.
- **Scaling and features are fixed:** the six engineered features and standardization shape what counts as unusual; changing them (issues #2 and #4) will change which projects are flagged.
- **Descriptive drivers only:** feature-level reasons in this report are z-score proxies, not model attributions.
