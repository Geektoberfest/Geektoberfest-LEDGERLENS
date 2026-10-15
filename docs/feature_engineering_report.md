# Feature Engineering Report — Issue #2
## Improved Feature Set for Public Development Project Anomaly Detection

**Branch:** `improve-feature-engineering-issue-2`  
**Dataset:** `sample_projects.csv` — 260 synthetic public development projects  
**Detector:** Isolation Forest (`n_estimators=250`, `contamination=0.12`, `random_state=42`)

---

## 1. Motivation

The baseline prototype used six indicators derived directly from raw columns.
Three structural problems limited their signal quality:

| Problem | Baseline behaviour | Impact |
|---|---|---|
| **Clipped schedule signal** | `progress_vs_time` clips `time_elapsed` at 1.0 | Projects already past deadline all map to the same clipped range; severity of overrun is hidden |
| **Conflated fund-progress gap** | `spend_vs_progress` divides `funds_used` by `funds_released`, not by `est_cost` | Tranche absorption rate is measured, not true budget expenditure vs delivery |
| **No peer context** | All thresholds are portfolio-wide | A cost overrun of 30% is normal for a Roads project but extreme for a Schools project; a global detector penalises sector norms rather than genuine outliers |

---

## 2. Revised Indicators

### 2.1 Retained from Baseline
- **`cost_overrun`** — `(actual_cost − est_cost) / est_cost`, winsorized `[−1.0, 10.0]`.  
  Core budget-discipline signal. Retained as anchor.
- **`fund_utilisation`** — `funds_used / funds_released`, clipped `[0, 1]`.  
  Tranche absorption rate. Retained for continuity with audit conventions.

### 2.2 Dropped from Baseline
| Feature | Reason |
|---|---|
| `funds_released_ratio` | High correlation with `time_elapsed` (r = 0.85); adds collinearity, not signal |
| `time_elapsed` | Used as component in `schedule_slippage`; standalone adds noise |
| `progress_vs_time` | Replaced by unclipped `schedule_slippage` |
| `spend_vs_progress` | Replaced by `budget_burn_vs_progress` (true budget denominator) |

### 2.3 New Improved Indicators

#### `budget_burn_vs_progress`
```
(funds_used / est_cost) − (progress_pct / 100)
```
Measures the divergence between **total budget expended** and **physical delivery**.
Uses `est_cost` as denominator (not `funds_released`) so the metric is consistent
across all lifecycle stages.  
**Range:** Winsorized `[−2.0, 2.0]`

#### `schedule_slippage`  
```
(progress_pct / 100) − (elapsed_days / planned_days)
```
Removes the artificial 1.0 ceiling present in the baseline `progress_vs_time`.
Projects past their planned deadline continue to produce increasingly negative values,
preserving visibility into severe overruns.  
**Correlation with `progress_vs_time`:** r = 0.913 for on-schedule projects,
diverges significantly once `time_elapsed > 1`.

#### `idle_funds_ratio` (Parked Capital)
```
(funds_released − funds_used) / est_cost
```
Treasury funds released to the implementing agency but sitting unspent.
High idle ratios are a standard audit red flag for absorption capacity failure.  
**Range:** Clipped `[0, 2.0]`

#### `cost_overrun_peer_gap`
```
cost_overrun − sector_median(cost_overrun)
```
Normalises cost overrun against the implementing sector's typical behaviour.
A project with a 25 % overrun in a sector where the median is 5 % is flagged;
the same overrun in a sector where the median is 22 % is not.  
**Range:** Winsorized `[−5.0, 10.0]`

#### `fund_util_peer_gap`
```
fund_utilisation − sector_median(fund_utilisation)
```
Flags abnormally low tranche absorption relative to sector peers — a common
indicator of stalled implementation despite fund release.  
**Range:** Full float; near-zero sector median by construction.

---

## 3. Feature Correlation Analysis

Computed on the full 260-project dataset.

| | `budget_burn_vs_progress` | `schedule_slippage` | `idle_funds_ratio` | `cost_overrun_peer_gap` | `fund_util_peer_gap` |
|---|---|---|---|---|---|
| `budget_burn_vs_progress` | 1.000 | −0.636 | −0.365 | 0.139 | 0.364 |
| `schedule_slippage` | −0.636 | 1.000 | −0.070 | −0.066 | −0.039 |
| `idle_funds_ratio` | −0.365 | −0.070 | 1.000 | −0.008 | −0.859 |
| `cost_overrun_peer_gap` | 0.139 | −0.066 | −0.008 | 1.000 | 0.019 |
| `fund_util_peer_gap` | 0.364 | −0.039 | −0.859 | 0.019 | 1.000 |

> [!NOTE]  
> `idle_funds_ratio` and `fund_util_peer_gap` share r = −0.859 because high idle funds  
> mechanically depress tranche absorption. Both are retained: `idle_funds_ratio` measures  
> absolute slack; `fund_util_peer_gap` measures it relative to sector peers.

### Notable baseline correlations addressed by dropping features:
- `funds_released_ratio` ↔ `time_elapsed`: r = **0.846** (severe collinearity — dropped both)
- `spend_vs_progress` ↔ `funds_released_ratio`: r = **−0.666** (lifecycle conflation)

---

## 4. Anomaly Detection Comparison

Both detectors run `contamination=0.12` on the same 260 projects, yielding exactly **32** flagged anomalies each (⌈0.12 × 260⌉ = 32).

| Group | Count |
|---|---|
| Flagged by **both** detectors | **29** |
| Flagged **only by baseline** | 3 |
| Flagged **only by improved** | 3 |

The 3 projects uniquely identified by the improved detector were missed by the baseline because:
- Their cost overruns were within portfolio norms but well above **sector** norms (`cost_overrun_peer_gap` fired).
- They showed high `idle_funds_ratio` — treasury funds released but not absorbed — a signal invisible to the baseline `spend_vs_progress`.

The 3 projects dropped by the improved detector had anomaly flags driven by high `funds_released_ratio` combined with sector-normal `fund_utilisation` — a lifecycle artefact of early-stage projects that received large first tranches, not genuine fiscal irregularity.

---

## 5. Limitations

> [!WARNING]  
> Results are derived from a **synthetic** 260-project dataset. The sector peer groups  
> are small (~52 projects each). In a real deployment, peer medians should be computed  
> on a rolling window of at least 500 same-sector projects to be statistically robust.

- **`idle_funds_ratio`** and **`fund_util_peer_gap`** are highly correlated (r = −0.859) on  
  this dataset; in production these may need dimensionality treatment.
- **`cost_overrun_peer_gap`** leaks information from the full dataset when computing sector  
  medians on the training set itself. In a batch-prediction setting, sector medians should  
  be fitted on a held-out reference period.
- The Isolation Forest contamination parameter (0.12) is a design assumption, not a  
  data-driven estimate. A calibration study with labelled audit outcomes would improve it.

---

## 6. Reproducibility

```bash
# Install dependencies
pip install -r requirements.txt

# Run all 33 tests (3 original + 30 new)
python -m pytest tests/ -v

# Launch the Streamlit app
streamlit run app.py
```

All tests pass on Python 3.13.5, scikit-learn 1.x, pandas 2.x.
