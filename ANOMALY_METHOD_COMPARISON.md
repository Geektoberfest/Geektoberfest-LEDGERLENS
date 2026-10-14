# Comparing Anomaly-Detection Methods (Issue #4)

**Question:** Do alternative anomaly-detection methods produce more useful and stable review queues than the Isolation Forest baseline?

**Short answer:** No. Isolation Forest is the most stable of the three methods, flags a comparable number of projects, and its flags are the easiest to explain. We recommend keeping it as the baseline. Local Outlier Factor (LOF) and One-Class SVM (OCSVM) disagree with it on a meaningful share of projects, and their unique flags are less stable and less explainable with the current rules.

Reproduce with:

```bash
python compare_methods.py
python -m pytest tests/test_compare_methods.py
```

## Data and preprocessing

- Data: `sample_projects.csv`, the 260-project **synthetic** starter dataset. It does not represent official government findings.
- Features: the six baseline features from `model.py` (`FEATURES`): cost overrun, fund utilisation, funds-released ratio, time elapsed, progress vs time, spend vs progress.
- Preprocessing: `StandardScaler` on the six features, identical for all methods. The scaler is fit on the full dataset, as in the baseline. It is unsupervised, so no labels leak into the comparison.

## Methods and parameters

| Method | Parameters | Notes |
|---|---|---|
| Isolation Forest (baseline) | `n_estimators=250`, `contamination=0.12`, `random_state=42` | Same as `model.py` |
| Local Outlier Factor | `n_neighbors=20`, `contamination=0.12` | Density-based, transductive (`fit_predict`) |
| One-Class SVM | `kernel="rbf"`, `nu=0.12`, `gamma="scale"` | `nu` is set to the same 0.12 contamination rate |

`contamination` and `nu` both fix the flagged fraction at about 12%, so the methods are compared at the same review-queue size. The LOF and OCSVM settings are sklearn defaults with a single neighbourhood size and kernel; they were not tuned.

## Metrics

No ground-truth labels exist, so no accuracy, precision, or recall is reported. The comparison uses only label-free measures:

1. **Anomaly rate:** how many projects each method flags.
2. **Overlap:** shared flags and Jaccard similarity between each pair of methods.
3. **Score agreement:** Spearman rank correlation of each method's continuous score.
4. **Seed stability:** Isolation Forest refit with 10 random seeds (0–9); mean pairwise Jaccard of flagged sets.
5. **Subsample stability:** 30 random subsamples of 80% of projects (sampling seed 0). For each run, the full pipeline (scaler and model) is refit on the subsample. The subsample's flags are compared with the full-data flags restricted to the same projects, using Jaccard.
6. **Representative cases:** projects flagged by only one method, explained with the existing `explain_project` rules.

## Results

### Anomaly rate

| Method | Flagged | Rate |
|---|---|---|
| Isolation Forest | 32 | 12.3% |
| Local Outlier Factor | 32 | 12.3% |
| One-Class SVM | 34 | 13.1% |

### Overlap of flagged sets

| Pair | Shared | Jaccard |
|---|---|---|
| Isolation Forest vs LOF | 25 | 0.64 |
| Isolation Forest vs OCSVM | 24 | 0.57 |
| LOF vs OCSVM | 22 | 0.50 |

19 projects are flagged by all three methods and 214 by none.

### Score agreement (Spearman)

| | Isolation Forest | LOF | OCSVM |
|---|---|---|---|
| Isolation Forest | 1.00 | 0.69 | 0.57 |
| LOF | 0.69 | 1.00 | 0.70 |
| OCSVM | 0.57 | 0.70 | 1.00 |

### Stability

| Method | Stability measure | Mean Jaccard | Std |
|---|---|---|---|
| Isolation Forest | 10 seeds (pairwise) | 0.98 | – |
| Isolation Forest | 30 subsamples | 0.93 | 0.04 |
| Local Outlier Factor | 30 subsamples | 0.70 | 0.05 |
| One-Class SVM | 30 subsamples | 0.71 | 0.06 |

Isolation Forest is the most stable on both measures. LOF and OCSVM are deterministic on a fixed dataset, so seed stability does not apply to them. Their subsample stability is about 0.70: when 20% of projects are removed and the models refit, the flagged set typically shifts by a substantial share (on average about 30% of the union changes).

## Representative differences

Feature values are the unscaled engineered features from `load_data()`, rounded to two decimals. Progress is implied by `progress_vs_time + time_elapsed`. The rule explanations come from `explain_project`.

**Flagged only by Isolation Forest (2 projects)**

- **PRJ-2030, Treatment plant #31 (Water):** fund utilisation 0.97, with 98% of the planned time elapsed but only about 19% physical progress (progress vs time −0.79). Rule hits: spending ahead of progress; progress behind the timeline.
- **PRJ-2064, Trauma unit #65 (Health):** cost overrun 0.15, fund utilisation 0.97, 82% of the planned time elapsed with about 18% progress (progress vs time −0.64). Same two rule hits.

These are the clearest cases: near-complete timelines with low physical progress and high spending. The baseline flags them and the rules explain them, which is the behaviour the project is built around.

**Flagged only by LOF (4 projects; 3 shown by the script)**

- **PRJ-2040, Highway widening #41 (Roads)** and **PRJ-2152, Bridge repair #153 (Roads)**: elapsed time exceeds the planned time (1.23 and 1.07), yet spend-vs-progress is near zero. Neither triggers a rule, so the only explanation is the generic "unusual relative to the portfolio" message.
- **PRJ-2168, Highway widening #169 (Roads):** no rule fires.

LOF finds real timeline overruns, but the current explanation rules cannot say why they were flagged.

**Flagged only by One-Class SVM (7 projects; 3 shown by the script)**

- **PRJ-2037, Treatment plant #38 (Water)**, **PRJ-2079, Rural link road #80 (Roads)**, and **PRJ-2112, Rural link road #113 (Roads)**: all show spending ahead of progress (spend-vs-progress 0.51–0.57) with low fund utilisation (0.81–0.98) and short elapsed times (0.26–0.32). These are early-stage projects where spending looks high relative to progress, but the timeline is far from over, so the signal is weaker than the baseline's cases.

## Recommendation

Keep Isolation Forest as the baseline. The evidence:

- It is the most stable method, both across seeds (0.98) and across subsamples (0.93 versus about 0.70 for the alternatives).
- It flags the same number of projects as LOF and about the same as OCSVM, so the review queue size is not a deciding factor.
- Its unique flags map onto the existing explanation rules. The alternatives' unique flags are harder to explain without new rules.
- The methods disagree substantially (Jaccard 0.50–0.64), but the disagreements are concentrated in the less explainable cases.

LOF and OCSVM are worth revisiting if the explanation layer (issue #3) gains timeline-overrun rules, or if the team tunes `n_neighbors` and `nu`. Tuning is outside the scope of this comparison.

## Assumptions and limitations

- **Synthetic data.** The dataset is synthetic. Stability and overlap say nothing about whether the flagged projects are actually problematic in real public development projects.
- **Stability is not validity.** A method can be stable and still wrong. Without labels, we cannot tell whether any method is correct.
- **Untuned alternatives.** LOF and OCSVM use a single set of default-style parameters. A different neighbourhood size or kernel could change the rankings. The conclusion holds for these settings only.
- **Fixed contamination.** All methods flag about 12% of projects by construction, so rate comparisons are not evidence of detection quality.
- **Subsample caveats.** Subsamples are drawn once with a fixed seed (0), and the Jaccard scores are computed only on projects in both the subsample and the full-data run.
- **Flags are candidates for review.** An anomaly is not proof of fraud or corruption. LEDGERLENS is a screening prototype for prioritising projects for human review, and this comparison does not change that.
