"""Baseline audit of the LEDGERLENS Isolation Forest (issue #1).

Usage:
    python baseline_audit.py                      # writes docs/baseline_report.md
    python baseline_audit.py --labels labels.csv  # optional: evaluate against labels

The audit only *reads* the existing detector in model.py; it does not change it.
Ground-truth labels (if any) are loaded AFTER the model is fitted and are never
used for fitting, scaling or parameter choice.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from model import FEATURES, load_data, run_anomaly_detection

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
SEEDS = range(10)
CONTAMINATIONS = (0.05, 0.08, 0.12, 0.15)


def md_table(df, floatfmt="{:.3f}"):
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) and not float(v).is_integer() else
                 str(int(v)) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def baseline_summary(data):
    flagged = int((data["anomaly"] == -1).sum())
    return {"total": len(data), "flagged": flagged, "rate": flagged / len(data)}


def sector_breakdown(data):
    g = data.groupby("sector")["anomaly"].agg(projects="size", flagged=lambda s: int((s == -1).sum()))
    g["rate"] = g["flagged"] / g["projects"]
    return g.reset_index().sort_values("rate", ascending=False)


def feature_comparison(data):
    flagged = data[data["anomaly"] == -1]
    normal = data[data["anomaly"] == 1]
    out = pd.DataFrame({
        "feature": FEATURES,
        "flagged_median": [flagged[f].median() for f in FEATURES],
        "unflagged_median": [normal[f].median() for f in FEATURES],
    })
    out["difference"] = out["flagged_median"] - out["unflagged_median"]
    return out


def top_drivers(data, scaler, n=2):
    """Features with the largest |z-score| per project.

    Isolation Forest has no built-in attribution, so this is a descriptive
    proxy: which standardized features sit furthest from the portfolio mean.
    It shows what is unusual, not what the forest 'used'.
    """
    z = pd.DataFrame(scaler.transform(data[FEATURES]), columns=FEATURES, index=data.index)
    drivers = z.abs().apply(lambda r: ", ".join(f"{f} ({z.loc[r.name, f]:+.1f}σ)" for f in r.nlargest(n).index), axis=1)
    return drivers


def primary_driver_counts(data, scaler):
    """For flagged projects, which standardized feature is furthest from the mean (and in which direction)."""
    flagged = data[data["anomaly"] == -1]
    z = pd.DataFrame(scaler.transform(flagged[FEATURES]), columns=FEATURES, index=flagged.index)
    top = z.abs().idxmax(axis=1)
    rows = [{"primary_feature": f, "direction": "high" if z.loc[i, f] > 0 else "low", "flagged_projects": 0}
            for i, f in top.items()]
    out = pd.DataFrame(rows).groupby(["primary_feature", "direction"], as_index=False).count()
    return out.sort_values("flagged_projects", ascending=False).reset_index(drop=True)


def representative_cases(data, drivers):
    cols = ["id", "sector", "anomaly_score"]
    ranked = data.sort_values("anomaly_score")
    flagged = ranked[ranked["anomaly"] == -1]
    normal = ranked[ranked["anomaly"] == 1]
    groups = {
        "Most anomalous (lowest score)": flagged.head(5),
        "Borderline flagged (just inside the 12% cut)": flagged.tail(3),
        "Borderline unflagged (just outside the cut)": normal.head(3),
        "Most typical (highest score)": normal.tail(3),
    }
    result = {}
    for title, frame in groups.items():
        t = frame[cols].copy()
        t["top_unusual_features"] = drivers.loc[frame.index]
        result[title] = t
    return result


def seed_stability(contamination=0.12):
    """Re-run the detector with several random seeds and compare flag sets."""
    base = set(run_anomaly_detection(contamination)["data"].query("anomaly == -1")["id"])
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    data = load_data()
    X = StandardScaler().fit_transform(data[FEATURES])
    counts, jaccards, all_flags = [], [], []
    for seed in SEEDS:
        pred = IsolationForest(n_estimators=250, contamination=contamination, random_state=seed).fit_predict(X)
        flags = set(data.loc[pred == -1, "id"])
        counts.append(len(flags))
        jaccards.append(len(flags & base) / len(flags | base))
        all_flags.append(flags)
    always = set.intersection(*all_flags)
    ever = set.union(*all_flags)
    return {"counts": counts, "mean_jaccard": float(np.mean(jaccards)), "min_jaccard": float(np.min(jaccards)),
            "always": len(always), "ever": len(ever), "base": base, "always_ids": always, "ever_ids": ever}


def contamination_sensitivity():
    base = set(run_anomaly_detection(0.12)["data"].query("anomaly == -1")["id"])
    rows = []
    for c in CONTAMINATIONS:
        flags = set(run_anomaly_detection(c)["data"].query("anomaly == -1")["id"])
        rows.append({"contamination": c, "flagged": len(flags), "overlap_with_baseline": len(flags & base)})
    return pd.DataFrame(rows)


def data_observations(data):
    return {
        "time_over_plan": int((data["time_elapsed"] > 1).sum()),
        "max_time_elapsed": float(data["time_elapsed"].max()),
        "cost_over_50pct": int((data["cost_overrun"] > 0.5).sum()),
        "complete": int((data["progress_pct"] >= 100).sum()),
        "spend_ahead_20": int((data["spend_vs_progress"] > 0.2).sum()),
        "spend_behind": int((data["spend_vs_progress"] < -0.2).sum()),
    }


def evaluate_labels(data, labels_path):
    """Optional: evaluate flags against labels. Runs only AFTER fitting."""
    labels = pd.read_csv(labels_path)
    if not {"id", "label"}.issubset(labels.columns):
        raise SystemExit("labels file needs columns: id,label (1 = planted/known anomaly)")
    merged = data[["id", "anomaly"]].merge(labels[["id", "label"]], on="id", how="inner")
    tp = int(((merged["anomaly"] == -1) & (merged["label"] == 1)).sum())
    fp = int(((merged["anomaly"] == -1) & (merged["label"] != 1)).sum())
    fn = int(((merged["anomaly"] == 1) & (merged["label"] == 1)).sum())
    return {"labelled": len(merged), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan")}


def build_report(labels_path=None):
    result = run_anomaly_detection()
    data, scaler = result["data"], result["scaler"]
    summary = baseline_summary(data)
    drivers = top_drivers(data, scaler)
    cases = representative_cases(data, drivers)
    stab = seed_stability()
    sens = contamination_sensitivity()
    obs = data_observations(data)
    has_labels = any(c.lower() in ("label", "is_anomaly", "planted", "ground_truth", "is_planted_anomaly")
                     for c in pd.read_csv(ROOT / "sample_projects.csv", nrows=1).columns)

    out = []
    out.append("# LEDGERLENS baseline audit (Issue #1)\n")
    out.append("> LEDGERLENS runs on a **synthetic** 260-project sample, not official government records. "
               "An anomaly is a signal for human review, **not** proof of fraud, corruption or misuse.\n")
    out.append("Regenerate this report with `python baseline_audit.py` (deterministic; no timestamps).\n")

    out.append("## 1. Baseline configuration\n")
    out.append("- Model: `IsolationForest(n_estimators=250, contamination=0.12, random_state=42)`")
    out.append("- Preprocessing: `StandardScaler` fitted on the 6 engineered features (no other preprocessing)")
    out.append("- Features: " + ", ".join(f"`{f}`" for f in FEATURES))
    out.append("- Detector code is unchanged by this audit.\n")

    out.append("## 2. Anomaly count and rate\n")
    out.append(f"- Projects scored: **{summary['total']}**")
    out.append(f"- Flagged as anomalies: **{summary['flagged']}** (**{summary['rate']:.1%}**)")
    out.append("- This rate is **set by the `contamination=0.12` parameter**, not discovered from the data: "
               "the forest is told to flag roughly the lowest-scoring 12%. "
               "It should not be read as an estimate of how many projects are truly problematic.\n")
    out.append("Flagged projects per sector:\n")
    out.append(md_table(sector_breakdown(data)) + "\n")

    out.append("## 3. What distinguishes flagged from unflagged projects\n")
    out.append("Median feature values (unscaled):\n")
    out.append(md_table(feature_comparison(data)) + "\n")
    comp = feature_comparison(data)
    comp["abs_gap"] = comp["difference"].abs()
    top2 = comp.nlargest(2, "abs_gap")["feature"].tolist()
    out.append(f"Largest median gaps between flagged and unflagged projects: `{top2[0]}` and `{top2[1]}`. "
               "Medians describe the typical flagged project; a few of the most extreme cases are driven by a different feature "
               "(see below).\n")
    out.append("Primary driver per flagged project (feature furthest from the portfolio mean, with direction):\n")
    out.append(md_table(primary_driver_counts(data, scaler)) + "\n")

    out.append("## 4. Representative cases\n")
    out.append("`top_unusual_features` lists the two standardized features furthest from the portfolio mean. "
               "Isolation Forest has no built-in attribution, so this is a descriptive proxy, not the model's reasoning.\n")
    for title, table in cases.items():
        out.append(f"**{title}**\n")
        out.append(md_table(table.reset_index(drop=True), "{:.3f}") + "\n")

    out.append("## 5. Reproducibility and stability\n")
    out.append(f"- With the fixed seed (`random_state=42`) results are identical on every run "
               f"(`tests/test_model.py` asserts exactly 32 flags).")
    lo, hi = min(stab["counts"]), max(stab["counts"])
    span = f"exactly {lo}" if lo == hi else f"{lo}-{hi}"
    out.append(f"- Re-running with {len(list(SEEDS))} different seeds flags {span} projects each time. "
               f"Mean Jaccard overlap with the baseline flag set is **{stab['mean_jaccard']:.2f}** (minimum {stab['min_jaccard']:.2f}).")
    out.append(f"- **{stab['always']}** projects are flagged under every seed; **{stab['ever']}** are flagged under at least one. "
               f"The remaining {stab['ever'] - stab['always']} are borderline and depend on the seed.\n")
    out.append("Sensitivity to the `contamination` setting (seed 42):\n")
    out.append(md_table(sens) + "\n")

    out.append("## 6. Data observations\n")
    out.append(f"- {obs['time_over_plan']} projects have run past their planned duration "
               f"(max elapsed/planned = {obs['max_time_elapsed']:.2f}); `progress_vs_time` clips this at 1.0, so overruns beyond the plan are not distinguished.")
    out.append(f"- `spend_vs_progress` > 0.2 (spending ahead of progress) for {obs['spend_ahead_20']} projects; < -0.2 (progress ahead of spending) for {obs['spend_behind']}. "
               "Spending ahead of progress is therefore common in this sample, so it is a weaker signal on its own than the extremes.")
    out.append(f"- {obs['cost_over_50pct']} projects have a cost overrun above 50%; {obs['complete']} projects report 100% progress.")
    out.append("- `load_data()` fills missing or non-numeric values with 0. The sample has no missing values, so this has no effect here, "
               "but on real data it could silently create misleading zeros.\n")

    out.append("## 7. Ground truth\n")
    if has_labels:
        out.append("The dataset contains a candidate label column; see the evaluation below.\n")
    else:
        out.append("`sample_projects.csv` has **no ground-truth or planted-anomaly marker** "
                   "(columns: project_id, project_name, sector, district, estimated_cost, actual_cost, funds_released, "
                   "funds_utilised, planned_days, elapsed_days, progress). "
                   "Planted anomalies therefore cannot be checked, and **no accuracy, precision or recall is reported**, "
                   "in line with CONTRIBUTING.md.\n")
    if labels_path:
        ev = evaluate_labels(data, labels_path)
        out.append("Optional label evaluation (labels loaded after fitting, never used for fitting):\n")
        out.append(f"- Labelled projects matched: {ev['labelled']}; TP={ev['tp']}, FP={ev['fp']}, FN={ev['fn']}; "
                   f"precision={ev['precision']:.2f}, recall={ev['recall']:.2f}\n")
    else:
        out.append("`python baseline_audit.py --labels labels.csv` (columns `id,label`) evaluates flags against labels "
                   "if they become available. Labels are read only after the model is fitted.\n")

    out.append("## 8. Limitations\n")
    out.append("- **No reliable labels:** without ground truth, anomaly detection cannot be scored objectively; "
               "'flagged' only means statistically unusual relative to this portfolio.")
    out.append("- **Detected outliers are not confirmed misuse:** an unusual project can be legitimate "
               "(for example a project with a justified scope change), and a problematic project can look ordinary.")
    out.append("- **Synthetic data:** the sample was generated to simulate project behaviour; findings do not describe any real government project or district.")
    out.append("- **Rate is a parameter:** the 12.3% flag rate follows from `contamination=0.12`.")
    out.append("- **Borderline cases are unstable:** projects near the cut-off change with the seed and contamination value, so the review queue should be read as a ranking, not a verdict.")
    out.append("- **Scaling and features are fixed:** the six engineered features and standardization shape what counts as unusual; "
               "changing them (issues #2 and #4) will change which projects are flagged.")
    out.append("- **Descriptive drivers only:** feature-level reasons in this report are z-score proxies, not model attributions.\n")
    return "\n".join(out), data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", help="optional CSV with columns id,label for post-hoc evaluation")
    args = parser.parse_args()
    report, data = build_report(args.labels)
    DOCS.mkdir(exist_ok=True)
    (DOCS / "baseline_report.md").write_text(report, encoding="utf-8")
    flagged = data[data["anomaly"] == -1].sort_values("anomaly_score")
    cols = ["id", "name", "sector", "district", "anomaly_score"] + FEATURES
    flagged[cols].round(4).to_csv(DOCS / "baseline_flagged_projects.csv", index=False)
    print(f"Wrote {DOCS / 'baseline_report.md'} and baseline_flagged_projects.csv ({len(flagged)} flagged)")


if __name__ == "__main__":
    main()
