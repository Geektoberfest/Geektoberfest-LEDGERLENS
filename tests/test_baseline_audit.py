import pandas as pd

from baseline_audit import baseline_summary, evaluate_labels, sector_breakdown, seed_stability
from model import run_anomaly_detection


def test_baseline_summary_matches_expected_counts():
    data = run_anomaly_detection()["data"]
    summary = baseline_summary(data)
    assert summary["total"] == 260
    assert summary["flagged"] == 32
    assert abs(summary["rate"] - 32 / 260) < 1e-9


def test_sector_breakdown_adds_up():
    data = run_anomaly_detection()["data"]
    table = sector_breakdown(data)
    assert table["projects"].sum() == 260
    assert table["flagged"].sum() == 32


def test_seed_stability_is_high_but_not_guaranteed():
    stab = seed_stability()
    assert 0.9 <= stab["mean_jaccard"] <= 1.0
    assert stab["always"] <= 32 <= stab["ever"]


def test_label_evaluation_runs_after_fitting(tmp_path):
    data = run_anomaly_detection()["data"]
    flagged_ids = data.loc[data["anomaly"] == -1, "id"].head(5)
    labels = pd.DataFrame({"id": data["id"], "label": data["id"].isin(flagged_ids).astype(int)})
    path = tmp_path / "labels.csv"
    labels.to_csv(path, index=False)
    ev = evaluate_labels(data, path)
    assert ev["tp"] == 5 and ev["fn"] == 0 and ev["labelled"] == 260
