"""Tests for improved feature engineering (Issue #2).

Validates that the revised indicators are correctly computed, that the improved
Isolation Forest run is deterministic, and that explain_project produces
richer output under the improved feature_set.
"""
import numpy as np
import pandas as pd
import pytest

from model import (
    BASELINE_FEATURES,
    IMPROVED_FEATURES,
    load_data,
    run_anomaly_detection,
    explain_project,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data():
    """Shared loaded dataset (sanitized, all features engineered)."""
    return load_data()


@pytest.fixture(scope="module")
def baseline_result():
    return run_anomaly_detection(feature_set="baseline")


@pytest.fixture(scope="module")
def improved_result():
    return run_anomaly_detection(feature_set="improved")


# ---------------------------------------------------------------------------
# 1. Feature presence and types
# ---------------------------------------------------------------------------

class TestFeaturePresence:
    """All engineered columns must exist and be numeric after load_data."""

    BASELINE_COLS = [
        "cost_overrun", "fund_utilisation", "funds_released_ratio",
        "time_elapsed", "progress_vs_time", "spend_vs_progress",
    ]
    IMPROVED_COLS = [
        "budget_spent_ratio", "budget_burn_vs_progress",
        "schedule_slippage", "schedule_overrun",
        "idle_funds_ratio", "fund_release_lead",
        "cost_overrun_peer_gap", "fund_util_peer_gap",
    ]

    def test_baseline_columns_present(self, data):
        missing = [c for c in self.BASELINE_COLS if c not in data.columns]
        assert missing == [], f"Missing baseline columns: {missing}"

    def test_improved_columns_present(self, data):
        missing = [c for c in self.IMPROVED_COLS if c not in data.columns]
        assert missing == [], f"Missing improved columns: {missing}"

    def test_all_feature_cols_are_numeric(self, data):
        all_cols = self.BASELINE_COLS + self.IMPROVED_COLS
        for col in all_cols:
            assert pd.api.types.is_numeric_dtype(data[col]), \
                f"Column '{col}' is not numeric (dtype={data[col].dtype})"

    def test_no_nulls_or_inf_in_features(self, data):
        all_cols = self.BASELINE_COLS + self.IMPROVED_COLS
        subset = data[all_cols]
        assert subset.isna().sum().sum() == 0, "NaN found in engineered features"
        assert not np.isinf(subset.values).any(), "Inf found in engineered features"


# ---------------------------------------------------------------------------
# 2. Feature value range / domain correctness
# ---------------------------------------------------------------------------

class TestFeatureDomains:
    """Improved indicators must stay within their designed bounds."""

    def test_idle_funds_ratio_non_negative(self, data):
        assert (data["idle_funds_ratio"] >= 0).all(), \
            "idle_funds_ratio should be >= 0 (winsorized)"

    def test_schedule_overrun_non_negative(self, data):
        assert (data["schedule_overrun"] >= 0).all(), \
            "schedule_overrun is clipped to lower=0"

    def test_cost_overrun_winsorized(self, data):
        assert (data["cost_overrun"] >= -1.0).all()
        assert (data["cost_overrun"] <= 10.0).all()

    def test_budget_burn_vs_progress_winsorized(self, data):
        assert (data["budget_burn_vs_progress"] >= -2.0).all()
        assert (data["budget_burn_vs_progress"] <= 2.0).all()

    def test_schedule_slippage_winsorized(self, data):
        assert (data["schedule_slippage"] >= -10.0).all()
        assert (data["schedule_slippage"] <= 2.0).all()

    def test_fund_utilisation_clipped_0_to_1(self, data):
        assert (data["fund_utilisation"] >= 0.0).all()
        assert (data["fund_utilisation"] <= 1.0).all()

    def test_progress_pct_in_range(self, data):
        assert (data["progress_pct"] >= 0).all()
        assert (data["progress_pct"] <= 100).all()


# ---------------------------------------------------------------------------
# 3. Peer-gap indicators are zero-mean within each sector
# ---------------------------------------------------------------------------

class TestPeerNormalization:
    """cost_overrun_peer_gap and fund_util_peer_gap should have near-zero
    mean within each sector (by construction from sector medians)."""

    def test_cost_overrun_peer_gap_sector_median_near_zero(self, data):
        sector_means = data.groupby("sector")["cost_overrun_peer_gap"].median()
        for sector, med in sector_means.items():
            assert abs(med) < 0.05, \
                f"Sector '{sector}' cost_overrun_peer_gap median = {med:.4f} (expected ≈ 0)"

    def test_fund_util_peer_gap_sector_median_near_zero(self, data):
        sector_means = data.groupby("sector")["fund_util_peer_gap"].median()
        for sector, med in sector_means.items():
            assert abs(med) < 0.05, \
                f"Sector '{sector}' fund_util_peer_gap median = {med:.4f} (expected ≈ 0)"

    def test_all_sectors_represented_in_peer_gap(self, data):
        expected_sectors = {"Roads", "Water", "Schools", "Health", "Power"}
        assert set(data["sector"].unique()) == expected_sectors


# ---------------------------------------------------------------------------
# 4. Budget burn vs progress: relationship check
# ---------------------------------------------------------------------------

class TestBudgetBurnIndicator:
    """budget_burn_vs_progress = budget_spent_ratio - progress_pct/100."""

    def test_budget_burn_formula_consistency(self, data):
        expected = data["budget_spent_ratio"] - data["progress_pct"] / 100.0
        # Allow for winsorization effect (may differ at extremes)
        delta = (data["budget_burn_vs_progress"] - expected.clip(-2.0, 2.0)).abs()
        assert (delta < 1e-9).all(), "budget_burn_vs_progress formula mismatch"

    def test_budget_spent_ratio_non_negative(self, data):
        assert (data["budget_spent_ratio"] >= 0).all()


# ---------------------------------------------------------------------------
# 5. Unclipped schedule slippage vs baseline progress_vs_time
# ---------------------------------------------------------------------------

class TestUnclippedScheduleSlippage:
    """schedule_slippage must differ from baseline progress_vs_time for
    projects where time_elapsed > 1.0 (i.e., already past deadline)."""

    def test_overdue_projects_differ_from_baseline(self, data):
        overdue = data[data["time_elapsed"] > 1.01]
        if len(overdue) == 0:
            pytest.skip("No overdue projects found in dataset")
        # baseline clips time_elapsed at 1.0 so progress_vs_time saturates
        # improved schedule_slippage keeps going negative
        worse = (overdue["schedule_slippage"] < overdue["progress_vs_time"]).sum()
        assert worse > 0, \
            "Unclipped schedule_slippage should be more negative for overdue projects"


# ---------------------------------------------------------------------------
# 6. Isolation Forest: feature_set selection and determinism
# ---------------------------------------------------------------------------

class TestAnomalyDetectionFeatureSets:

    def test_baseline_result_uses_baseline_features(self, baseline_result):
        assert set(baseline_result["features"]) == set(BASELINE_FEATURES)
        assert baseline_result["feature_set"] == "baseline"

    def test_improved_result_uses_improved_features(self, improved_result):
        assert set(improved_result["features"]) == set(IMPROVED_FEATURES)
        assert improved_result["feature_set"] == "improved"

    def test_both_flag_exactly_32_anomalies(self, baseline_result, improved_result):
        n_base = int((baseline_result["data"]["anomaly"] == -1).sum())
        n_impr = int((improved_result["data"]["anomaly"] == -1).sum())
        # contamination=0.12 × 260 = 31.2 → rounds to 32 regardless of features
        assert n_base == 32, f"Baseline flagged {n_base}, expected 32"
        assert n_impr == 32, f"Improved flagged {n_impr}, expected 32"

    def test_improved_and_baseline_flag_different_projects(self, baseline_result, improved_result):
        base_ids = set(
            baseline_result["data"][baseline_result["data"]["anomaly"] == -1]["id"]
        )
        impr_ids = set(
            improved_result["data"][improved_result["data"]["anomaly"] == -1]["id"]
        )
        # The sets must overlap but should NOT be identical – richer features
        # reprioritise which projects are most anomalous
        assert base_ids != impr_ids, \
            "Improved features should flag at least one different project than baseline"

    def test_custom_feature_list_accepted(self):
        custom = ["cost_overrun", "schedule_slippage", "idle_funds_ratio"]
        result = run_anomaly_detection(feature_set=custom)
        assert result["features"] == custom
        assert result["feature_set"] == "custom"

    def test_result_is_deterministic(self):
        r1 = run_anomaly_detection(feature_set="improved", random_state=0)
        r2 = run_anomaly_detection(feature_set="improved", random_state=0)
        assert (r1["data"]["anomaly"] == r2["data"]["anomaly"]).all()

    def test_result_contains_required_keys(self, improved_result):
        assert {"data", "model", "scaler", "features", "feature_set"}.issubset(
            improved_result.keys()
        )

    def test_anomaly_strength_non_negative(self, improved_result):
        assert (improved_result["data"]["anomaly_strength"] >= 0).all()


# ---------------------------------------------------------------------------
# 7. explain_project: improved path produces richer context
# ---------------------------------------------------------------------------

class TestExplainProject:

    def test_returns_list_for_normal_project(self, improved_result):
        data = improved_result["data"]
        normal = data[data["anomaly"] == 1].iloc[0]
        result = explain_project(normal, data, feature_set="improved")
        assert isinstance(result, list)

    def test_flagged_project_has_at_least_one_reason_or_fallback(self, improved_result):
        data = improved_result["data"]
        flagged = data[data["anomaly"] == -1]
        for _, row in flagged.iterrows():
            reasons = explain_project(row, data, feature_set="improved")
            assert len(reasons) >= 1, \
                f"Project {row['id']} flagged but explain_project returned no reasons"

    def test_baseline_explain_returns_list(self, baseline_result):
        data = baseline_result["data"]
        row = data.iloc[0]
        result = explain_project(row, data, feature_set="baseline")
        assert isinstance(result, list)

    def test_explain_high_idle_funds(self):
        """Synthetic row with very high idle funds must trigger the idle-funds rule."""
        data = load_data()
        # Build a synthetic row that clearly has high idle funds
        row = data.iloc[0].copy()
        row["idle_funds_ratio"] = 0.50
        row["cost_overrun_peer_gap"] = 0.00
        row["budget_burn_vs_progress"] = 0.00
        row["schedule_slippage"] = 0.00
        row["fund_util_peer_gap"] = 0.00
        row["anomaly"] = -1
        reasons = explain_project(row, data, feature_set="improved")
        idle_reasons = [r for r in reasons if "idle" in r.lower()]
        assert len(idle_reasons) >= 1, \
            f"Expected idle-funds reason, got: {reasons}"

    def test_explain_cost_overrun_peer_gap(self):
        """Synthetic row with large cost_overrun_peer_gap must trigger cost rule."""
        data = load_data()
        row = data.iloc[0].copy()
        row["cost_overrun_peer_gap"] = 0.50
        row["cost_overrun"] = 0.60
        row["budget_burn_vs_progress"] = 0.00
        row["schedule_slippage"] = 0.00
        row["idle_funds_ratio"] = 0.00
        row["fund_util_peer_gap"] = 0.00
        row["anomaly"] = -1
        reasons = explain_project(row, data, feature_set="improved")
        cost_reasons = [r for r in reasons if "cost overrun" in r.lower()]
        assert len(cost_reasons) >= 1, \
            f"Expected cost-overrun reason, got: {reasons}"
