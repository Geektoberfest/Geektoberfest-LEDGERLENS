import numpy as np
import pandas as pd
from model import load_data, run_anomaly_detection, build_evidence, explain_project, robust_z, SIGNALS, MIN_PEERS

def test_size_bands_cover_each_sector():
    data=load_data()
    counts=pd.crosstab(data["sector"],data["size_band"])
    assert set(data["size_band"])=={"Small","Medium","Large"}
    assert (counts.min(axis=1)>=MIN_PEERS).all()

def test_evidence_ties_observed_values_to_data():
    result=run_anomaly_detection()
    data=result["data"]
    row=data.iloc[0]
    evidence=build_evidence(row,data)
    assert len(evidence["signals"])==len(SIGNALS)
    for signal in evidence["signals"]:
        assert signal["observed"]==row[signal["key"]]
        assert signal["gap"]==signal["observed"]-signal["peer_median"]
    zs=[abs(s["z"]) for s in evidence["signals"]]
    assert zs==sorted(zs,reverse=True)

def test_peers_exclude_the_project_itself():
    data=load_data()
    row=data.iloc[0]
    evidence=build_evidence(row,data)
    assert evidence["peer_count"]<len(data)
    assert evidence["peer_count"]>=MIN_PEERS

def test_flagged_project_reasons_contain_numbers():
    result=run_anomaly_detection()
    data=result["data"]
    flagged=data[data["anomaly"]==-1]
    reasons=[r for _,row in flagged.iterrows() for r in explain_project(row,data)]
    assert reasons
    assert all("peer median" in r or "portfolio" in r for r in reasons)

def test_non_flagged_projects_only_explained_by_warning_signals():
    result=run_anomaly_detection()
    data=result["data"]
    for _,row in data[data["anomaly"]==1].iterrows():
        evidence=build_evidence(row,data)
        assert len(explain_project(row,data))==sum(s["notable"] for s in evidence["signals"])

def test_robust_z_is_zero_when_peers_have_no_spread():
    peers=pd.Series([5.0,5.0,5.0,5.0])
    assert robust_z(5.0,peers)==0.0
    assert robust_z(7.0,peers)==0.0
