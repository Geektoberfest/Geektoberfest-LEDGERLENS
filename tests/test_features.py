import numpy as np
from features import read_raw, clean, engineer, build, winsorize, FEATURE_SETS, BASELINE_FEATURES
from model import load_data

def test_baseline_features_match_model_on_clean_data():
    features,_=build()
    model_data=load_data()
    for column in BASELINE_FEATURES:
        assert np.allclose(features[column].to_numpy(),model_data[column].to_numpy(),atol=1e-9)

def test_engineered_features_are_finite_and_complete():
    features,_=build()
    for columns in FEATURE_SETS.values():
        assert np.isfinite(features[columns].to_numpy()).all()

def test_percent_progress_is_rescaled_and_reported():
    raw=read_raw()
    raw["progress"]=raw["progress"]*100
    data,report=clean(raw)
    assert report["progress_rescaled_from_percent"] is True
    assert data["progress"].between(0,1).all()

def test_inconsistent_funds_are_clipped_and_counted():
    raw=read_raw()
    raw.loc[0,"funds_utilised"]=raw.loc[0,"funds_released"]*1.2
    data,report=clean(raw)
    assert report["utilised_above_released_clipped"]==1
    assert data.loc[0,"funds_utilised"]==data.loc[0,"funds_released"]

def test_non_positive_denominators_become_neutral_not_neighbour_values():
    raw=read_raw()
    raw.loc[0,"estimated_cost"]=0
    data,report=clean(raw)
    assert report["non_positive_estimated_cost"]==1
    features=engineer(data)
    assert features.loc[0,"cost_overrun"]==0
    assert features.loc[0,"funds_released_ratio"]==0

def test_zero_elapsed_time_gives_neutral_performance_ratio():
    raw=read_raw()
    raw.loc[0,"elapsed_days"]=0
    data,_=clean(raw)
    features=engineer(data)
    assert features.loc[0,"schedule_performance"]==1.0
    assert features.loc[0,"spend_pace"]==1.0

def test_winsorize_limits_extremes_to_percentiles():
    frame=read_raw()[["estimated_cost","actual_cost"]].astype(float)
    frame.loc[0,"actual_cost"]=1e12
    clipped=winsorize(frame)
    assert clipped["actual_cost"].max()<1e12
    assert clipped["actual_cost"].max()<=frame["actual_cost"].quantile(.99)+1e-6

def test_peer_z_is_centred_within_each_sector():
    features,_=build()
    for _,group in features.groupby("sector"):
        assert abs(group["peer_cost_overrun_z"].median())<1e-9
