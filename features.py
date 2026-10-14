from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATASET_PATH=Path(__file__).resolve().parent/"sample_projects.csv"
BASELINE_FEATURES=["cost_overrun","fund_utilisation","funds_released_ratio","time_elapsed","progress_vs_time","spend_vs_progress"]
RATIO_FEATURES=["schedule_performance","cost_performance","spend_pace"]
PEER_FEATURES=["peer_cost_overrun_z","peer_spend_gap_z"]
FEATURE_SETS={
    "baseline":BASELINE_FEATURES,
    "ratios":BASELINE_FEATURES+RATIO_FEATURES,
    "ratios+peers":BASELINE_FEATURES+RATIO_FEATURES+PEER_FEATURES,
}
RAW_COLUMNS=["project_id","project_name","sector","district","estimated_cost","actual_cost","funds_released","funds_utilised","planned_days","elapsed_days","progress"]
RATIO_CAP=3
WINSOR_LIMITS=(.01,.99)

def read_raw(path=DATASET_PATH):
    return pd.read_csv(path)[RAW_COLUMNS]

def clean(raw):
    # Explicit data-quality handling. Every correction is counted in the report
    # so the evaluation can state exactly what was changed.
    data=raw.copy()
    report={"rows":len(data),"missing_values":int(data.isna().sum().sum())}
    numeric=["estimated_cost","actual_cost","funds_released","funds_utilised","planned_days","elapsed_days","progress"]
    data[numeric]=data[numeric].apply(pd.to_numeric,errors="coerce")
    if data["progress"].max()>1.5:
        data["progress"]=data["progress"]/100
        report["progress_rescaled_from_percent"]=True
    else:
        report["progress_rescaled_from_percent"]=False
    out_of_range=int(((data["progress"]<0)|(data["progress"]>1)).sum())
    data["progress"]=data["progress"].clip(0,1)
    report["progress_clipped"]=out_of_range
    over_released=int((data["funds_utilised"]>data["funds_released"]).sum())
    data["funds_utilised"]=np.minimum(data["funds_utilised"],data["funds_released"])
    report["utilised_above_released_clipped"]=over_released
    report["elapsed_beyond_plan"]=int((data["elapsed_days"]>data["planned_days"]).sum())
    for col in ["estimated_cost","actual_cost","funds_released","planned_days"]:
        bad=data[col]<=0
        report[f"non_positive_{col}"]=int(bad.sum())
        data.loc[bad,col]=np.nan
    return data,report

def winsorize(frame,limits=WINSOR_LIMITS):
    low=frame.quantile(limits[0])
    high=frame.quantile(limits[1])
    return frame.clip(lower=low,upper=high,axis=1)

def peer_z(data,column):
    # Robust z-score against sector peers (median and MAD). Includes the project itself;
    # sectors have 43-62 projects, so the self-inclusion effect is small.
    group=data.groupby("sector")[column]
    median=group.transform("median")
    spread=1.4826*(data[column]-median).abs().groupby(data["sector"]).transform("median")
    fallback=group.transform("std")
    spread=spread.where(spread>0,fallback)
    return ((data[column]-median)/spread.replace(0,np.nan)).fillna(0)

def engineer(data):
    # Baseline formulas match model.load_data. Missing or zero denominators are set to
    # a neutral value (0 for deviations, 1 for performance ratios) and never silently
    # filled with the value of another project.
    out=pd.DataFrame({"id":data["project_id"],"sector":data["sector"]})
    cost_overrun=(data["actual_cost"]-data["estimated_cost"])/data["estimated_cost"]
    out["cost_overrun"]=cost_overrun.fillna(0)
    out["fund_utilisation"]=(data["funds_utilised"]/data["funds_released"]).fillna(0).clip(0,1)
    out["funds_released_ratio"]=(data["funds_released"]/data["estimated_cost"]).fillna(0)
    time_elapsed=(data["elapsed_days"]/data["planned_days"]).fillna(0)
    out["time_elapsed"]=time_elapsed
    out["progress_vs_time"]=data["progress"]-time_elapsed.clip(upper=1)
    out["spend_vs_progress"]=out["fund_utilisation"]-data["progress"]
    schedule=(data["progress"]/time_elapsed).where(time_elapsed>0)
    out["schedule_performance"]=schedule.fillna(1).clip(0,RATIO_CAP)
    earned=data["progress"]*data["estimated_cost"]
    cost_perf=(earned/data["actual_cost"]).where(data["actual_cost"]>0)
    out["cost_performance"]=cost_perf.fillna(1).clip(0,RATIO_CAP)
    pace=(out["fund_utilisation"]/time_elapsed).where(time_elapsed>0)
    out["spend_pace"]=pace.fillna(1).clip(0,RATIO_CAP)
    out["peer_cost_overrun_z"]=peer_z(out,"cost_overrun")
    out["peer_spend_gap_z"]=peer_z(out,"spend_vs_progress")
    return out

def build(raw=None):
    raw=read_raw() if raw is None else raw
    data,report=clean(raw)
    features=engineer(data)
    return features,report

def scaled_matrix(features,columns):
    values=winsorize(features[columns].astype(float))
    return StandardScaler().fit_transform(values)
