from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DATASET_PATH=Path(__file__).resolve().parent/"sample_projects.csv"
FEATURES=["cost_overrun","fund_utilisation","funds_released_ratio","time_elapsed","progress_vs_time","spend_vs_progress"]
SIZE_BANDS=["Small","Medium","Large"]
MIN_PEERS=8
# concern: +1 if a high value is a warning sign, -1 if a low value is, 0 for context only
# min_gap: smallest difference from the peer median (in fraction units) worth explaining
SIGNALS={
    "cost_overrun":{"label":"Cost overrun (actual vs estimated cost)","kind":"share","concern":1,"min_gap":.10},
    "fund_utilisation":{"label":"Share of released funds spent","kind":"share","concern":1,"min_gap":.10},
    "funds_released_ratio":{"label":"Share of estimated cost released","kind":"share","concern":0,"min_gap":.15},
    "time_elapsed":{"label":"Share of planned time elapsed","kind":"share","concern":1,"min_gap":.15},
    "progress_vs_time":{"label":"Physical progress relative to time elapsed","kind":"gap","concern":-1,"min_gap":.15},
    "spend_vs_progress":{"label":"Fund use relative to physical progress","kind":"gap","concern":1,"min_gap":.15},
}
Z_THRESHOLD=2

def load_data():
    data=pd.read_csv(DATASET_PATH)
    rename={"project_id":"id","project_name":"name","estimated_cost":"est_cost","funds_utilised":"funds_used","planned_days":"planned_days","elapsed_days":"elapsed_days","progress":"progress_pct"}
    data=data.rename(columns=rename)
    numeric=["est_cost","actual_cost","funds_released","funds_used","planned_days","elapsed_days","progress_pct"]
    for col in numeric:
        data[col]=pd.to_numeric(data[col],errors="coerce").fillna(0)
    if data["progress_pct"].max(skipna=True)<=1.5:
        data["progress_pct"]=data["progress_pct"]*100
    data["progress_pct"]=data["progress_pct"].clip(0,100)
    data["cost_overrun"]=(data["actual_cost"]-data["est_cost"])/data["est_cost"].replace(0,np.nan)
    data["fund_utilisation"]=(data["funds_used"]/data["funds_released"].replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0).clip(0,1)
    data["funds_released_ratio"]=(data["funds_released"]/data["est_cost"].replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0)
    data["time_elapsed"]=(data["elapsed_days"]/data["planned_days"].replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0)
    data["progress_vs_time"]=data["progress_pct"]/100-data["time_elapsed"].clip(upper=1)
    data["spend_vs_progress"]=data["fund_utilisation"]-data["progress_pct"]/100
    data["size_band"]=data.groupby("sector")["est_cost"].transform(lambda s:pd.qcut(s.rank(method="first"),3,labels=SIZE_BANDS)).astype(str)
    return data.replace([np.inf,-np.inf],np.nan).fillna(0)

def run_anomaly_detection(contamination=.12):
    data=load_data()
    scaler=StandardScaler()
    X=scaler.fit_transform(data[FEATURES])
    model=IsolationForest(n_estimators=250,contamination=contamination,random_state=42)
    data["anomaly"]=model.fit_predict(X)
    data["anomaly_score"]=model.decision_function(X)
    data["anomaly_strength"]=(-data["anomaly_score"]).clip(lower=0)
    return {"data":data,"model":model,"scaler":scaler,"features":FEATURES}

def format_signal(key,value):
    if SIGNALS[key]["kind"]=="share":
        return f"{value:.0%}"
    return f"{value*100:.0f} points"

def robust_z(value,peer_values):
    median=peer_values.median()
    spread=1.4826*(peer_values-median).abs().median()
    if spread==0 or np.isnan(spread):
        spread=peer_values.std()
    if spread==0 or np.isnan(spread):
        return 0.0
    return (value-median)/spread

def peer_group(row,data):
    others=data[data["id"]!=row["id"]]
    sector_peers=others[others["sector"]==row["sector"]]
    band_peers=sector_peers[sector_peers["size_band"]==row["size_band"]]
    if len(band_peers)>=MIN_PEERS:
        return band_peers,f"{row['sector']} projects of {row['size_band'].lower()} size"
    return sector_peers,f"{row['sector']} projects"

def build_evidence(row,data):
    peers,label=peer_group(row,data)
    signals=[]
    for key,spec in SIGNALS.items():
        observed=row[key]
        peer_median=peers[key].median()
        gap=observed-peer_median
        z=robust_z(observed,peers[key])
        notable=spec["concern"]!=0 and np.sign(gap)==spec["concern"] and abs(z)>=Z_THRESHOLD and abs(gap)>=spec["min_gap"]
        signals.append({"key":key,"label":spec["label"],"observed":observed,"peer_median":peer_median,"gap":gap,"z":z,"notable":bool(notable)})
    return {"peer_label":label,"peer_count":len(peers),"signals":sorted(signals,key=lambda s:-abs(s["z"]))}

def explain_project(row,data):
    evidence=build_evidence(row,data)
    reasons=[]
    for signal in evidence["signals"]:
        if not signal["notable"]:
            continue
        key=signal["key"]
        group=f"{evidence['peer_label']} ({evidence['peer_count']} projects)"
        observed=format_signal(key,signal["observed"])
        peer_median=format_signal(key,signal["peer_median"])
        if SIGNALS[key]["kind"]=="share":
            direction="above" if signal["gap"]>0 else "below"
            reasons.append(f"{signal['label']} is {observed}, {abs(signal['gap'])*100:.0f} points {direction} the peer median of {peer_median} for {group}.")
        else:
            reasons.append(f"{signal['label']} is {observed}, against a peer median of {peer_median} for {group}.")
    if not reasons and row["anomaly"]==-1:
        reasons.append("No single signal stands out against comparable projects, but the combination of metrics is unusual relative to the overall portfolio.")
    return reasons
