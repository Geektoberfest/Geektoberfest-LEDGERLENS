from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DATASET_PATH=Path(__file__).resolve().parent/"sample_projects.csv"
FEATURES=["cost_overrun","fund_utilisation","funds_released_ratio","time_elapsed","progress_vs_time","spend_vs_progress"]

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

def explain_project(row,data):
    peers=data[data["sector"]==row["sector"]]
    reasons=[]
    peer_cost=peers["cost_overrun"].median()
    peer_util=peers["fund_utilisation"].median()
    peer_gap=peers["spend_vs_progress"].median()
    if row["cost_overrun"]>peer_cost+.15:
        reasons.append(f"Cost overrun is {row['cost_overrun']:.0%}, above the sector peer median of {peer_cost:.0%}.")
    if row["fund_utilisation"]>peer_util+.15:
        reasons.append(f"Fund utilisation is {row['fund_utilisation']:.0%}, above the sector peer median of {peer_util:.0%}.")
    if row["spend_vs_progress"]>peer_gap+.20:
        reasons.append("Spending is running ahead of reported physical progress.")
    if row["progress_vs_time"]<-.20:
        reasons.append("Physical progress is materially behind the elapsed timeline.")
    if not reasons and row["anomaly"]==-1:
        reasons.append("The combination of project metrics is unusual relative to the overall portfolio.")
    return reasons
