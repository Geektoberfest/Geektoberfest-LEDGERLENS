from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from features import build, read_raw, FEATURE_SETS, scaled_matrix

CONTAMINATION=.12
SEEDS=range(10)
SUBSAMPLES=30
SUBSAMPLE_FRACTION=.8
PLANT_PROJECTS=20
PLANT_TRIALS=5
PATTERNS=["schedule slip","cost overrun","spend ahead of progress"]

def detect(features,columns,seed=42):
    X=scaled_matrix(features,columns)
    model=IsolationForest(n_estimators=250,contamination=CONTAMINATION,random_state=seed)
    return model.fit_predict(X)==-1

def jaccard(a,b):
    union=a|b
    return 1.0 if not union else len(a&b)/len(union)

def stability(features):
    ids=features["id"].to_numpy()
    rows={}
    for name,columns in FEATURE_SETS.items():
        reference=set(ids[detect(features,columns)])
        seed_sets=[set(ids[detect(features,columns,seed)]) for seed in SEEDS]
        seed_j=[jaccard(a,b) for a,b in combinations(seed_sets,2)]
        rng=np.random.default_rng(0)
        size=int(round(SUBSAMPLE_FRACTION*len(features)))
        sub_j=[]
        for _ in range(SUBSAMPLES):
            idx=rng.choice(len(features),size=size,replace=False)
            kept=set(ids[idx])
            sub=features.iloc[idx].reset_index(drop=True)
            flagged=set(ids[idx][detect(sub,columns)])
            sub_j.append(jaccard(reference&kept,flagged))
        rows[name]={"flagged":len(reference),"seed_jaccard":np.mean(seed_j),"subsample_jaccard":np.mean(sub_j),"subsample_std":np.std(sub_j)}
    return pd.DataFrame(rows).T

def plant(raw,pattern,ids):
    raw=raw.copy()
    mask=raw["project_id"].isin(ids)
    if pattern=="schedule slip":
        raw.loc[mask,"elapsed_days"]=(raw.loc[mask,"planned_days"]*1.6).round()
    elif pattern=="cost overrun":
        raw.loc[mask,"actual_cost"]=raw.loc[mask,"estimated_cost"]*1.5
    elif pattern=="spend ahead of progress":
        raw.loc[mask,"progress"]=raw.loc[mask,"progress"]*.5
        raw.loc[mask,"funds_utilised"]=raw.loc[mask,"funds_released"]*.95
    return raw

def planted_benchmark(raw):
    rows=[]
    for pattern in PATTERNS:
        for trial in range(PLANT_TRIALS):
            rng=np.random.default_rng(100+trial)
            ids=rng.choice(raw["project_id"].to_numpy(),size=PLANT_PROJECTS,replace=False)
            features,_=build(plant(raw,pattern,ids))
            is_planted=features["id"].isin(ids).to_numpy()
            for name,columns in FEATURE_SETS.items():
                flagged=detect(features,columns,seed=trial)
                rows.append({"pattern":pattern,"trial":trial,"set":name,"recall":flagged[is_planted].mean(),"unplanted_flag_rate":flagged[~is_planted].mean()})
    frame=pd.DataFrame(rows)
    return frame.groupby(["pattern","set"],sort=False)[["recall","unplanted_flag_rate"]].mean()

def changed_examples(features,limit=5):
    base=detect(features,FEATURE_SETS["baseline"])
    full=detect(features,FEATURE_SETS["ratios+peers"])
    cols=["id","sector","cost_overrun","time_elapsed","progress_vs_time","cost_performance","schedule_performance","peer_cost_overrun_z"]
    only_new=features[full&~base][cols]
    only_base=features[base&~full][cols]
    return only_new.head(limit),only_base.head(limit)

def main():
    raw=read_raw()
    features,report=build(raw)
    pd.set_option("display.width",200)
    print("## Data quality report")
    for key,value in report.items(): print(f"- {key}: {value}")
    print("\n## Stability on the real data")
    print(stability(features).to_string(float_format=lambda x:f"{x:.3f}"))
    print("\n## Planted-anomaly recall (20 planted projects per trial, 5 trials)")
    print(planted_benchmark(raw).to_string(float_format=lambda x:f"{x:.3f}"))
    new_only,base_only=changed_examples(features)
    print("\n## Flagged only by ratios+peers")
    print(new_only.to_string(index=False,float_format=lambda x:f"{x:.2f}"))
    print("\n## Flagged only by baseline")
    print(base_only.to_string(index=False,float_format=lambda x:f"{x:.2f}"))

if __name__=="__main__":
    main()
