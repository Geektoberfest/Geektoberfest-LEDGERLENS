from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from model import FEATURES, load_data, explain_project

CONTAMINATION=.12
METHODS=["Isolation Forest","Local Outlier Factor","One-Class SVM"]
SEEDS=range(10)
SUBSAMPLES=30
SUBSAMPLE_FRACTION=.8

def make_model(name,seed=42):
    if name=="Isolation Forest":
        return IsolationForest(n_estimators=250,contamination=CONTAMINATION,random_state=seed)
    if name=="Local Outlier Factor":
        return LocalOutlierFactor(n_neighbors=20,contamination=CONTAMINATION)
    return OneClassSVM(nu=CONTAMINATION,kernel="rbf",gamma="scale")

def fit_flags(name,X,seed=42):
    model=make_model(name,seed)
    if name=="Local Outlier Factor":
        labels=model.fit_predict(X)
        scores=model.negative_outlier_factor_
    else:
        labels=model.fit(X).predict(X)
        scores=model.decision_function(X)
    return labels,scores

def scaled_features(data):
    return StandardScaler().fit_transform(data[FEATURES])

def jaccard(a,b):
    union=a|b
    return 1.0 if not union else len(a&b)/len(union)

def compare():
    data=load_data()
    ids=data["id"].to_numpy()
    X=scaled_features(data)

    labels={};scores={}
    for name in METHODS:
        labels[name],scores[name]=fit_flags(name,X)
    flags=pd.DataFrame({name:labels[name]==-1 for name in METHODS},index=ids)

    summary=pd.DataFrame({
        "Flagged":flags.sum(),
        "Rate":flags.mean(),
    })

    overlap=[]
    for a,b in combinations(METHODS,2):
        overlap.append({"Pair":f"{a} vs {b}","Shared":int((flags[a]&flags[b]).sum()),"Jaccard":jaccard(set(ids[flags[a]]),set(ids[flags[b]]))})
    overlap=pd.DataFrame(overlap)

    rank_corr=pd.DataFrame(scores).corr(method="spearman")

    reference={name:set(ids[flags[name]]) for name in METHODS}
    seed_flags=[set(ids[fit_flags("Isolation Forest",X,seed)[0]==-1]) for seed in SEEDS]
    seed_jaccard=[jaccard(a,b) for a,b in combinations(seed_flags,2)]

    rng=np.random.default_rng(0)
    subsample_size=int(round(SUBSAMPLE_FRACTION*len(data)))
    subsample_jaccard={name:[] for name in METHODS}
    for _ in range(SUBSAMPLES):
        idx=rng.choice(len(data),size=subsample_size,replace=False)
        Xs=scaled_features(data.iloc[idx])
        kept=set(ids[idx])
        for name in METHODS:
            sub_labels,_=fit_flags(name,Xs)
            sub_flagged=set(ids[idx][sub_labels==-1])
            subsample_jaccard[name].append(jaccard(reference[name]&kept,sub_flagged))
    stability=pd.DataFrame({
        "Mean Jaccard":{name:np.mean(subsample_jaccard[name]) for name in METHODS},
        "Std":{name:np.std(subsample_jaccard[name]) for name in METHODS},
    })

    unique={}
    for name in METHODS:
        others=set().union(*[reference[o] for o in METHODS if o!=name])
        unique[name]=sorted(reference[name]-others)

    return {
        "data":data,
        "labels":labels,
        "flags":flags,
        "summary":summary,
        "overlap":overlap,
        "rank_corr":rank_corr,
        "seed_jaccard":np.mean(seed_jaccard),
        "stability":stability,
        "unique":unique,
    }

def explain_cases(result,name,limit=3):
    data=result["data"].assign(anomaly=result["labels"][name])
    cases=[]
    for project_id in result["unique"][name][:limit]:
        row=data[data["id"]==project_id].iloc[0]
        cases.append((project_id,row["name"],row["sector"],explain_project(row,data)))
    return cases

def main():
    result=compare()
    pd.set_option("display.width",140)
    print("## Flag summary (contamination=0.12, 260 projects)")
    print(result["summary"].to_string(float_format=lambda x:f"{x:.3f}"))
    print("\n## Pairwise overlap of flagged sets")
    print(result["overlap"].to_string(index=False,float_format=lambda x:f"{x:.3f}"))
    print("\n## Spearman rank correlation of scores (higher = more normal)")
    print(result["rank_corr"].to_string(float_format=lambda x:f"{x:.3f}"))
    print(f"\n## Isolation Forest seed stability ({len(list(SEEDS))} seeds): mean pairwise Jaccard = {result['seed_jaccard']:.3f}")
    print(f"\n## Subsample stability ({SUBSAMPLES} runs, {SUBSAMPLE_FRACTION:.0%} of projects, full pipeline refit)")
    print(result["stability"].to_string(float_format=lambda x:f"{x:.3f}"))
    print(f"\n## Flagged by all three methods: {int(result['flags'].all(axis=1).sum())}")
    print(f"## Flagged by none: {int((~result['flags'].any(axis=1)).sum())}")
    for name in METHODS:
        print(f"\n## Flagged only by {name}: {len(result['unique'][name])}")
        for project_id,project_name,sector,reasons in explain_cases(result,name):
            print(f"- {project_id} {project_name} ({sector})")
            for reason in reasons:
                print(f"    - {reason}")

if __name__=="__main__":
    main()
