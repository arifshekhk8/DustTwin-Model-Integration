"""Recompute published new test predictions from pinned raw sources without fitting."""
import json
import numpy as np
import pandas as pd
import joblib
from common import ARTIFACTS, CONFIG, REPORTS, ROOT, digest, error_metrics, write_json
from datasets import CHANNELS, SHORT_CHANNELS, hourly_features, short_features, short_frames, uci_frames
from train_forecasts import predict_bundle, temporal_parts


def same(a,b):
    np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-8)


def main():
    counts = {}
    for kind in ["outdoor","hourly"]:
        bundle = joblib.load(ARTIFACTS/("outdoor-30s.joblib" if kind=="outdoor" else "environment-hourly.joblib"))
        report = json.loads((REPORTS/f"{kind}-evaluation.json").read_text())
        trace = pd.read_csv(REPORTS/f"{kind}-test-trace.csv.gz")
        if kind=="outdoor":
            frames, source = short_frames()
            f = frames[1]
            x = short_features(f)
            y = pd.DataFrame({f"{c}@30s": f[c].shift(-30) for c in SHORT_CHANNELS})
            valid = x.notna().all(axis=1)&y.notna().all(axis=1)
            valid &= f.notna().all(axis=1).rolling(121,min_periods=121).sum().eq(121)
            x, y, current = x.loc[valid], y.loc[valid], f.loc[valid]
            same(x.index,trace.issue_second)
            clocks = x.index
        else:
            frames, source = uci_frames()
            xx,yy,cc,tt = [],[],[],[]
            for f in frames:
                x = hourly_features(f,f.attrs["station"])
                target = {f"{c}@1h": f[c].shift(-1) for c in CHANNELS[:6]}
                for h in [3,6]:
                    for c in ["pm25","pm10"]: target[f"{c}@{h}h"] = f[c].shift(-h)
                y = pd.DataFrame(target,index=f.index)
                valid=x.notna().all(axis=1)&y.notna().all(axis=1)
                valid &= pd.concat([f[c].shift(-h) for c in ["pm25","pm10"] for h in range(1,7)],axis=1).notna().all(axis=1)
                valid &= f.notna().all(axis=1).rolling(25,min_periods=25).sum().eq(25)
                parts=temporal_parts(f.index,6*3600,24*3600,CONFIG["forecast"])
                chosen=valid&parts["test"]
                xx.append(x.loc[chosen]); yy.append(y.loc[chosen]); cc.append(f.loc[chosen]); tt.extend(f.index[chosen])
            x,y,current = pd.concat(xx),pd.concat(yy),pd.concat(cc)
            clocks=pd.DatetimeIndex(tt)
            if list(clocks.astype(str))!=list(trace.issue_local_time): raise ValueError("Hourly test clocks changed")
        if len(x)!=report["split_counts"]["test"]: raise ValueError("Complete source coverage mismatch")
        current_arrays={c:current[c].to_numpy() for c in current}
        p=predict_bundle(bundle,x.reindex(columns=bundle["features"]).to_numpy(),current_arrays)
        for name, values in p.items():
            metrics=error_metrics(y[name],values["selected"])
            for m in ["mae","rmse"]: same(metrics[m],report["test"][name]["selected"][m])
            coverage=float(((y[name]>=values["lower"])&(y[name]<=values["upper"])).mean())
            same(coverage,report["test"][name]["empirical_90pct_interval_coverage"])
            if name+"_actual" in trace:
                same(y[name],trace[name+"_actual"])
                same(values["selected"],trace[name+"_prediction"])
                same(current[name.split("@")[0]],trace[name+"_persistence"])
        if kind=="hourly":
            # Identity keys make coincident clocks from different stations unambiguous.
            station=x[[c for c in x if c.startswith("station_")]].idxmax(axis=1).str.removeprefix("station_").to_numpy()
            if "station" not in trace or list(trace.station)!=list(station): raise ValueError("Missing or mismatched station identity")
            if trace[["station","issue_local_time"]].duplicated().any(): raise ValueError("Duplicate station clocks")
        counts[kind]=len(x)
    provenance=json.loads((ROOT.parent/"models/upstream-provenance.json").read_text())
    for file in provenance["unchanged_upstream_files"]:
        path=ROOT.parent/file["path"]
        if digest(path)!=file["sha256"] or path.stat().st_size!=file["bytes"]: raise ValueError("Frozen upstream file changed")
    write_json(REPORTS/"source-reproduction.json",{"status":"passed","refitted":False,
        "all_new_forecasts_recomputed_from_pinned_sources":counts,"interval_coverage_recomputed":True,
        "original_upstream_files_unchanged":len(provenance["unchanged_upstream_files"]),
        "scripts":{"datasets.py":digest(ROOT/"datasets.py"),"source_audit.py":digest(ROOT/"source_audit.py")}})
    print("Source reproduction passed",counts)


if __name__=="__main__": main()
