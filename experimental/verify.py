"""Accept the prototype from tracked assets; no fitting or raw downloads needed."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

from common import ARTIFACTS, REPORTS, ROOT, digest, error_metrics, write_json
from runtime import Prototype
from service import create_app
from simulation_models import ACTIONS, FEATURES, plume_points


def close(a,b):
    if not np.isclose(a,b,rtol=1e-10,atol=1e-9):
        raise ValueError(f"Evidence mismatch: {a} != {b}")


def verify_traces():
    counts = {}
    for kind in ["outdoor", "hourly"]:
        trace = pd.read_csv(REPORTS/f"{kind}-test-trace.csv.gz")
        report = json.loads((REPORTS/f"{kind}-evaluation.json").read_text())
        if len(trace) != report["split_counts"]["test"]:
            raise ValueError("Missing or extra trace observations")
        if kind == "outdoor":
            if trace.issue_second.duplicated().any() or not np.equal(trace.target_second-trace.issue_second,30).all():
                raise ValueError("Outdoor target clocks/coverage corrupt")
        elif "station" not in trace or trace[["station","issue_local_time"]].duplicated().any():
            raise ValueError("Hourly station/clock identity corrupt")
        for column in trace:
            if not column.endswith("_actual"):
                continue
            name = column.removesuffix("_actual")
            for series, report_key in [("prediction", "selected"), ("persistence", "persistence")]:
                metric = error_metrics(trace[column],trace[name+"_"+series])
                for m in ["mae","rmse"]:
                    close(metric[m],report["test"][name][report_key][m])
        counts[kind] = len(trace)
    return counts


def cases():
    return [
        ("outdoor", "/prototype/forecast/30s", json.loads((REPORTS/"outdoor-request.json").read_text())),
        ("hourly", "/prototype/forecast/hourly", json.loads((REPORTS/"hourly-request.json").read_text())),
        ("plume", "/prototype/plume", {"simulation_only": True, "points": [
            {"x":.3,"y":0.,"t":.6,"east_velocity":.3,"north_velocity":0.,"diffusivity":.03,"removal":.2}]}),
        ("efficiency", "/prototype/efficiency", {"simulation_only":True,"humidity":60.,"temperature":30.,"particle_um":6.,"droplet_um":80.,"wind_speed_m_s":4.2}),
        ("control", "/prototype/control", {"simulation_only":True,"policy":"dqn","source_pm10":500.,"previous_source_pm10":450.,
            "zone_pm10":[20.,100.,90.,10.],"wind_from_degrees":315.,"wind_speed_m_s":4.2,"humidity":60.,"temperature":30.,
            "last_duties":[0.,0.,0.,0.],"rollouts":64})]


def verify_all(http=False, save_examples=False):
    started = time.monotonic()
    manifest = json.loads((REPORTS/"artifact-manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        path = ARTIFACTS/name
        if digest(path)!=expected["sha256"] or path.stat().st_size!=expected["bytes"]:
            raise ValueError("Fitted artifact identity mismatch: "+name)
    if digest(ROOT/"config.json")!=manifest["config_sha256"]:
        raise ValueError("Declared training protocol changed")
    original = ROOT.parent/"models/artifacts/pm10-initial.joblib"
    if digest(original)!=manifest["frozen_original_pm10_sha256"]:
        raise ValueError("Original laboratory model changed")
    counts = verify_traces()
    prototype = Prototype()
    if prototype.errors:
        raise ValueError(str(prototype.errors))
    responses = {}
    with TestClient(create_app()) as client:
        if not client.get("/health").json()["ready"]:
            raise ValueError("API not ready")
        for name, path, request in cases():
            response = client.post(path,json=request)
            if response.status_code!=200:
                raise ValueError(f"{name} API failed: {response.text}")
            responses[name] = response.json()
            if name in ["plume","efficiency","control"] and not response.json()["simulation_only"]:
                raise ValueError("Simulator output not labelled")
            if save_examples:
                write_json(ROOT/f"examples/{name}-request.json",request)
                write_json(ROOT/f"examples/{name}-response.json",{"kind":"saved_verification_response_not_live", "response":response.json()})
    # Check exported runtime executes the same floating-point networks.
    import onnxruntime as ort
    rng = np.random.default_rng(9192)
    onnx = {}
    for name, component, x in [("plume","plume",plume_points(rng,100)),
        ("spraying","control",rng.uniform(0,1,(100,len(FEATURES))).astype(np.float32))]:
        session = ort.InferenceSession(str(ARTIFACTS/(name+".onnx")),providers=["CPUExecutionProvider"])
        actual = session.run(None,{"features":x})[0]
        with torch.no_grad():
            expected = prototype.models[component](torch.from_numpy(x)).numpy()
        maximum = float(np.max(np.abs(actual-expected)))
        if maximum > 1e-4:
            raise ValueError("Export inference mismatch")
        onnx[name] = maximum
    http_result = "not_requested"
    if http:
        with socket.socket() as s:
            s.bind(("127.0.0.1",0))
            port = s.getsockname()[1]
        process = subprocess.Popen([sys.executable,str(ROOT/"serve.py"),"--port",str(port)],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        base = f"http://127.0.0.1:{port}"
        try:
            for _ in range(100):
                try:
                    health = json.loads(urlopen(base+"/health",timeout=1).read())
                    if health["ready"]:
                        break
                except OSError:
                    time.sleep(.1)
            else:
                raise ValueError("Real HTTP service did not start")
            for name,path,request in cases():
                req = Request(base+path,data=json.dumps(request).encode(),headers={"Content-Type":"application/json"})
                response = json.loads(urlopen(req,timeout=10).read())
                if response != responses[name]:
                    raise ValueError("Real HTTP versus in-process response mismatch")
            http_result = "five endpoints match actual trained inference over localhost HTTP"
        finally:
            process.terminate()
            process.wait(timeout=10)
    return {"status":"passed","platform":{"python":sys.version.split()[0],"os":platform.system(),"machine":platform.machine()},
        "artifact_files":len(manifest["artifacts"]),"trace_rows":counts,"onnx_max_absolute_difference":onnx,
        "api_endpoints_verified":5,"actual_http":http_result,"fitting_performed":False,"raw_data_required":False,
        "field_validated":False,"seconds":time.monotonic()-started}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http",action="store_true")
    parser.add_argument("--save-examples",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=verify_all(args.http,args.save_examples)
    if args.output:
        write_json(args.output,result)
    print(json.dumps(result,indent=2))
