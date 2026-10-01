"""Train new measured-data experiments. Never load the original laboratory test."""
from pathlib import Path
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier, IsolationForest
from sklearn.isotonic import IsotonicRegression

from common import ARTIFACTS, CONFIG, REPORTS, digest, error_metrics, write_json
from datasets import (CHANNELS, SHORT_CHANNELS, hourly_features, short_features,
                      short_frames, uci_frames)


def temporal_parts(clocks, horizon_seconds, lookback_seconds, config, short=False):
    if short:
        end = float(clocks.max() + horizon_seconds)
        boundary1 = int(end * config["train_fraction"])
        boundary2 = int(end * (config["train_fraction"] + config["validation_fraction"]))
        gap = config["split_gap_seconds"]
        return {"train": clocks + horizon_seconds < boundary1,
                "validation": (clocks >= boundary1 + gap) & (clocks + horizon_seconds < boundary2),
                "calibration": clocks >= boundary2 + gap}
    target = clocks + pd.Timedelta(seconds=horizon_seconds)
    return {"train": target < pd.Timestamp(config["train_target_before"]),
        "validation": (clocks >= pd.Timestamp(config["validation_start"]) + pd.Timedelta(seconds=lookback_seconds))
                      & (target < pd.Timestamp(config["validation_target_before"])),
        "calibration": (clocks >= pd.Timestamp(config["calibration_start"]) + pd.Timedelta(seconds=lookback_seconds))
                       & (target < pd.Timestamp(config["calibration_target_before"])),
        "test": clocks >= pd.Timestamp(config["test_start"]) + pd.Timedelta(seconds=lookback_seconds)}


def bounded(pred, target):
    if target.startswith("pm"):
        return np.maximum(pred, 0)
    if target == "humidity":
        return np.clip(pred, 0, 100)
    return pred


def fit_regressors(x, targets, current, parts, cfg, minimum_leaf=100):
    models, summaries = {}, {}
    for name, actual in targets.items():
        channel = name.split("@")[0]
        # Predict a change from the current value; all features are causal.
        model = HistGradientBoostingRegressor(max_iter=cfg["iterations"], max_leaf_nodes=cfg["max_leaf_nodes"],
            min_samples_leaf=minimum_leaf, early_stopping=False, l2_regularization=1,
            random_state=CONFIG["seed"])
        model.fit(x[parts["train"]], (actual - current[channel])[parts["train"]])
        val = parts["validation"]
        learned = bounded(model.predict(x[val]) + current[channel][val], channel)
        baseline = current[channel][val]
        choices = [0, .25, .5, .75, 1]
        weight = min(choices, key=lambda w: np.mean((actual[val] - bounded(w * learned + (1-w) * baseline, channel)) ** 2))
        cal = parts["calibration"]
        cal_prediction = bounded(weight * model.predict(x[cal]) + current[channel][cal], channel)
        residual = actual[cal] - cal_prediction
        rank = min(len(residual), int(np.ceil((len(residual)+1) * .9)))
        radius = float(np.partition(np.abs(residual), rank-1)[rank-1])
        models[name] = {"model": model, "learned_weight": weight, "interval_radius": radius,
                        "calibration_residuals": residual.astype(np.float32)}
        summaries[name] = {"validation_learned": error_metrics(actual[val], learned),
            "validation_persistence": error_metrics(actual[val], baseline),
            "validation_selected": error_metrics(actual[val], bounded(weight*learned + (1-weight)*baseline, channel)),
            "learned_weight_selected_without_test": weight, "interval_radius": radius,
            "calibration_rows": int(cal.sum())}
        print("Fitted", name, "blend", weight, "interval radius", round(radius, 3), flush=True)
    return models, summaries


def predict_bundle(bundle, x, current):
    result = {}
    for name, fitted in bundle["models"].items():
        channel = name.split("@")[0]
        delta = fitted["model"].predict(x)
        learned = bounded(current[channel] + delta, channel)
        selected = bounded(current[channel] + fitted["learned_weight"] * delta, channel)
        result[name] = {"learned": learned, "selected": selected,
            "lower": bounded(selected - fitted["interval_radius"], channel),
            "upper": bounded(selected + fitted["interval_radius"], channel)}
    return result


def evaluate(bundle, x, targets, current, mask):
    pred = predict_bundle(bundle, x[mask], {k: v[mask] for k, v in current.items()})
    report = {}
    for name, p in pred.items():
        channel = name.split("@")[0]
        y = targets[name][mask]
        report[name] = {"rows": int(mask.sum()), "learned": error_metrics(y, p["learned"]),
            "selected": error_metrics(y, p["selected"]), "persistence": error_metrics(y, current[channel][mask]),
            "empirical_90pct_interval_coverage": float(((y >= p["lower"]) & (y <= p["upper"])).mean()),
            "mean_interval_width": float((p["upper"]-p["lower"]).mean())}
    return report, pred


def train_short(output):
    frames, provenance = short_frames()
    matrices = []
    for f in frames:
        x = short_features(f)
        y = f[SHORT_CHANNELS].shift(-30)
        valid = x.notna().all(axis=1) & y.notna().all(axis=1)
        # Also require the complete causal 121-snapshot history.
        valid &= f.notna().all(axis=1).rolling(121, min_periods=121).sum().eq(121)
        matrices.append((x.loc[valid], y.loc[valid], f.loc[valid]))
    x0, y0, c0 = matrices[0]
    parts = temporal_parts(x0.index.to_numpy(), 30, 120, CONFIG["short_forecast"], short=True)
    if any(mask.sum() < 50 for mask in parts.values()):
        raise ValueError("Outdoor recording too short after independent split gaps")
    targets = {f"{c}@30s": y0[c].to_numpy() for c in SHORT_CHANNELS}
    current = {c: c0[c].to_numpy() for c in SHORT_CHANNELS}
    models, val = fit_regressors(x0.to_numpy(), targets, current, parts, CONFIG["short_forecast"], 20)
    bundle = {"kind": "measured_outdoor_30s", "models": models, "features": list(x0.columns),
              "channels": SHORT_CHANNELS, "config": CONFIG["short_forecast"], "source": provenance}
    artifact = output / "outdoor-30s.joblib"
    joblib.dump(bundle, artifact, compress=3)
    frozen_hash = digest(artifact)  # Freeze before touching Day 2 targets.
    x1, y1, c1 = matrices[1]
    actual = {f"{c}@30s": y1[c].to_numpy() for c in SHORT_CHANNELS}
    base = {c: c1[c].to_numpy() for c in SHORT_CHANNELS}
    result, predictions = evaluate(bundle, x1.to_numpy(), actual, base, np.ones(len(x1), dtype=bool))
    trace = pd.DataFrame({"issue_second": x1.index, "target_second": x1.index + 30})
    for name, p in predictions.items():
        trace[name+"_actual"] = actual[name]
        trace[name+"_prediction"] = p["selected"]
        trace[name+"_persistence"] = base[name.split("@")[0]]
    trace.to_csv(REPORTS / "outdoor-test-trace.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    issue = int(x0.index[parts["validation"]][0])
    history = frames[0].loc[issue-120:issue].reset_index(names="second")
    history["second"] = history["second"].astype(int)
    write_json(REPORTS / "outdoor-request.json", {"history": history.to_dict(orient="records"), "history_kind": "measured_validation_recording"})
    write_json(REPORTS / "outdoor-evaluation.json", {"source": "measured outdoor OPC-N3; no spraying treatment",
        "artifact_sha256_frozen_before_test": frozen_hash, "validation": val, "test": result,
        "split_counts": {k: int(v.sum()) for k, v in parts.items()} | {"test": len(x1)},
        "clock_spans": {k: [float(x0.index[v].min()), float(x0.index[v].max())] for k, v in parts.items()},
        "limits": ["Only two related outdoor recordings; no general site validation", "Day 1 dates absent and timestamps quantized to seconds",
                   "Temperature/RH are instrument readings; wind absent", "Coverage empirical on this recording, not guaranteed for correlated streams"]})
    return provenance


def train_hourly(output, frames_and_provenance=None, report_dir=REPORTS):
    frames, provenance = uci_frames() if frames_and_provenance is None else frames_and_provenance
    matrices, targets, currents, clock_parts = [], [], [], []
    fc = CONFIG["forecast"]
    for f in frames:
        x = hourly_features(f, f.attrs["station"], provenance["stations"])
        target = {f"{c}@1h": f[c].shift(-1) for c in CHANNELS[:6]}
        for horizon in [3, 6]:
            for c in ["pm25", "pm10"]:
                target[f"{c}@{horizon}h"] = f[c].shift(-horizon)
        for c, threshold in [("pm25", 35), ("pm10", 150)]:
            for horizon in [1, 3, 6]:
                future = pd.concat([f[c].shift(-i) for i in range(1, horizon+1)], axis=1)
                label = future.max(axis=1).ge(threshold).astype(float)
                label[future.isna().any(axis=1)] = np.nan
                target[f"{c}_cross@{horizon}h"] = label
        t = pd.DataFrame(target, index=f.index)
        valid = x.notna().all(axis=1) & t.notna().all(axis=1)
        valid &= f.notna().all(axis=1).rolling(25, min_periods=25).sum().eq(25)
        matrices.append(x.loc[valid])
        targets.append(t.loc[valid])
        currents.append(f.loc[valid])
        clock_parts.append(f.index[valid])
    x = pd.concat(matrices)
    y = pd.concat(targets)
    c = pd.concat(currents)
    clocks = pd.DatetimeIndex(np.concatenate(clock_parts))
    parts = temporal_parts(clocks, 6*3600, 24*3600, fc)
    if parts["train"].sum() < 500 or any(parts[k].sum() < 100 for k in ["validation", "calibration", "test"]):
        raise ValueError("Insufficient fresh training/validation/calibration/test observations")
    targets = {name: y[name].to_numpy() for name in y if "_cross" not in name}
    current = {name: c[name].to_numpy() for name in CHANNELS}
    models, validation = fit_regressors(x.to_numpy(), targets, current, parts, fc)
    risks = {}
    for name in [name for name in y if "_cross" in name]:
        classifier = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
            min_samples_leaf=100, early_stopping=False, l2_regularization=1, random_state=CONFIG["seed"])
        classifier.fit(x.to_numpy()[parts["train"]], y[name].to_numpy()[parts["train"]])
        cal = parts["calibration"]
        raw = classifier.predict_proba(x.to_numpy()[cal])[:, 1]
        calibration = IsotonicRegression(out_of_bounds="clip").fit(raw, y[name].to_numpy()[cal])
        risks[name] = {"model": classifier, "calibration": calibration}
        print("Fitted cumulative threshold risk", name, flush=True)
    anomaly = IsolationForest(n_estimators=100, max_samples=1024, random_state=CONFIG["seed"], n_jobs=2)
    anomaly.fit(x.to_numpy()[parts["train"]])
    normal_scores = anomaly.score_samples(x.to_numpy()[parts["calibration"]])
    anomaly_cut = float(np.quantile(normal_scores, .01))
    bundle = {"kind": "measured_hourly_environment", "models": models, "risks": risks,
        "anomaly": anomaly, "anomaly_cut": anomaly_cut, "features": list(x.columns),
        "channels": CHANNELS, "config": fc, "source": provenance}
    artifact = output / "environment-hourly.joblib"
    joblib.dump(bundle, artifact, compress=3)
    frozen_hash = digest(artifact)  # All selection/calibration completed before opening test.
    report, pred = evaluate(bundle, x.to_numpy(), targets, current, parts["test"])
    test_x = x.to_numpy()[parts["test"]]
    risk_report = {}
    for channel in ["pm25", "pm10"]:
        previous = np.zeros(len(test_x))
        for h in [1, 3, 6]:
            name = f"{channel}_cross@{h}h"
            r = risks[name]
            probability = np.maximum(previous, r["calibration"].predict(r["model"].predict_proba(test_x)[:, 1]))
            label = y[name].to_numpy()[parts["test"]]
            train_rate = float(y[name].to_numpy()[parts["train"]].mean())
            risk_report[name] = {"brier_score": float(np.square(probability - label).mean()),
                "constant_training_rate_brier": float(np.square(train_rate - label).mean()),
                "observed_event_rate": float(label.mean()), "mean_predicted_probability": float(probability.mean()),
                "threshold": 35 if channel == "pm25" else 150}
            previous = probability
    normal = anomaly.score_samples(test_x)
    injected = test_x.copy()
    # Anomaly challenge is artificial; these labels do not prove real sensor clog diagnosis.
    injected[:, :10] *= 20
    fault_report = {"normal_test_flag_rate": float((normal < anomaly_cut).mean()),
        "injected_extreme_value_detection_rate": float((anomaly.score_samples(injected) < anomaly_cut).mean()),
        "labels": "synthetically multiplied lag features, not real faults",
        "model": "IsolationForest trained on measured training windows"}
    index = np.flatnonzero(parts["validation"])[0]
    station = next(name.removeprefix("station_") for name in x.columns if name.startswith("station_") and x.iloc[index][name] == 1)
    issue = clocks[index]
    source = next(f for f in frames if f.attrs["station"] == station)
    history = source.loc[issue-pd.Timedelta(hours=24):issue].reset_index(names="timestamp")
    history["timestamp"] = history.timestamp.astype(str)
    write_json(report_dir / "hourly-request.json", {"station": station, "history": history.to_dict(orient="records"),
        "history_kind": "measured_validation_recording_with_derived_RH"})
    station_columns = [c for c in x.columns if c.startswith("station_")]
    trace = pd.DataFrame({"station": x[station_columns].loc[parts["test"]].idxmax(axis=1).str.removeprefix("station_").to_numpy(),
                          "issue_local_time": clocks[parts["test"]].astype(str)})
    for name in ["pm25@1h", "pm10@1h"]:
        trace[name+"_actual"] = targets[name][parts["test"]]
        trace[name+"_prediction"] = pred[name]["selected"]
        trace[name+"_persistence"] = current[name.split("@")[0]][parts["test"]]
    trace.to_csv(report_dir / "hourly-test-trace.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    write_json(report_dir / "hourly-evaluation.json", {"source": provenance.get("kind", "measured hourly urban station records, derived humidity"),
        "artifact_sha256_frozen_before_test": frozen_hash, "validation": validation, "test": report,
        "cumulative_threshold_risk": risk_report, "anomaly_challenge": fault_report,
        "split_counts": {k: int(v.sum()) for k, v in parts.items()}, "total_usable_rows": len(x),
        "discarded_incomplete_or_gap_rows": sum(map(len, frames))-len(x),
        "split_spans": {k: [str(clocks[v].min()), str(clocks[v].max())] for k, v in parts.items()},
        "limits": ["Hourly cadence cannot validate 30-second wind-aware forecasts", "Station weather matched from nearest weather station",
                   "Humidity is derived, not directly measured", "No spraying, nozzle or spatial plume measurements",
                   "Intervals and probabilities need recalibration at a new site", "Anomalies are possible faults, not diagnoses"]})
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    output = args.candidate_dir
    if any((output / name).exists() for name in ["outdoor-30s.joblib", "environment-hourly.joblib"]):
        raise SystemExit("Refusing to overwrite fitted models. Use a new candidate directory and a newly untouched evaluation period.")
    output.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    # Retraining candidates write their own evidence; never overwrite published test results.
    if output != ARTIFACTS:
        raise SystemExit("New campaigns must copy the experiment with newly declared splits/evidence paths; see retrain_candidate.py")
    started = time.monotonic()
    short = train_short(output)
    hourly = train_hourly(output)
    write_json(REPORTS / "datasets.json", {"mendeley_outdoor": short, "uci": hourly,
        "training_seconds": time.monotonic()-started, "config_sha256": digest(Path(__file__).with_name("config.json"))})
    print("Forecast training complete", flush=True)


if __name__ == "__main__":
    main()
