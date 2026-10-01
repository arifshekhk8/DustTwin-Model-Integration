"""Check the frozen AI handoff using tracked files only; never fit a model."""

import argparse
import csv
from datetime import datetime
import gzip
import json
import math
from pathlib import Path
import platform
import sys
from zoneinfo import ZoneInfo

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "services/inference"))
from app import create_app
from dusttwin.events import score_warnings, threshold_events
from dusttwin.model import ForecastModel
from dusttwin.preparation import file_sha256
from dusttwin.replay import ReplayStore
from fastapi.testclient import TestClient


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, label, tolerance=1e-8):
    try:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)
    except AssertionError as error:
        raise ValueError(f"{label} differs from frozen evidence") from error


def verify_replays(store, model):
    """Reload every saved forecast from its causal history, including baselines."""
    require(store.index["artifact_sha256"] == model.metadata["artifact_sha256"], "Replay artifact identity differs")
    verified = {}
    for episode, recording in store.episodes.items():
        values = np.asarray(recording["pm10_ug_m3"], dtype=float)
        observations = np.asarray(recording["observation_seconds"], dtype=float)
        available = np.asarray(recording["available"], dtype=bool)
        issues = np.asarray(recording["forecast_issue_seconds"], dtype=int)
        require(len(values) == len(observations) == len(available), f"{episode}: grid lengths differ")
        require(len(issues) > 0 and (np.diff(issues) > 0).all(), f"{episode}: issue clocks must be unique and increasing")
        require(issues.min() >= 120 and issues.max() + 30 < len(values), f"{episode}: invalid history/target range")
        histories = sliding_window_view(values, 121)[issues - 120]
        native_times = sliding_window_view(observations, 121)[issues - 120]
        grid_times = issues[:, None] - np.arange(120, -1, -1)
        ages = grid_times - native_times
        require(np.isfinite(ages).all() and (ages >= 0).all() and (ages <= 1.5).all(), f"{episode}: future/stale history")
        require((np.diff(native_times, axis=1) >= 0).all(), f"{episode}: native times move backward")
        require(sliding_window_view(available, 121)[issues - 120].all(), f"{episode}: unavailable history")
        targets = issues + 30
        target_age = targets - observations[targets]
        require(available[targets].all() and np.isfinite(values[targets]).all()
                and np.isfinite(target_age).all() and (target_age >= 0).all()
                and (target_age <= 1.5).all(), f"{episode}: invalid target observations")
        predictions = model.predict_history(histories)
        trailing_mean = histories[:, -61:].mean(axis=1)
        same(predictions, recording["saved_forecast_pm10_ug_m3"], f"{episode}: saved predictions")
        same(trailing_mean, recording["saved_trailing_mean_pm10_ug_m3"], f"{episode}: saved mean baseline")
        verified[episode] = {"partition": recording["partition"], "issues": issues,
                             "actual": values[targets], "target_observation": observations[targets],
                             "target_age": target_age, "history_max_age": ages.max(axis=1),
                             "selected_model": predictions, "persistence": values[issues],
                             "trailing_mean": trailing_mean}
    return verified


def verify_metrics(rows, report):
    require(len(rows) == report["samples"], "Test trace sample count differs")
    episodes = {row["episode_id"] for row in rows}
    require(episodes == set(report["by_recording"]), "Test recording coverage differs")
    calculated = {}
    for episode in (None, *sorted(episodes)):
        subset = rows if episode is None else [row for row in rows if row["episode_id"] == episode]
        published = report["models"] if episode is None else report["by_recording"][episode]
        for name in report["models"]:
            errors = [float(row[f"{name}_pm10_ug_m3"]) - float(row["actual_pm10_ug_m3"]) for row in subset]
            require(all(math.isfinite(error) for error in errors), "Nonfinite test error")
            actual = {"samples": len(errors), "mae_ug_m3": math.fsum(map(abs, errors)) / len(errors),
                      "rmse_ug_m3": math.sqrt(math.fsum(error * error for error in errors) / len(errors)),
                      "mean_error_ug_m3": math.fsum(errors) / len(errors)}
            for metric, value in actual.items():
                require(math.isclose(value, published[name][metric], rel_tol=1e-12, abs_tol=1e-10),
                        f"{episode or 'pooled'}/{name}/{metric}: reported metric differs")
            if episode is None:
                calculated[name] = actual
    return calculated


def verify_trace(rows, verified):
    expected = {(episode, int(second)): index for episode, item in verified.items()
                if item["partition"] == "test" for index, second in enumerate(item["issues"])}
    seen = set()
    for row in rows:
        key = row["episode_id"], int(row["issue_second"])
        require(key in expected and key not in seen, "Unexpected or duplicate test trace clock")
        seen.add(key)
        item, index = verified[key[0]], expected[key]
        require(int(row["target_second"]) == key[1] + 30, "Test target is not issue +30 seconds")
        require(row["input_availability_mask"] == "1" * 121, "Test history is unavailable")
        for column, name in (("actual_pm10_ug_m3", "actual"), ("target_observation_second", "target_observation"),
                             ("target_age_seconds", "target_age"), ("history_max_age_seconds", "history_max_age"),
                             *((f"{name}_pm10_ug_m3", name) for name in ("selected_model", "persistence", "trailing_mean"))):
            same(float(row[column]), item[name][index], f"{key}/{column}")
    require(seen == set(expected), "Test trace omits eligible forecasts")


def verify_api(root, store, model):
    fixture = json.loads((root / "reports/training/prediction-fixture.json").read_text())
    same(model.predict_history(fixture["history_pm10_ug_m3"]), fixture["expected_predictions_ug_m3"], "Five fixtures")
    with TestClient(create_app(root=root, allowed_origins=[])) as live:
        require(live.get("/health").json()["ready"], "Live model is unavailable")
        for episode, recording in store.episodes.items():
            second = recording["forecast_issue_seconds"][0]
            snapshot = live.get(f"/v1/replay/{episode}?second={second}")
            require(snapshot.status_code == 200, "Live replay failed")
            body = snapshot.json()
            response = live.post("/v1/predict", json=body["request"])
            require(response.status_code == 200, "Direct prediction failed")
            forecast = response.json()
            same(forecast["predicted_pm10_ug_m3"], recording["saved_forecast_pm10_ug_m3"][0], "API forecast")
            require(forecast["crossing_eta_seconds"] is None and "actual_pm10_ug_m3" not in forecast, "API invents ETA or reveals future actual")
            require(body["matured_forecast"] is None and all(point["time_seconds"] <= second for point in body["past_observations"]), "Replay reveals future actual")
            later = live.get(f"/v1/replay/{episode}?second={second + 30}").json()
            require(later["matured_forecast"]["target_time_seconds"] == second + 30, "Matured target clock differs")
            same(later["matured_forecast"]["actual_pm10_ug_m3"], recording["pm10_ug_m3"][second + 30], "Matured actual")
    with TestClient(create_app(root=root, load_model=False, allowed_origins=[])) as saved:
        require(not saved.get("/health").json()["ready"], "Unavailable model claims readiness")
        require(saved.post("/v1/predict", json=body["request"]).status_code == 503, "Unavailable model fabricates a prediction")
        for episode, recording in store.episodes.items():
            response = saved.get(f"/v1/replay/{episode}?second={recording['forecast_issue_seconds'][0]}").json()
            require(response["forecast"]["mode"] == "saved_inference", "Saved replay is not labelled")
            same(response["forecast"]["predicted_pm10_ug_m3"], recording["saved_forecast_pm10_ug_m3"][0], "Saved API forecast")


def verify(root):
    manifest = json.loads((root / "models/upstream-provenance.json").read_text())
    for item in manifest["unchanged_upstream_files"]:
        path = root / item["path"]
        require(path.stat().st_size == item["bytes"] and file_sha256(path) == item["sha256"], f"Upstream file changed: {item['path']}")
    model, store = ForecastModel(root), ReplayStore(root)
    verified = verify_replays(store, model)
    report = json.loads((root / "reports/evaluation/test-metrics.json").read_text())
    for name, digest in report["evidence_sha256"].items():
        require(file_sha256(root / name) == digest, f"Evaluation evidence changed: {name}")
    require(report["artifact_sha256"] == model.metadata["artifact_sha256"], "Evaluation artifact differs")
    with gzip.open(root / report["trace_file"], "rt") as stream:
        rows = list(csv.DictReader(stream))
    verify_trace(rows, verified)
    metrics = verify_metrics(rows, report)
    warnings = {name: {"matched_events": 0, "false_alerts": 0, "scorable_events": 0} for name in metrics}
    for episode, published in report["descriptive_warnings_by_recording"].items():
        recording, item = store.episodes[episode], verified[episode]
        events = threshold_events({"available": recording["available"], "pm10": recording["pm10_ug_m3"]}, item["issues"], report["event_configuration"])
        require(events == published["events"], f"{episode}: event counts differ")
        for name in metrics:
            scored = score_warnings(item["issues"], item["persistence"], item[name], events, report["event_configuration"])
            require(scored == published["models"][name], f"{episode}/{name}: warning evidence differs")
            for field in ("matched_events", "scorable_events"):
                warnings[name][field] += scored[field]
            warnings[name]["false_alerts"] += len(scored["false_alert_issue_seconds"])
    verify_api(root, store, model)
    return {"verified_at": datetime.now(ZoneInfo("Asia/Dhaka")).isoformat(), "status": "passed",
            "scope": "Frozen laboratory PM10 model/backend handoff; no field, hardware or teammate frontend acceptance",
            "platform": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine()},
            "model_id": model.metadata["model_id"], "artifact_sha256": model.metadata["artifact_sha256"],
            "verification_script_sha256": file_sha256(Path(__file__)),
            "unchanged_upstream_files": len(manifest["unchanged_upstream_files"]), "fixture_predictions": 5,
            "recordings": len(verified), "recomputed_saved_forecasts": sum(len(item["issues"]) for item in verified.values()),
            "verified_test_rows": len(rows), "test_metrics": metrics, "descriptive_warnings": warnings,
            "checks": ["all included causal histories and target timestamps", "all saved forecasts and mean baselines",
                       "complete test clock coverage, trace values and pooled/per-recording errors", "recomputed descriptive warning events",
                       "live replay/direct API equality and clock-only actual reveal", "unavailable live model returns 503; saved replay is labelled"],
            "raw_or_prepared_training_data_required": False, "trained_again": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save actual check results as JSON")
    args = parser.parse_args()
    try:
        result = verify(ROOT)
    except (OSError, ValueError, KeyError, IndexError) as error:
        print(f"AI verification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
