"""Train a versioned candidate from NEW collected hourly records; never replace a live model.

Input columns: timestamp,station,pm25,pm10,temperature,humidity,wind_east,
wind_north,pressure,rain. Units match the API. This workflow needs actual fresh
observations and an explicitly declared untouched test period.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd

from common import CONFIG, ROOT, digest, write_json
from datasets import CHANNELS
from train_forecasts import train_hourly


def validate_new_records(path, split_config):
    data = pd.read_csv(path)
    required = ["timestamp", "station", *CHANNELS]
    if set(data.columns) != set(required):
        raise ValueError("New records must contain exactly the documented timestamp/station/environment columns")
    data["timestamp"] = pd.to_datetime(data.timestamp, errors="raise")
    if data.timestamp.dt.tz is not None:
        raise ValueError("Use original local calendar timestamps")
    if data.timestamp.min() < pd.Timestamp("2026-10-01"):
        raise ValueError("Use newly collected records, not the published UCI or laboratory evaluation recordings")
    if data[CHANNELS].isna().any(axis=None):
        raise ValueError("Incomplete new measurements; do not backfill or fabricate them")
    import numpy as np
    if not np.isfinite(data[CHANNELS].to_numpy(float)).all():
        raise ValueError("Non-finite measurements")
    limits = {"pm25": (0,10000), "pm10": (0,10000), "temperature": (-40,80), "humidity": (0,100),
              "wind_east": (-40,40), "wind_north": (-40,40), "pressure": (800,1100), "rain": (0,500)}
    if any(not data[c].between(low,high).all() for c,(low,high) in limits.items()):
        raise ValueError("Measurements outside documented units/ranges")
    dates = [pd.Timestamp(split_config[k]) for k in ["train_target_before", "validation_start", "validation_target_before",
            "calibration_start", "calibration_target_before", "test_start"]]
    if not all(a<b for a,b in zip(dates,dates[1:])):
        raise ValueError("Declare ascending train/validation/calibration/test dates with separate gaps")
    frames = []
    stations = sorted(data.station.unique().tolist())
    if not all(isinstance(s, str) and s for s in stations):
        raise ValueError("Named monitoring stations required")
    for station, rows in data.groupby("station", sort=True):
        rows = rows.sort_values("timestamp")
        if rows.timestamp.duplicated().any() or not (rows.timestamp.diff().dropna() == pd.Timedelta(hours=1)).all():
            raise ValueError("Each station needs a consecutive hourly recording")
        frame = rows.set_index("timestamp")[CHANNELS]
        frame.attrs["station"] = station
        frames.append(frame)
    return frames, {"kind": "new caller-supplied hourly observations; provenance requires team verification",
        "source_file_sha256": digest(path), "rows": len(data), "stations": stations,
        "license": "Team must document collection authority and reuse terms for these new records",
        "humidity": "caller-supplied; verify measurement or derivation in collection manifest",
        "time": "Original station local calendar", "generated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    for arg in ["train-target-before", "validation-start", "validation-target-before", "calibration-start", "calibration-target-before", "test-start"]:
        parser.add_argument("--"+arg, required=True)
    args = parser.parse_args()
    candidate = args.candidate_dir.resolve()
    if candidate.exists() or candidate == ROOT or ROOT in candidate.parents and "tmp" not in candidate.parts:
        raise SystemExit("Use a new directory outside published experimental artifacts, preferably ignored tmp/candidates/<version>")
    fc = CONFIG["forecast"].copy()
    for key in ["train_target_before", "validation_start", "validation_target_before", "calibration_start", "calibration_target_before", "test_start"]:
        fc[key] = getattr(args, key)
    frames, provenance = validate_new_records(args.csv, fc)
    candidate.mkdir(parents=True)
    artifacts, reports = candidate/"artifacts", candidate/"reports"
    artifacts.mkdir()
    reports.mkdir()
    CONFIG["forecast"] = fc
    write_json(candidate/"declared-protocol.json", {"declared_before_fit": datetime.now(timezone.utc).isoformat(),
        "config": fc, "input": provenance, "promotion": "manual review; no live weights overwritten"})
    train_hourly(artifacts, (frames, provenance), reports)
    artifact = artifacts/"environment-hourly.joblib"
    write_json(reports/"artifact-manifest.json", {"artifacts": {artifact.name: {"sha256": digest(artifact), "bytes": artifact.stat().st_size}}})
    print("Candidate trained/evaluated in", candidate, "; live models unchanged", flush=True)


if __name__ == "__main__":
    main()
