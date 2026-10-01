"""Record actual artifacts and map report items to implemented prototype limits."""
import json
import platform
import sys
from datetime import datetime, timezone
from common import ARTIFACTS, CONFIG, REPORTS, ROOT, digest, write_json


def main():
    artifacts = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in sorted(ARTIFACTS.iterdir()) if p.is_file()}
    write_json(REPORTS / "artifact-manifest.json", {"experiment": CONFIG["experiment_id"],
        "artifacts": artifacts, "config_sha256": digest(ROOT/"config.json"),
        "python": sys.version.split()[0], "platform": platform.platform(), "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_original_pm10_sha256": "d78f1b37269f72af45933e01722968fb13ed82178f6d8b3e4c5584d46cec09c7"})
    evidence = {name: json.loads((REPORTS/name).read_text()) for name in ["datasets.json", "outdoor-evaluation.json", "hourly-evaluation.json",
        "synthetic-dataset.json", "plume-evaluation.json", "efficiency-evaluation.json", "control-evaluation.json", "onnx-verification.json"]}
    capabilities = [
        {"report_item": "PM2.5 prediction", "now": "New measured outdoor +30s model and urban +1/+3/+6h models",
         "limit": "Outdoor PM2.5 loses persistence MAE; preliminary two-recording study"},
        {"report_item": "Wind speed/direction prediction", "now": "Learned +1h east/north wind components from measured hourly urban history",
         "limit": "No measured second-scale construction wind forecast; nearest weather station and compass sectors"},
        {"report_item": "Humidity/temperature", "now": "Multi-input forecasts use both; learned +1h temperature and derived-RH forecasts",
         "limit": "Short-term temperature uses selected persistence; short RH loses baseline; urban RH is derived"},
        {"report_item": "Plume footprint / PINN", "now": "Trained PINN with advection-diffusion-reaction residual loss, normalized footprint API",
         "limit": "Idealized dimensionless simulation; approximate footprint, no calibrated site map"},
        {"report_item": "Zone selection / reinforcement learning", "now": "Trained Double-DQN, 31 discrete OFF/50%/100% zone actions",
         "limit": "Simulation only; DQN lost declared reward to reactive control, which remains default"},
        {"report_item": "Pumps/relays/valves/PWM", "now": "Bounded simulated duty and assumed-flow recommendations",
         "limit": "No physical hardware state, drivers or actuation; teammate handles integration"},
        {"report_item": "Boundary arrival / threshold probabilities", "now": "Learned calibrated hourly exceedance curves; conditional Monte Carlo zone arrival/exceedance in simulation",
         "limit": "No exact arrival guarantee or measured perimeter-validation; curves stop at declared horizons"},
        {"report_item": "No fabricated fallback", "now": "Checksum-verified model loading; unavailable components return 503",
         "limit": "Caller inputs remain unauthenticated; all simulator outputs explicitly labelled"},
        {"report_item": "Multi-modal environmental AI", "now": "Hourly PM2.5/PM10/weather multi-input regressors and risk classifiers, plus short outdoor multi-input model",
         "limit": "Two separate cadences/datasets; no invented joining of hourly wind to second-scale dust"},
        {"report_item": "Dynamic misting / particle-droplet efficiency", "now": "Trained response surrogate with particle size, droplet size, RH, temperature and wind; duty selection via DQN",
         "limit": "Generated toy capture labels; changed-efficiency stress test exposes poor transfer"},
        {"report_item": "Probabilistic threshold crossing", "now": "Six isotonic-calibrated classifiers for any future hourly exceedance in 1/3/6h",
         "limit": "Illustrative thresholds, not exact probability curves or physical perimeter probabilities"},
        {"report_item": "IoT edge streaming / ONNX", "now": "Plume/policy float ONNX and int8 policy exported, CPU parity verified; typed streaming-history API",
         "limit": "No microcontroller deployment/latency/power test; int8 changed some selected actions and is not the serving default"},
        {"report_item": "Online continuous retraining", "now": "Versioned candidate-training workflow accepts fresh measured hourly CSV, distinct declared train/validation/calibration/test periods",
         "limit": "No incoming field feed or active continuous retraining; candidate promotion remains a reviewed step"},
        {"report_item": "Uncertainty / sensor fault detection", "now": "Calibration-residual intervals and learned IsolationForest anomalies; strict missing/duplicate-clock/invalid-window rejection",
         "limit": "Not Bayesian intervals or fault diagnosis; PM2.5 coverage about 68%; extreme injection detection about 32%"}
    ]
    write_json(REPORTS / "capabilities.json", {"scope": "User-approved measured forecasts plus simulated control prototype",
        "status": "trained experimental package; not field validated", "physical_actuation": False,
        "original_model_unchanged": True, "capabilities": capabilities, "evidence": evidence})
    print("Manifest and capability evidence recorded")


if __name__ == "__main__":
    main()
