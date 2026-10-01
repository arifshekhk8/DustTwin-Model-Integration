"""Load checksum-verified models and expose honest experimental predictions."""
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch

from common import ARTIFACTS, ROOT, digest
from datasets import CHANNELS, SHORT_CHANNELS, STATIONS, hourly_features, short_features
from simulation_models import ACTIONS, PlumePINN, QNetwork, SprayEnvironment, ZONES
from train_forecasts import predict_bundle


class Prototype:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.models, self.errors = {}, {}
        manifest = self.root / "reports/artifact-manifest.json"
        self.manifest = json.loads(manifest.read_text()) if manifest.exists() else {}
        for component, name in [("outdoor", "outdoor-30s.joblib"), ("hourly", "environment-hourly.joblib"),
                                 ("efficiency", "spray-efficiency.joblib"), ("plume", "plume-pinn.pt"), ("control", "spraying-dqn.pt")]:
            try:
                path = self.root / "artifacts" / name
                expected = self.manifest.get("artifacts", {}).get(name)
                if expected is None or digest(path) != expected["sha256"]:
                    raise ValueError("Missing or mismatched verified model artifact")
                if name.endswith(".joblib"):
                    self.models[component] = joblib.load(path)
                else:
                    model = PlumePINN() if component == "plume" else QNetwork()
                    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                    self.models[component] = model.eval()
            except (OSError, ValueError, RuntimeError, KeyError, ImportError, AttributeError, EOFError) as error:
                self.errors[component] = str(error)

    def require(self, component):
        if component not in self.models:
            raise RuntimeError(f"{component} model unavailable; no generated fallback prediction")
        return self.models[component]

    def short(self, history):
        bundle = self.require("outdoor")
        frame = pd.DataFrame(history).set_index("second")
        matrix = short_features(frame).iloc[-1:].reindex(columns=bundle["features"])
        if matrix.isna().any(axis=None):
            raise ValueError("Incomplete causal feature window")
        current = {c: frame[c].iloc[-1:].to_numpy() for c in SHORT_CHANNELS}
        predictions = predict_bundle(bundle, matrix.to_numpy(), current)
        evaluation = json.loads((self.root / "reports/outdoor-evaluation.json").read_text())
        outputs = {}
        for name, p in predictions.items():
            f = bundle["models"][name]
            outputs[name] = {"prediction": float(p["selected"][0]), "learned_prediction": float(p["learned"][0]),
                "unit": {"pm25":"ug/m3","pm10":"ug/m3","temperature":"degC","humidity":"percent"}[name.split("@")[0]],
                "persistence_baseline": float(current[name.split("@")[0]][0]), "learned_weight": f["learned_weight"],
                "method": "persistence_selected_on_validation" if f["learned_weight"] == 0 else "validation_selected_learned_persistence_blend",
                "interval": {"lower": float(p["lower"][0]), "upper": float(p["upper"][0]),
                    "nominal_coverage": .9, "heldout_recording_coverage": evaluation["test"][name]["empirical_90pct_interval_coverage"],
                    "guaranteed_coverage": False}}
        return {"experiment": "measured_outdoor_30s", "horizon_seconds": 30,
            "issue_second": int(frame.index[-1]), "target_second": int(frame.index[-1]+30),
            "model_training_data": "measured PM2.5, PM10, instrument temperature and humidity",
            "input_provenance": "supplied_by_caller; not independently authenticated", "outputs": outputs,
            "field_validated": False, "crossing_eta_seconds": None,
            "warnings": ["New outdoor experiment; original laboratory model unchanged",
                         "PM2.5 nominal 90% interval covered only about 68% on the held-out recording",
                         "Humidity prediction lost to persistence on the held-out recording",
                         "No wind inputs, spraying labels or causal treatment estimate"]}

    def hourly(self, station, history):
        bundle = self.require("hourly")
        stations = bundle["source"]["stations"]
        if station not in stations:
            raise ValueError("Unknown station: this hourly model is trained for the listed Beijing stations")
        frame = pd.DataFrame(history)
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("timestamp")))
        matrix = hourly_features(frame, station, stations).iloc[-1:].reindex(columns=bundle["features"])
        if matrix.isna().any(axis=None):
            raise ValueError("Incomplete hourly feature window")
        current = {c: frame[c].iloc[-1:].to_numpy() for c in CHANNELS}
        predictions = predict_bundle(bundle, matrix.to_numpy(), current)
        outputs = {}
        for name, p in predictions.items():
            f = bundle["models"][name]
            outputs[name] = {"prediction": float(p["selected"][0]), "learned_prediction": float(p["learned"][0]),
                "unit": {"pm25":"ug/m3","pm10":"ug/m3","temperature":"degC","humidity":"percent","wind_east":"m/s","wind_north":"m/s"}[name.split("@")[0]],
                "persistence_baseline": float(current[name.split("@")[0]][0]), "learned_weight": f["learned_weight"],
                "method": "persistence_selected_on_validation" if f["learned_weight"] == 0 else "validation_selected_learned_persistence_blend",
                "interval": {"lower": float(p["lower"][0]), "upper": float(p["upper"][0]),
                             "nominal_coverage": .9, "guaranteed_coverage": False}}
        curves = {}
        for channel, threshold in [("pm25", 35), ("pm10", 150)]:
            last = 0.
            curves[channel] = []
            for h in [1, 3, 6]:
                r = bundle["risks"][f"{channel}_cross@{h}h"]
                raw = r["model"].predict_proba(matrix.to_numpy())[:, 1]
                last = max(last, float(r["calibration"].predict(raw)[0]))
                curves[channel].append({"within_hours": h, "probability_at_any_future_hourly_observation": last,
                                       "illustrative_threshold_ug_m3": threshold})
        score = float(bundle["anomaly"].score_samples(matrix.to_numpy())[0])
        ue, vn = outputs["wind_east@1h"]["prediction"], outputs["wind_north@1h"]["prediction"]
        wind = {"speed_m_s": float(np.hypot(ue, vn)), "from_degrees": float((np.degrees(np.arctan2(ue, vn))+180)%360)}
        if wind["speed_m_s"] < .1:
            wind["from_degrees"] = None
        return {"experiment": "measured_hourly_environment", "station": station, "cadence_seconds": 3600,
            "issue_local_time": str(frame.index[-1]), "outputs": outputs, "wind_1h": wind,
            "threshold_probability_curves": curves, "anomaly": {"possible_sensor_or_environment_shift": score<bundle["anomaly_cut"],
                "score": score, "fault_diagnosis": None, "calibration_normal_flag_target": .01},
            "field_validated": False, "crossing_eta_seconds": None,
            "warnings": ["Hourly model cannot support a 30-second wind-aware claim", "Training humidity was derived from dewpoint",
                         "Thresholds are illustrative experiment settings", "Probability and interval calibration are site/distribution dependent",
                         "A model anomaly is not proof of a clogged or failed sensor"]}

    def plume(self, points):
        model = self.require("plume")
        x = np.asarray(points, dtype=np.float32)
        with torch.no_grad():
            prediction = model(torch.from_numpy(x)).numpy()[:, 0]
        return {"simulation_only": True, "model": "physics_informed_neural_network", "normalized_concentration": prediction.tolist(),
            "coordinates": "dimensionless idealized free-space pulse", "exact_footprint": False,
            "warnings": ["No site geometry or measured plume calibration", "Do not convert these normalized outputs to absolute field PM"]}

    def efficiency(self, features):
        model = self.require("efficiency")
        value = float(np.clip(model.predict([features])[0], 0, 1))
        return {"simulation_only": True, "assumed_capture_fraction": value,
            "model_training_data": "generated toy droplet/particle response", "measured_capture_validation": False}

    def control(self, request):
        model = self.require("control")
        efficiency = self.require("efficiency")
        theta = np.radians((request["wind_from_degrees"]+180)%360)
        concentration = np.asarray(request["zone_pm10"], float)
        obs = np.array([request["source_pm10"]/1500, (request["source_pm10"]-request["previous_source_pm10"])/500,
            *concentration/150, np.sin(theta), np.cos(theta), request["wind_speed_m_s"]/8,
            request["humidity"]/100, request["temperature"]/50, *request["last_duties"]], np.float32)
        with torch.no_grad():
            learned_action = int(model(torch.from_numpy(obs[None])).argmax(1).item())
        if request["policy"] == "reactive":
            action = sum(1 << j for j, c in enumerate(concentration) if c>120)
        else:
            action = learned_action
        duty = ACTIONS[action]
        # Conditional simulator rollouts; no true future observations enter the result.
        trajectories = []
        for sample in range(request["rollouts"]):
            env = SprayEnvironment(9191+sample, efficiency)
            env.source, env.previous_source = request["source_pm10"], request["previous_source_pm10"]
            env.concentration = concentration.copy()
            env.theta, env.speed = float(theta), request["wind_speed_m_s"]
            env.humidity, env.temperature = request["humidity"], request["temperature"]
            env.last_duty = np.asarray(request["last_duties"], float)
            env.efficiency = float(np.clip(efficiency.predict([[env.humidity, env.temperature, 6, 80, env.speed]])[0], 0, 1))
            trajectory = [env.concentration.copy()]
            for _ in range(12):
                observation = env.observe()
                if request["policy"] == "reactive":
                    a = sum(1 << j for j, c in enumerate(env.concentration) if c>120)
                else:
                    with torch.no_grad():
                        a = int(model(torch.from_numpy(observation[None])).argmax(1).item())
                env.step(a)
                trajectory.append(env.concentration.copy())
            trajectories.append(trajectory)
        values = np.asarray(trajectories)
        exceeded = values >= 150
        cumulative = np.maximum.accumulate(exceeded, axis=1)
        curves, eta = {}, {}
        for zone, name in enumerate(ZONES):
            curves[name] = [{"within_seconds": i*5, "simulated_probability": float(cumulative[:, i, zone].mean())} for i in range(13)]
            first = np.where(exceeded[:, :, zone].any(axis=1), exceeded[:, :, zone].argmax(axis=1)*5, np.inf)
            median = float(np.median(first))
            eta[name] = None if not np.isfinite(median) else median
        return {"simulation_only": True, "policy": request["policy"], "trained_dqn_action": learned_action,
            "recommendations": {name: {"duty_fraction": float(duty[i]), "assumed_flow_L_min": float(duty[i]*.5)} for i, name in enumerate(ZONES)},
            "physical_actuation": False, "perimeter_probability_curves": curves, "simulated_median_arrival_seconds": eta,
            "arrival_is_exact": False, "illustrative_threshold_ug_m3": 150, "rollouts": request["rollouts"],
            "warnings": ["Synthetic conditional source/wind evolution; not measured arrival accuracy",
                         "DQN lost to reactive control on the declared test reward; default is reactive",
                         "DQN reduced test exposure but used more water than reactive control",
                         "Particle=6um and droplet=80um are fixed toy control assumptions", "Commands are recommendations only"]}
