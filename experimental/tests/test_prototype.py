import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import os

import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import digest
from datasets import hourly_features, short_features, CHANNELS
from runtime import Prototype
from service import create_app
from simulation_models import plume_truth
from train_forecasts import temporal_parts
from retrain_candidate import validate_new_records


class CausalityTests(unittest.TestCase):
    def test_future_values_do_not_change_features(self):
        rng = np.random.default_rng(12)
        short = pd.DataFrame(rng.uniform(1, 10, (181, 4)), columns=["pm25", "pm10", "temperature", "humidity"])
        before = short_features(short).loc[120].to_numpy()
        short.loc[121:] *= 1000
        np.testing.assert_equal(before, short_features(short).loc[120].to_numpy())
        hourly = pd.DataFrame(rng.uniform(1, 10, (31, 8)), columns=CHANNELS,
                              index=pd.date_range("2014-01-01", periods=31, freq="h"))
        before = hourly_features(hourly, "Dongsi").iloc[24].to_numpy()
        hourly.iloc[25:] *= 1000
        np.testing.assert_equal(before, hourly_features(hourly, "Dongsi").iloc[24].to_numpy())

    def test_outdoor_split_windows_are_disjoint(self):
        cfg = json.loads((ROOT/"config.json").read_text())["short_forecast"]
        clocks = np.arange(120, 3700)
        parts = temporal_parts(clocks, 30, 120, cfg, short=True)
        for a,b in [("train", "validation"), ("validation", "calibration")]:
            self.assertLess(clocks[parts[a]].max()+30, clocks[parts[b]].min()-120)

    def test_synthetic_truth_satisfies_declared_pde(self):
        x = torch.tensor([[.1,.2,.4,.2,-.1,.02,.3]], dtype=torch.float64, requires_grad=True)
        xx,yy,t,u,v,d,k = x.unbind(1)
        variance = .04 + 2*d*t
        c = .04/variance*torch.exp(-k*t)*torch.exp(-((xx-u*t)**2+(yy-v*t)**2)/(2*variance))
        g = torch.autograd.grad(c.sum(), x, create_graph=True)[0]
        cxx = torch.autograd.grad(g[:,0].sum(), x, create_graph=True)[0][:,0]
        cyy = torch.autograd.grad(g[:,1].sum(), x, create_graph=True)[0][:,1]
        residual = g[:,2]+u*g[:,0]+v*g[:,1]-d*(cxx+cyy)+k*c
        self.assertLess(float(residual.abs().max().detach()), 1e-12)
        np.testing.assert_allclose(c.detach().numpy(), plume_truth(x.detach().numpy())[:,0], atol=1e-12)

    def test_original_artifact_unchanged(self):
        self.assertEqual(digest(ROOT.parent/"models/artifacts/pm10-initial.joblib"),
                         "d78f1b37269f72af45933e01722968fb13ed82178f6d8b3e4c5584d46cec09c7")


class APIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(create_app())
        cls.client = cls.context.__enter__()
        cls.short = json.loads((ROOT/"reports/outdoor-request.json").read_text())
        cls.hourly = json.loads((ROOT/"reports/hourly-request.json").read_text())
        cls.control = {"simulation_only": True, "policy": "dqn", "source_pm10": 500., "previous_source_pm10": 450.,
            "zone_pm10": [20.,100.,90.,10.], "wind_from_degrees": 315., "wind_speed_m_s": 4.2,
            "humidity": 60., "temperature": 30., "last_duties": [0.,0.,0.,0.], "rollouts": 8}

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None,None,None)

    def test_health_and_measured_forecasts(self):
        self.assertTrue(self.client.get("/health").json()["ready"])
        result = self.client.post("/prototype/forecast/30s", json=self.short)
        self.assertEqual(result.status_code, 200, result.text)
        data = result.json()
        self.assertEqual(data["target_second"]-data["issue_second"], 30)
        self.assertIn("pm25@30s", data["outputs"])
        self.assertFalse(data["field_validated"])
        self.assertLess(data["outputs"]["pm25@30s"]["interval"]["heldout_recording_coverage"], .7)

    def test_hourly_probabilities_wind_and_anomaly(self):
        result = self.client.post("/prototype/forecast/hourly", json=self.hourly)
        self.assertEqual(result.status_code, 200, result.text)
        data = result.json()
        self.assertEqual(data["cadence_seconds"], 3600)
        for curve in data["threshold_probability_curves"].values():
            values = [p["probability_at_any_future_hourly_observation"] for p in curve]
            self.assertTrue(all(0<=v<=1 for v in values))
            self.assertEqual(values, sorted(values))
        self.assertIsNone(data["anomaly"]["fault_diagnosis"])

    def test_invalid_or_missing_history_is_not_filled(self):
        for mutation in ["gap", "missing", "negative", "extra"]:
            request = copy.deepcopy(self.short)
            if mutation == "gap": request["history"][-1]["second"] += 1
            if mutation == "missing": request["history"].pop()
            if mutation == "negative": request["history"][-1]["pm25"] = -1
            if mutation == "extra": request["history"][-1]["actual_future_pm10"] = 12
            self.assertEqual(self.client.post("/prototype/forecast/30s", json=request).status_code, 422)
        request = copy.deepcopy(self.hourly)
        request["history"][-1]["timestamp"] = request["history"][-2]["timestamp"]
        self.assertEqual(self.client.post("/prototype/forecast/hourly", json=request).status_code, 422)
        request = copy.deepcopy(self.hourly)
        request["station"] = "Unvalidated construction site"
        self.assertEqual(self.client.post("/prototype/forecast/hourly", json=request).status_code, 422)

    def test_simulations_require_label_and_do_not_actuate(self):
        result = self.client.post("/prototype/control", json=self.control)
        self.assertEqual(result.status_code, 200, result.text)
        data = result.json()
        self.assertTrue(data["simulation_only"])
        self.assertFalse(data["physical_actuation"])
        self.assertFalse(data["arrival_is_exact"])
        self.assertTrue(all(0<=r["duty_fraction"]<=1 for r in data["recommendations"].values()))
        for curve in data["perimeter_probability_curves"].values():
            probability = [p["simulated_probability"] for p in curve]
            self.assertEqual(probability, sorted(probability))
        bad = self.control | {"simulation_only": False}
        self.assertEqual(self.client.post("/prototype/control", json=bad).status_code, 422)

    def test_plume_and_efficiency_are_in_domain_only(self):
        request = {"simulation_only": True, "points": [{"x": 0.,"y":0.,"t":.3,"east_velocity":.2,"north_velocity":0.,"diffusivity":.02,"removal":.1}]}
        result = self.client.post("/prototype/plume", json=request)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertTrue(result.json()["simulation_only"])
        self.assertFalse(result.json()["exact_footprint"])
        request["points"][0]["x"] = 100
        self.assertEqual(self.client.post("/prototype/plume", json=request).status_code, 422)
        result = self.client.post("/prototype/efficiency", json={"simulation_only":True,"humidity":60,"temperature":30,"particle_um":6,"droplet_um":80,"wind_speed_m_s":4.2})
        self.assertEqual(result.status_code, 200)
        self.assertFalse(result.json()["measured_capture_validation"])

    def test_unavailable_models_return_503_without_fallback(self):
        with tempfile.TemporaryDirectory() as temporary, TestClient(create_app(Path(temporary))) as client:
            self.assertFalse(client.get("/health").json()["ready"])
            self.assertEqual(client.post("/prototype/forecast/30s", json=self.short).status_code, 503)
            self.assertEqual(client.post("/prototype/control", json=self.control).status_code, 503)

    def test_nonfinite_inputs_are_rejected_without_echo(self):
        request = copy.deepcopy(self.short)
        request["history"][-1]["pm10"] = float("nan")
        result = self.client.post("/prototype/forecast/30s", content=json.dumps(request),headers={"Content-Type":"application/json"})
        self.assertEqual(result.status_code,422)
        self.assertNotIn("NaN",result.text)

    def test_explicit_frontend_origin_only(self):
        origin="http://teammate.local:5173"
        with patch.dict(os.environ,{"DUSTTWIN_PROTOTYPE_ORIGINS":origin}), TestClient(create_app()) as client:
            allowed=client.get("/health",headers={"Origin":origin})
            self.assertEqual(allowed.headers.get("access-control-allow-origin"),origin)
            denied=client.get("/health",headers={"Origin":"http://unlisted.local"})
            self.assertNotIn("access-control-allow-origin",denied.headers)

    def test_changed_artifact_identity_disables_component(self):
        manifest=json.loads((ROOT/"reports/artifact-manifest.json").read_text())
        manifest["artifacts"]["outdoor-30s.joblib"]["sha256"]="0"*64
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/"artifacts").mkdir(); (root/"reports").mkdir()
            for name in manifest["artifacts"]:
                (root/"artifacts"/name).symlink_to(ROOT/"artifacts"/name)
            (root/"reports/artifact-manifest.json").write_text(json.dumps(manifest))
            prototype=Prototype(root)
            self.assertNotIn("outdoor",prototype.models)
            self.assertIn("plume",prototype.models)
            self.assertIn("outdoor",prototype.errors)


class CandidateGuards(unittest.TestCase):
    def test_old_records_cannot_enter_new_candidate(self):
        frame = pd.DataFrame({"timestamp": ["2016-01-01"], "station": ["siteA"],
                              **{c: [1.] for c in CHANNELS}})
        cfg = json.loads((ROOT/"config.json").read_text())["forecast"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"old.csv"
            frame.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,"newly collected"):
                validate_new_records(path,cfg)


if __name__ == "__main__":
    unittest.main()
