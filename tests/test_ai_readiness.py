"""The handoff check must reject flattering, shifted or incomplete evidence."""

import copy
import csv
import gzip
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_ai import ForecastModel, ReplayStore, verify, verify_metrics, verify_replays, verify_trace


class AIReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ForecastModel(ROOT)
        cls.store = ReplayStore(ROOT)
        cls.verified = verify_replays(cls.store, cls.model)
        cls.report = json.loads((ROOT / "reports/evaluation/test-metrics.json").read_text())
        with gzip.open(ROOT / cls.report["trace_file"], "rt") as stream:
            cls.rows = list(csv.DictReader(stream))

    def test_complete_verification_needs_no_raw_or_prepared_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("configs", "models", "src", "demo", "reports", "services"):
                shutil.copytree(ROOT / name, root / name, ignore=shutil.ignore_patterns("__pycache__"))
            manifest = json.loads((ROOT / "models/upstream-provenance.json").read_text())
            for item in manifest["unchanged_upstream_files"]:
                destination = root / item["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / item["path"], destination)
            self.assertFalse((root / "data/raw").exists())
            self.assertFalse((root / "data/processed").exists())
            result = verify(root)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["verified_test_rows"], 15065)
            self.assertEqual(result["recomputed_saved_forecasts"], 27178)

    def test_invented_saved_forecast_is_rejected(self):
        store = copy.deepcopy(self.store)
        store.episodes["lab_e3_drill10"]["saved_forecast_pm10_ug_m3"][0] += 1
        with self.assertRaisesRegex(ValueError, "saved predictions"):
            verify_replays(store, self.model)

    def test_shifted_target_is_rejected(self):
        rows = [dict(self.rows[0]), *self.rows[1:]]
        rows[0]["target_second"] = str(int(rows[0]["target_second"]) + 1)
        with self.assertRaisesRegex(ValueError, r"issue \+30"):
            verify_trace(rows, self.verified)

    def test_missing_and_duplicate_test_forecasts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "omits"):
            verify_trace(self.rows[1:], self.verified)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            verify_trace([self.rows[0], *self.rows], self.verified)

    def test_understated_pooled_or_recording_error_is_rejected(self):
        for metrics in ("models", "by_recording"):
            report = copy.deepcopy(self.report)
            target = report[metrics] if metrics == "models" else report[metrics]["lab_e4_drill10"]
            target["selected_model"]["mae_ug_m3"] -= 1
            with self.assertRaisesRegex(ValueError, "reported metric differs"):
                verify_metrics(self.rows, report)


if __name__ == "__main__":
    unittest.main()
