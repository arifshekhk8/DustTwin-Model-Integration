"""Shared experiment paths, serialization, checksums, and bounded CPU use."""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
CONFIG = json.loads((ROOT / "config.json").read_text())
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
RAW = REPO / "data/raw/experimental"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def error_metrics(actual, predicted):
    import numpy as np
    delta = np.asarray(actual) - np.asarray(predicted)
    return {"mae": float(np.abs(delta).mean()),
            "rmse": float(np.sqrt(np.square(delta).mean()))}
