# DustTwin experimental AI prototype

**Trained models are included.** The user approved measured forecasts with
simulated control on 1 October 2026. Start with [the teammate report](teammate-report.md).
The original laboratory `/v1` model remains a separate, unchanged handoff.

## Run the supplied models

Use a separate **Python 3.13** environment. Python 3.13.5 / macOS arm64 is verified;
the original service uses Python 3.14. From the repository root:

```sh
python3.13 -m venv experimental/.venv
experimental/.venv/bin/python -m pip install -r experimental/requirements.txt
experimental/.venv/bin/python experimental/verify.py --http
experimental/.venv/bin/python -m unittest discover -s experimental/tests -v
export DUSTTWIN_PROTOTYPE_ORIGINS='http://localhost:5173'
experimental/.venv/bin/python experimental/serve.py
```

Set the exact frontend origin. `*` is rejected. The API runs at
`http://127.0.0.1:8011`; `/health` checks each model and `/docs` provides schemas.
No raw data download or fitting is required. GitHub does not host the API.

| Endpoint | Input / result |
|---|---|
| `/prototype/forecast/30s` | POST 121 consecutive one-second PM2.5, PM10, temperature and RH snapshots; returns outdoor +30s forecasts, baselines and intervals |
| `/prototype/forecast/hourly` | POST 25 hourly PM/weather snapshots from a listed Beijing station; returns PM +1/3/6h, weather +1h, exceedance probabilities and anomaly flag |
| `/prototype/plume` | POST dimensionless coordinates/time/transport parameters; returns normalized PINN concentrations |
| `/prototype/efficiency` | POST assumed particle/droplet size and weather; returns learned toy capture fraction |
| `/prototype/control` | POST simulated source/zone readings and weather; returns duty/flow recommendations and simulated crossing/arrival curves |
| `/prototype/evidence` | GET actual capability map and evaluation evidence |

Copy [complete request examples](examples/) into your integration. Saved response
examples are labelled `saved_verification_response_not_live`; do not show them as
live values when offline. Unavailable components return 503. Missing, invalid,
duplicate-clock or non-finite histories return 422. Caller observations are not
independently authenticated. Units are returned with forecasts: PM `ug/m3`,
temperature degrees C, RH percent and wind m/s.

The outdoor model has **no wind channel**. Hourly weather was never joined at
invented second-scale timestamps. Do not claim a 30-second wind-aware forecast.
Hourly humidity was derived from dewpoint. New stations need fresh training and
calibration. Wind components point toward east/north; control takes meteorological
wind **from** degrees. With A north/B east/C south/D west, NW wind travels toward B/C.

The short model returns its learned forecast and a validation-selected blend with
persistence. Short temperature selected persistence; short RH lost that baseline
on test. Intervals are residual-calibrated, **not Bayesian**. Short PM2.5's nominal
90% interval covered only 67.6% on test. An anomaly flag does not diagnose a clog.

Control defaults to reactive. Choose `policy: "dqn"` for the learned controller.
DQN reduced exposure but used more water than reactive and lost the combined reward.
All plume/efficiency/control requests require `simulation_only: true`. No pump is connected.

## Data and evidence

[Provenance](reports/datasets.json) and [NOTICE](NOTICE.md) record authors, reuse
terms, checksums and transformations. Outdoor Day 1 has separate train/validation/
calibration blocks with gaps; Day 2 is test. Hourly 2013-2014 trains, separate halves
of 2015 validate/calibrate, and 2016-February 2017 tests, with boundary gaps. Weights
and calibration were fixed before test scoring. Overlapping windows are correlated.
No original laboratory final-test data was used for fitting.

Synthetic equations/seeds are in [config.json](config.json) and
[simulation_models.py](simulation_models.py). The PINN has an actual automatic-
derivative advection/diffusion/reaction residual loss plus analytical-pulse labels.
Its units are dimensionless. Spray efficiency labels use a toy response. DQN trains
in a separate directional control environment; the PINN is not a calibrated part
of that environment. Separate seeds and changed-efficiency stress tests verify
assumptions, without supplying field evidence.

[Verification](reports/verification.json) checks artifacts, trace metrics, five
actual HTTP endpoints and ONNX parity. [Source reproduction](reports/source-reproduction.json)
recomputed all new test forecasts without fitting and checked all 75 frozen files.
An actual [fresh GitHub clone](reports/fresh-clone-verification.json) with a newly
installed pinned environment passed all 14 tests, five HTTP endpoints, artifact/
trace/ONNX checks and 75 original file identities, without raw/prepared data.
Use `source_audit.py` to repeat that stronger check; it needs pinned raw downloads.

To reproduce this fixed fitting protocol, use an empty scratch copy:
`train_forecasts.py`, `train_simulation.py`, then `finalize.py`. Scripts refuse to
overwrite fitted artifacts. A changed/new research experiment needs a newly untouched
evaluation period. Bulk raw data and generated transitions stay ignored.
Delivered hashes/inference are checked; bitwise cross-platform training is not promised.

## Future data and deployment

[retrain_candidate.py](retrain_candidate.py) accepts new collected hourly CSV and
explicit split dates. Its `--help` lists arguments. It checks schema/units/clock/
freshness, declares a protocol before fitting and writes a new version directory;
it never replaces live weights. The old-data guard is tested. No field candidate
or active continuous retraining feed exists yet; promotion needs reviewed evidence
and a new untouched test. Exclude all previously published test labels from future
training/selection.

Float ONNX exports execute on desktop CPU. The 12,157-byte int8 policy changed 4.9%
of actions on the random-state check; the service uses full precision. A chosen
board/runtime and memory/latency/power tests are still needed. Physical validation
requires aligned site dust/weather/position, nozzle duty, measured flow, treatment
response, particle/droplet and fault records. Reliable faults, exact field arrival,
physical actuation and general water-saving claims remain unvalidated.
