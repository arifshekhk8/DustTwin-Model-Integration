# DustTwin AI prototype - now and remaining work

**1 October 2026: trained measured forecasts plus explicitly simulated control.**
New files are under `experimental/`; the original laboratory model is unchanged.
Physical pumps, a real construction site and general effectiveness are unvalidated.

| Available now | Meaning |
|---|---|
| PM2.5 / PM10 +30s | New model uses 120s of outdoor PM2.5/PM10, temperature and RH history. Tested on 3,882 windows from a separate recording. |
| Weather-aware forecasts | Separate real hourly models forecast PM at +1/3/6h and wind/temperature/derived RH at +1h. Tested on 83,032 station-hours. |
| Plume AI | Trained PINN approximates idealized advection/diffusion/removal; normalized simulation outputs. |
| Learned spraying/intensity | Double-DQN chooses OFF/50%/100% zone duties; learned toy particle/droplet capture model. |
| Probabilities / arrival | Learned hourly exceedance curves; simulated zone crossing curves and arrival estimates. No exact field ETA. |
| Uncertainty / anomalies | Residual intervals and learned anomaly flags; weak coverage/detection is reported. No sensor-clog diagnosis. |
| Exports / future retraining | Float ONNX and 12.2 KB int8 controller; guarded fresh-data candidate-training workflow. Desktop tested; no board deployment or online field feed. |

**Measured comparisons (MAE, ug/m3; lower is better):**

| Forecast | Selected forecast | No-change baseline |
|---|---:|---:|
| Outdoor PM10 +30s | 6.310 | 7.300 |
| Outdoor PM2.5 +30s | 1.412 | 1.340 |
| Hourly PM10 +1h | 14.725 | 15.377 |
| Hourly PM2.5 +1h | 8.837 | 9.530 |

These datasets differ from the old laboratory task, so absolute errors do not
prove an improvement over that original model. Short PM2.5 loses on MAE and
slightly wins on RMSE. Its nominal 90% interval covered only **67.6%**. Short RH
lost persistence; short temperature selected persistence on validation. The
anomaly model caught only **32.1%** of injected extreme-value cases.

Across 40 simulated 10-minute cases, DQN averaged **4.896 L / 23.625 zone-seconds
above the illustrative threshold**; reactive **2.707 L / 70.500**; continuous
**20.000 L / 20.500**. DQN used less water than continuous and reduced exposure
versus reactive, but **lost the combined reward to reactive**. Reactive is default.
These are assumed-flow simulation results, not measured field savings.

**Remaining:** independent site tests; better short PM2.5/RH and calibration;
measured plume/treatment/droplet/fault data; physical hardware; target-device
performance; field streaming and reviewed retraining promotion. Every research
item is not yet validated.

**Connect:** separate Python 3.13 environment, `experimental/requirements.txt`,
then `experimental/serve.py` on port 8011. Set the exact frontend origin and use
`experimental/examples/`. Keep hourly and simulated outputs labelled. Missing
models return 503 without invented values. Detailed evidence: `experimental/reports/`.

Data: [Askarov & Choi 2024](https://data.mendeley.com/datasets/7f22n9v7hp/1),
[Chen 2017 / UCI](https://archive.ics.uci.edu/dataset/501/beijingmultisiteairqualitydata), CC BY 4.0.
