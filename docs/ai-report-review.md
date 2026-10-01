# AI model completion and report review

1 October 2026, Asia/Dhaka. Reviewed `DustTwin_AI_Integration_One_Page_Summary.pdf` against the latest public model handoff, initially at commit `97a1d15`. The PDF is a description to check, not an instruction to implement its research roadmap.

**The current laboratory PM10 model and its backend are complete for the Round 1 software handoff. The PDF is partly accurate, but its control, hardware and performance language needs correction.** No further fitting is required for this frozen model. The teammate connects it to the team's frontend and arranges backend hosting if the site is online.

## What matches the repository

| Report statement | Verified result |
|---|---|
| Trained HistGradientBoosting model | True: `hist_gb_depth3_iter100`, 100 iterations, maximum depth 3. The actual 54,679-byte artifact is tracked in Git. |
| 120-second history / 121 one-second snapshots | True: inclusive history, with causal latest-past observations no more than 1.5 seconds old. These are gridded snapshots, not necessarily 121 distinct native sensor observations. |
| 16 PM10 features | True: seven lags plus mean, population standard deviation and slope for three inclusive windows. |
| PM10 forecast at +30 seconds | True: a point forecast for the latest observed laboratory OPC-N3 snapshot at issue+30. This does not promise 30 seconds of advance warning. |
| MAE 88.405 / RMSE 179.272 micrograms per cubic metre | True on 15,065 overlapping windows from three related held-out recordings. The model has the lowest pooled RMSE among itself and the two published baselines. |
| Measured-data replay and live inference | True: `/v1/replay/{episode_id}` uses the real model when ready; `/v1/predict` validates and executes the model. Saved replay is explicitly labelled. |
| Earlier actual revealed from clock 150 onward | True for eligible clocks: only the forecast issued at clock-30 is compared with the recorded observation now available. |
| PM2.5, wind, humidity and temperature are not forecast features | True. The model is PM10-only. Their presence in a separate frontend is not established by this backend handoff. |
| Dataset authors and reuse terms | Correct: Askarov and Choi (2024), Mendeley V1, DOI `10.17632/7f22n9v7hp.1`, CC BY 4.0. [Original dataset](https://data.mendeley.com/datasets/7f22n9v7hp/1); [attribution](../NOTICE.md). |

## Statements to correct before presenting

| Report wording | Accurate replacement / reason |
|---|---|
| "Empirically proven accuracy" / "robust" | Preliminary laboratory evaluation with mixed baseline results. MAE improves 7.62% over persistence but is 8.39% worse than trailing mean. One test recording is a material failure. No independent field validation or calibrated uncertainty. |
| "AI forecasts sudden surge" as reliable proactive warning | The descriptive model experiment matches 13 of 18 correlated threshold runs with 15 false alerts. Only one of three first onsets receives an advance warning. A future endpoint is not a guaranteed alert lead. |
| "NW wind selects A and D" | Under the handoff's A=north, B=east, C=south, D=west and meteorological wind-from convention, wind from NW travels SE and exposes B and C. Actual activation also depends on modeled risk and switching rules. |
| "Real example" at NW 4.2 m/s saves 93% water | No matching measured/control experiment supports this combined example. Water figures are scenario-specific software simulations. The frozen east case uses 1.842 L predictive versus 16.000 L continuous over 480 s, about 88.49% less, with 86 s above the simulation setting in each. This is not field water saving. |
| "Plume simulated via Gaussian physics" | This handoff uses cosine-squared directional weighting, an assumed transport gain, distance/speed transport delay and a first-order boundary response. It does not implement a Gaussian plume model. |
| "PM2.5 values modeled through deterministic ratios" | This handoff does not provide a PM2.5 output. Do not imply PM2.5 validation from the PM10 model or invent a ratio-based concentration. |
| "Humidity/temperature are measured physical inputs" | Not provided by the current model API. Wind values in the site cases are configured simulation inputs. Field environmental streaming is not part of this handoff. |
| "Pumps, relays, solenoids and PWM run" | Simulated zone commands and flow accounting exist. Physical hardware, PWM drivers and sensor streaming were not built or tested. |
| "Crossing ETA = distance / wind speed" | Distance/speed contributes to assumed transport delay. Simulated threshold ETA comes from a forward boundary trajectory. The measured PM10 point forecast explicitly returns `crossing_eta_seconds: null`. |
| "Always displays OFFLINE when backend unavailable" | A network loss clears live values in the supplied example. A running saved-only backend can serve previously computed, explicitly labelled forecasts. Those are genuine saved outputs, not fabricated live predictions. |
| "Live HUD trend status" | The handoff returns current/predicted values and features, and the example shows baselines and matured actual. The teammate's specific dashboard/HUD and trend labels have not been integrated or tested here. |
| "Competition-proven" / "safe site mitigation" | No competition outcome or physical mitigation trial has been supplied. Describe a verified software prototype and its limits. |

The eight future additions in the PDF are research options, not outstanding requirements for the current model: environmental multi-input training, PINNs, learned spraying, particle/droplet efficiency, crossing probabilities, edge deployment, online retraining and uncertainty/fault modelling. They need suitable data, separately declared evaluation and, where applicable, physical trials. Merely adding code would not complete or validate them today. Preserve the revealed final test; future model selection needs an untouched evaluation set.

## AI acceptance completed today

The old `scripts/verify_evaluation.py` remains unchanged historical source reproduction: it requires `data/processed/.../test.npz`. That file is intentionally absent from a normal handoff clone. Use the new check for acceptance of the delivered AI:

```sh
.venv/bin/python scripts/verify_ai.py
.venv/bin/python -m unittest discover -s tests -v
```

`verify_ai.py` checks the 75 frozen upstream files, artifact/environment/feature/configuration identity, five fixtures, every included replay history and target timestamp, all 27,178 saved predictions and mean baselines, and all 15,065 test trace rows. It independently recalculates pooled/per-recording error metrics from the exported trace and recomputes descriptive warning counts using the frozen rules. It verifies live replay/direct API agreement, clock-only actual reveal, unavailable-model rejection and labelled saved replay. No raw downloads, prepared training arrays, retraining or network access are needed.

Actual results are in [AI readiness](../reports/integration/ai-readiness.json) and [AI fresh-clone verification](../reports/integration/ai-fresh-clone.json). Commit `8decca9` was cloned directly from GitHub, a new environment installed the pinned packages, and all AI checks plus 28 Python tests passed without raw/prepared training data. Three adapter checks, three Chromium journeys and TypeScript checking also passed in the main checkout against the unchanged connection example. [Integration checks](../reports/integration/checks.json) retain the earlier fresh-clone setup. Scope remains laboratory software, and macOS arm64 / Python 3.14.6 is the verified live platform.

## Handoff boundary

AI owner: provide the frozen artifact, strict input/API contract, baseline/evaluation evidence, recorded demonstration and repeatable acceptance check. All are present and verified.

Teammate: follow [integration.md](integration.md), connect the existing frontend, configure its API address and allowed origin, and deploy Python if an online demo is needed. Repository visibility does not host the API. Physical site control, environmental ingestion and future research are separate work.

Suggested accurate report sentence: "DustTwin demonstrates a trained PM10 model that forecasts a laboratory OPC-N3 observation 30 seconds ahead from 120 seconds of causal history. Its held-out errors are MAE 88.405 and RMSE 179.272 micrograms per cubic metre, with mixed baseline performance. Site transport, zone commands and water comparisons are separate, uncalibrated software simulations; hardware and field effectiveness remain unvalidated."
