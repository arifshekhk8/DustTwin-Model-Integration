# Integration handoff plan

## Repository completion, 1 October 2026

All authorized original-model and experimental software milestones are complete. The trained artifacts, APIs, connection examples, honest evaluation, reproducibility checks and teammate reports are tracked and published. Final acceptance verifies the unchanged laboratory evidence and all eight prototype artifacts, ONNX parity and five live HTTP endpoints. The remaining checkpoint publishes documentation consistency fixes and confirms a clean local/GitHub state. The teammate owns frontend integration/hosting; physical and target-device validation remain future work requiring their own authorization/data. No additional model fitting is required to use this package.

## Newly authorized experimental expansion, 1 October 2026

The user requested dataset discovery/building and training for the broader AI
roadmap, then explicitly selected **measured forecasts with simulated control**.
This new scope supersedes the earlier pause on changes for feasibility review.
Preserve the frozen model and all 75 upstream files. Add the experiment separately
under `experimental/`, with a new untouched temporal test and distinct synthetic
test seeds. Predeclare splits/configuration before evaluation. Train and verify
real-data multi-input forecasts and clearly labelled simulated plume/spraying
models; supply model artifacts, an optional separate API, evidence and a teammate
guide. Report unsuccessful comparisons and unfinished physical/edge work plainly.

**Complete:** fitted artifacts and five prototype API endpoints are published at
`2f91d56`. All 14 tests, actual HTTP inference, raw-source forecast reproduction,
ONNX checks and all 75 frozen file hashes pass. A direct fresh GitHub clone with
a newly installed separate environment passes every prototype check without
training data/refitting. Its actual evidence is
`experimental/reports/fresh-clone-verification.json`. The updated one-page brief
and teammate guide accurately report measured/simulated results and limitations.
Physical/device/continuous-field validation is outside the accepted prototype.

User request, 1 October 2026: create a public repository with every file the teammate needs to use the existing trained AI in his DustTwin frontend.

1. **Complete:** package the actual trained model, shared features, strict inference service, pinned requirements, attributed recorded replay and evaluation. Add explicit frontend-origin configuration. Verify reload and API behavior.
2. **Complete:** supply framework-independent browser/TypeScript adapters, a runnable connection example, sample request and response, and a step-by-step integration guide. Verify a real browser calling the model across origins.
3. **Complete:** publish meaningful commits, verify from a fresh GitHub clone, and record readiness and exact next steps in `following.md`. Verification covers runtime commit `6c924e4`; the final checkpoint adds the readiness record and continuation instructions.

The teammate connects these outputs to his existing components. Publishing this repository does not deploy the backend or edit his frontend. A hosting address is chosen when the team deploys. No new training or hardware step is required for the current software demo.

## AI-only report review and completion, 1 October 2026

The user requested verification of the one-page integration report and completion of the AI model part today, leaving the rest to the teammate. The model itself is already trained and accepted; the report's future research options are not current completion gates.

4. **Complete:** inspect the report against source/artifact/evidence, document supported claims and corrections in `docs/ai-report-review.md`, and preserve the frozen model/test.
5. **Complete and published:** close the clone's evidence-check gap with `scripts/verify_ai.py`, using included replays and test traces instead of absent prepared arrays. All 27,178 saved predictions and 15,065 test trace rows verify, with causal timestamps, reported errors/warnings and live/saved API behavior. All 28 Python tests, three adapter checks, three browser journeys and TypeScript checks pass. Runtime commit `8decca9` was cloned directly from GitHub; a newly installed pinned environment passed all AI checks and 28 Python tests without raw/prepared training arrays. Actual evidence is in `reports/integration/ai-readiness.json` and `ai-fresh-clone.json`. No required AI-only software work remains.
