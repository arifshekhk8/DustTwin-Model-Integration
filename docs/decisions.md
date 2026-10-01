# Handoff decisions

## 1 October 2026 — Public teammate package

The user authorized a new public GitHub repository with a suitable name. Use `arifshekhk8/DustTwin-Model-Integration`. Package the trained backend for the team's existing frontend; keep its design and source with its owner. The model is small enough to track directly, so no separate artifact download is necessary.

## Preserve the trained task

Copy the frozen model/configuration/preparation and evaluation unchanged from `DustTwin-AI` at `ea5c9c5`. The learned model improves MAE over persistence but loses to the trailing mean. Preserve both baselines and all limitations. No training or new test tuning is part of this handoff.

## Separate frontend and backend

An existing website can call the Python API through an explicit origin allowlist, or a same-origin reverse proxy. Default cross-origin access is disabled until configured. The backend runs locally by default and does not require the upstream website or Node. Public repository visibility is separate from backend hosting.

## Evidence and continuity

Recorded replay remains attributed CC BY 4.0 data. Simulation remains uncalibrated software, separate from measured forecast errors. The existing daily automation belongs to the upstream project; no duplicate automation is created here. Round 1 remains software only.

## 1 October 2026 — AI-only completion and one-page report audit

The user asked to check the report against the last repository and complete only the AI model today; the teammate owns the rest. The frozen model and current API already meet their training/inference gates. Do not treat speculative future AI additions in the PDF as required functionality. Preserve its model, metadata, preparation code, configuration and revealed final test.

The report correctly identifies the artifact, PM10-only task, causal history, features and held-out metrics. Its claims about reliable surge anticipation, wind/zone mapping, Gaussian dispersion, PM2.5 ratios, exact crossing ETA, physical actuation, general water saving and competition validation exceed or conflict with this handoff. Record precise corrections in `docs/ai-report-review.md`.

Running the inherited evaluation verifier exposed a real handoff gap: it loads ignored prepared test arrays that a clone lacks. Keep this historical source unchanged and add a separate AI acceptance command that verifies the tracked replays and all exported test rows against the frozen artifact, without data download or fitting. This completes evidence verification for the delivered package; it does not validate a field system or the teammate's website.
