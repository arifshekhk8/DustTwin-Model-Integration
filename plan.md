# Integration handoff plan

User request, 1 October 2026: create a public repository with every file the teammate needs to use the existing trained AI in his DustTwin frontend.

1. Package the actual trained model, shared features, strict inference service, pinned requirements, attributed recorded replay and evaluation. Add explicit frontend-origin configuration. Verify reload and API behavior.
2. Supply framework-independent browser/TypeScript adapters, a runnable connection example, sample request and response, and a step-by-step integration guide. Verify a real browser calling the model across origins.
3. Publish meaningful commits, verify from a fresh GitHub clone, and record readiness and exact next steps in `following.md`.

The teammate connects these outputs to his existing components. Publishing this repository does not deploy the backend or edit his frontend. A hosting address is chosen when the team deploys. No new training or hardware step is required for the current software demo.
