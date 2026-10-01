import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { DustTwinClient, DustTwinApiError } from "../frontend/dusttwin-client.js";

const fixture = JSON.parse(fs.readFileSync(new URL("../examples/replay-snapshot.json", import.meta.url), "utf8"));
const response = body => new Response(JSON.stringify(body), { status: 200 });

test("adapter preserves the exact returned forecast and rejects identity/future-clock errors", async () => {
  const api = new DustTwinClient("http://backend.example", { fetchImpl: async () => response(fixture) });
  const snapshot = await api.replay(fixture.episode_id, 120);
  assert.equal(snapshot.forecast.predicted_pm10_ug_m3, fixture.forecast.predicted_pm10_ug_m3);
  for (const change of [
    value => { value.forecast.artifact_sha256 = "wrong"; },
    value => { value.forecast.target_time_seconds = 151; },
    value => { value.past_observations.push({ time_seconds: 121, pm10_ug_m3: 100 }); },
    value => { value.matured_forecast = { target_time_seconds: 150 }; },
  ]) {
    const bad = structuredClone(fixture);
    change(bad);
    const invalid = new DustTwinClient("", { fetchImpl: async () => response(bad) });
    await assert.rejects(() => invalid.replay(fixture.episode_id, 120));
  }
});

test("422 remains actionable and prediction never silently becomes saved output", async () => {
  const api = new DustTwinClient("", { fetchImpl: async () => new Response(JSON.stringify({ errors: [{ field: "body.history", message: "invalid" }] }), { status: 422 }) });
  await assert.rejects(() => api.predict(fixture.request), error => error instanceof DustTwinApiError && error.status === 422 && error.message.includes("body.history"));
  const saved = { ...fixture.forecast, mode: "saved_inference" };
  const invalid = new DustTwinClient("", { fetchImpl: async () => response(saved) });
  await assert.rejects(() => invalid.predict(fixture.request), /did not execute live/);
});

test("cancel and timeout interrupt the underlying request", async () => {
  const waitingFetch = async (url, { signal }) => new Promise((resolve, reject) => {
    if (signal.aborted) return reject(signal.reason);
    signal.addEventListener("abort", () => reject(signal.reason), { once: true });
  });
  const api = new DustTwinClient("", { timeoutMs: 20, fetchImpl: waitingFetch });
  await assert.rejects(() => api.health(), error => error.name === "TimeoutError");
  const controller = new AbortController();
  const pending = api.health({ signal: controller.signal, timeoutMs: 1000 });
  controller.abort();
  await assert.rejects(() => pending, error => error.name === "AbortError");
});
