/** One API connection shared by the example and the teammate's frontend. */
export const MODEL_ID = "hist_gb_depth3_iter100";
export const ARTIFACT_SHA256 = "d78f1b37269f72af45933e01722968fb13ed82178f6d8b3e4c5584d46cec09c7";

export class DustTwinApiError extends Error {
  constructor(status, body) {
    const detail = body?.detail ?? body?.errors?.map(item => `${item.field}: ${item.message}`).join("; ");
    super(`DustTwin API ${status}${detail ? `: ${detail}` : ""}`);
    this.name = "DustTwinApiError";
    this.status = status;
    this.body = body;
  }
}

function verifyForecast(forecast, second) {
  if (!forecast || forecast.model_id !== MODEL_ID || forecast.artifact_sha256 !== ARTIFACT_SHA256
      || !["live_inference", "saved_inference"].includes(forecast.mode)
      || forecast.units !== "ug/m3" || forecast.horizon_seconds !== 30
      || forecast.issue_time_seconds !== second || forecast.target_time_seconds !== second + 30
      || !Number.isFinite(forecast.predicted_pm10_ug_m3) || forecast.predicted_pm10_ug_m3 < 0) {
    throw new Error("Forecast identity, clock or units do not match the frozen DustTwin model");
  }
  return forecast;
}

export class DustTwinClient {
  constructor(baseUrl = "", { timeoutMs = 10000, fetchImpl = globalThis.fetch } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.timeoutMs = timeoutMs;
    this.fetchImpl = fetchImpl.bind(globalThis);
  }

  async request(path, { method = "GET", body, signal, timeoutMs = this.timeoutMs } = {}) {
    const controller = new AbortController();
    const abort = () => controller.abort(signal?.reason);
    if (signal?.aborted) abort();
    else signal?.addEventListener("abort", abort, { once: true });
    const timer = setTimeout(() => controller.abort(new DOMException("DustTwin API timed out", "TimeoutError")), timeoutMs);
    try {
      const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
        method, signal: controller.signal,
        ...(body === undefined ? {} : { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new DustTwinApiError(response.status, payload);
      if (payload === null) throw new Error("DustTwin API returned an invalid JSON response");
      return payload;
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
    }
  }

  health(options) { return this.request("/health", options); }
  recordings(options) { return this.request("/v1/replay", options); }
  evidence(options) { return this.request("/v1/evidence", options); }
  scenarios(options) { return this.request("/v1/scenarios", options); }
  savedScenario(id, options) { return this.request(`/v1/scenarios/${encodeURIComponent(id)}`, options); }
  simulate(assumptions, options = {}) {
    return this.request("/v1/simulate", { timeoutMs: 60000, ...options, method: "POST", body: assumptions });
  }

  async replay(episodeId, second, options) {
    if (!Number.isInteger(second) || second < 120) throw new Error("Replay clock must be an integer at least 120");
    const snapshot = await this.request(`/v1/replay/${encodeURIComponent(episodeId)}?second=${second}`, options);
    verifyForecast(snapshot.forecast, second);
    if (snapshot.episode_id !== episodeId || snapshot.clock_second !== second
        || !Array.isArray(snapshot.past_observations)
        || snapshot.past_observations.some(point => point.time_seconds > second)
        || (snapshot.matured_forecast && snapshot.matured_forecast.target_time_seconds > second)) {
      throw new Error("Replay response does not match the selected recording and past-only clock");
    }
    return snapshot;
  }

  async predict(request, options = {}) {
    const forecast = await this.request("/v1/predict", { ...options, method: "POST", body: request });
    if (forecast.mode !== "live_inference") throw new Error("POST /v1/predict did not execute live inference");
    return verifyForecast(forecast, request.issue_time_seconds);
  }
}
