import { DustTwinClient } from "../frontend/dusttwin-client.js";

const byId = id => document.getElementById(id);
let api, index, snapshot;
let currentRequest;
let generation = 0;

function startRequest() {
  currentRequest?.abort();
  currentRequest = new AbortController();
  generation += 1;
  return { signal: currentRequest.signal, generation };
}

function clearOutput() {
  snapshot = null;
  for (const id of ["prediction", "mode", "current", "clock", "baselines", "actual"]) byId(id).textContent = "—";
  byId("raw").textContent = "No current response.";
  byId("predict").disabled = true;
}

function status(text, state) {
  byId("status").textContent = text;
  byId("status").dataset.state = state;
}

function render(value) {
  snapshot = value;
  const forecast = value.forecast;
  byId("prediction").textContent = `${forecast.predicted_pm10_ug_m3.toFixed(3)} µg/m³`;
  byId("mode").textContent = forecast.mode === "live_inference" ? "Live trained model" : "Saved inference — previously computed";
  byId("current").textContent = `${forecast.current_pm10_ug_m3.toFixed(3)} µg/m³`;
  byId("clock").textContent = `${forecast.issue_time_seconds}s → ${forecast.target_time_seconds}s (+30s)`;
  byId("baselines").textContent = `${forecast.baselines.persistence_pm10_ug_m3.toFixed(3)} / ${forecast.baselines.trailing_mean_pm10_ug_m3.toFixed(3)} µg/m³`;
  const matured = value.matured_forecast;
  byId("actual").textContent = matured
    ? `Earlier forecast ${matured.predicted_pm10_ug_m3.toFixed(3)}; recorded target ${matured.actual_pm10_ug_m3.toFixed(3)} µg/m³ at ${matured.target_time_seconds}s`
    : "No earlier forecast has matured at this clock. Current forecast's target is still hidden.";
  byId("scope").textContent = forecast.scope;
  byId("attribution").textContent = value.attribution;
  byId("raw").textContent = JSON.stringify(value, null, 2);
  byId("predict").disabled = forecast.mode !== "live_inference";
  status(`Forecast received for ${value.episode_id} at ${value.clock_second}s.`, forecast.mode);
}

async function loadForecast() {
  const request = startRequest();
  clearOutput();
  status("Requesting the selected clock…", "loading");
  try {
    const value = await api.replay(byId("episode").value, Number(byId("second").value), request);
    if (request.generation === generation && !request.signal.aborted) render(value);
  } catch (error) {
    if (request.generation === generation && !request.signal.aborted) status(error.message, "unavailable");
  }
}

function selectRecording() {
  const item = index.episodes.find(item => item.episode_id === byId("episode").value);
  byId("second").min = item.first_issue_second;
  byId("second").max = item.last_issue_second;
  byId("second").value = item.first_issue_second;
}

byId("connect-form").addEventListener("submit", async event => {
  event.preventDefault();
  const request = startRequest();
  clearOutput();
  for (const id of ["episode", "second", "load", "advance"]) byId(id).disabled = true;
  byId("health").textContent = "Checking backend…";
  status("Connecting…", "loading");
  api = new DustTwinClient(byId("base-url").value);
  try {
    const [health, recordings] = await Promise.all([api.health(request), api.recordings(request)]);
    if (request.generation !== generation || request.signal.aborted) return;
    index = recordings;
    byId("health").textContent = health.ready ? "Backend ready — live trained model" : `Saved-only backend: ${health.reason}`;
    byId("episode").replaceChildren(...index.episodes.map(item => {
      const option = document.createElement("option"); option.value = item.episode_id; option.textContent = item.label; return option;
    }));
    for (const id of ["episode", "second", "load", "advance"]) byId(id).disabled = false;
    selectRecording();
    await loadForecast();
  } catch (error) {
    if (request.generation === generation && !request.signal.aborted) {
      byId("health").textContent = "Backend unavailable";
      status(`${error.message}. Check the backend address and allowed frontend origin.`, "unavailable");
    }
  }
});

byId("forecast-form").addEventListener("submit", event => { event.preventDefault(); loadForecast(); });
byId("episode").addEventListener("change", () => { selectRecording(); loadForecast(); });
byId("second").addEventListener("input", () => { startRequest(); clearOutput(); status("Clock changed — request its forecast.", "loading"); });
byId("advance").addEventListener("click", () => {
  byId("second").value = Math.min(Number(byId("second").max), Number(byId("second").value) + 30);
  loadForecast();
});
byId("predict").addEventListener("click", async () => {
  const previous = snapshot;
  if (!previous) return;
  const request = startRequest();
  clearOutput();
  status("Executing POST /v1/predict with the returned history…", "loading");
  try {
    const forecast = await api.predict(previous.request, request);
    if (request.generation === generation && !request.signal.aborted) {
      render({ ...previous, forecast });
      status("POST /v1/predict executed the included trained artifact.", "live_inference");
    }
  } catch (error) {
    if (request.generation === generation && !request.signal.aborted) status(error.message, "unavailable");
  }
});
