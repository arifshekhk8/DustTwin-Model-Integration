export const MODEL_ID: "hist_gb_depth3_iter100";
export const ARTIFACT_SHA256: string;
export interface RequestOptions { signal?: AbortSignal; timeoutMs?: number }
export interface Measurement { time_seconds: number; observation_time_seconds: number; pm10_ug_m3: number }
export interface PredictionRequest {
  task_id: "construction_pm10_30s_v1";
  monitor_id: "OPC-N3";
  clock_type: "elapsed_seconds_per_recording";
  units: "ug/m3";
  horizon_seconds: 30;
  issue_time_seconds: number;
  history: Measurement[];
}
export interface Forecast {
  mode: "live_inference" | "saved_inference";
  model_id: string;
  artifact_sha256: string;
  issue_time_seconds: number;
  target_time_seconds: number;
  horizon_seconds: 30;
  units: "ug/m3";
  predicted_pm10_ug_m3: number;
  current_pm10_ug_m3: number;
  baselines: { persistence_pm10_ug_m3: number; trailing_mean_pm10_ug_m3: number };
  scope: string;
  snapshot_id?: string;
  task_id?: string;
  dataset_doi?: string;
  monitor_id?: string;
  input_quality?: { snapshots: 121; maximum_observation_age_seconds: number };
  features?: Record<string, number>;
  inference_milliseconds?: number;
  demo_setting_ug_m3?: number;
  crossing_status?: "already_exceeded" | "endpoint_exceeds_setting" | "endpoint_below_setting";
  crossing_eta_seconds?: null;
}
export interface MaturedForecast {
  issue_time_seconds: number;
  target_time_seconds: number;
  predicted_pm10_ug_m3: number;
  actual_pm10_ug_m3: number;
  target_observation_time_seconds: number;
}
export interface ReplaySnapshot {
  episode_id: string;
  partition: "validation" | "test";
  clock_second: number;
  request: PredictionRequest;
  forecast: Forecast;
  matured_forecast: MaturedForecast | null;
  past_observations: { time_seconds: number; pm10_ug_m3: number }[];
  attribution: string;
}
export interface Recording {
  episode_id: string; partition: "validation" | "test"; group: number; label: string;
  file: string; sha256: string; last_second: number;
  first_issue_second: number; last_issue_second: number; suggested_start_second: number;
}
export interface ReplayIndex {
  schema_version: number; task_id: string; model_id: string; artifact_sha256: string;
  attribution: string; source_url: string; license_url: string; episodes: Recording[];
}
export interface Health {
  ready: boolean; mode: "live_inference" | "saved_inference";
  model_id: string; artifact_sha256: string; task_id: string; monitor_id: string;
  horizon_seconds: 30; grid_interval_seconds: 1; reason: string | null;
}
export type ScenarioId = "low-risk" | "east" | "diagonal" | "wind-shift" | "data-loss";
export interface SimulationRequest {
  scenario_id: ScenarioId;
  source_scale?: number; wind_from_degrees?: number | null; wind_speed_metres_second?: number;
  flow_litres_minute_per_zone?: number; mist_source_fraction_removed?: number;
}
export class DustTwinApiError extends Error { status: number; body: unknown; constructor(status: number, body: unknown) }
export class DustTwinClient {
  baseUrl: string;
  constructor(baseUrl?: string, options?: { timeoutMs?: number; fetchImpl?: typeof fetch });
  health(options?: RequestOptions): Promise<Health>;
  recordings(options?: RequestOptions): Promise<ReplayIndex>;
  replay(episodeId: string, second: number, options?: RequestOptions): Promise<ReplaySnapshot>;
  predict(request: PredictionRequest, options?: RequestOptions): Promise<Forecast>;
  evidence(options?: RequestOptions): Promise<Record<string, unknown>>;
  scenarios(options?: RequestOptions): Promise<Record<string, unknown>>;
  savedScenario(id: ScenarioId, options?: RequestOptions): Promise<Record<string, unknown>>;
  simulate(assumptions: SimulationRequest, options?: RequestOptions): Promise<Record<string, unknown>>;
}
