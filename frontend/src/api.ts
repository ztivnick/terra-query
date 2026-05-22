// Single source of truth for the backend base URL.

const RAW_BASE = import.meta.env.VITE_TERRA_QUERY_API_URL ?? "http://localhost:8000";
export const API_BASE_URL = RAW_BASE.replace(/\/$/, "");

export interface HealthResponse {
  status: string;
  model_id: string;
  embed_dim: number;
  experiment_id: string;
}

export interface AoiResponse {
  aoi_id: string;
  geojson: GeoJSON.FeatureCollection;
}

export interface SearchHit {
  chip_location_id: string;
  score: number;
  winning_cycle: string;
  /** [lat, lon] */
  center_wgs84: [number, number];
  /** ring of [lat, lon], closed */
  bbox_wgs84: [number, number][];
  inside_aoi: boolean;
  thumbnail_url: string;
}

export interface SearchResponse {
  query: string;
  top_k: number;
  inside_aoi_only: boolean;
  results: SearchHit[];
}

async function getJson<T>(path: string): Promise<T> {
  const r = await fetch(`${API_BASE_URL}${path}`);
  if (!r.ok) throw new Error(`${path}: ${r.status} ${r.statusText}`);
  return (await r.json()) as T;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    let detail = "";
    try {
      detail = JSON.stringify(await r.json());
    } catch {
      detail = await r.text();
    }
    throw new Error(`${path}: ${r.status} ${detail}`);
  }
  return (await r.json()) as T;
}

export function healthz(): Promise<HealthResponse> {
  return getJson<HealthResponse>("/healthz");
}

export function getAoi(): Promise<AoiResponse> {
  return getJson<AoiResponse>("/aoi");
}

export interface SearchParams {
  text: string;
  top_k?: number;
  inside_aoi_only?: boolean;
}

export function postSearch(params: SearchParams): Promise<SearchResponse> {
  return postJson<SearchResponse>("/search", {
    text: params.text,
    top_k: params.top_k ?? 10,
    inside_aoi_only: params.inside_aoi_only ?? false,
  });
}

/** Resolve a backend-relative thumbnail URL to an absolute one. */
export function absThumbUrl(url: string): string {
  return url.startsWith("http") ? url : `${API_BASE_URL}${url}`;
}
