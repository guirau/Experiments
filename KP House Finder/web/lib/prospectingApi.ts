// Client for the local FastAPI trigger service, proxied via next.config.ts rewrite
// (/api/py/* -> http://127.0.0.1:8000/*). The service must be running:
//   poetry run uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000

const API = "/api/py";

export interface JobState {
  status: "idle" | "running" | "done" | "error";
  result: Record<string, number> | null;
  error: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface JobStatus {
  discover: JobState;
  enrich: JobState;
}

async function post(path: string, body?: unknown): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API}${path}`, {
      method: "POST",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new Error("Can't reach the trigger service — is uvicorn running on :8000?");
  }
  if (res.status === 409) throw new Error((await res.json().catch(() => ({}))).detail ?? "Already running.");
  if (!res.ok) throw new Error(`Trigger failed (HTTP ${res.status}).`);
}

export const triggerDiscover = (limit: number | null): Promise<void> => post("/discover", { limit });
export const triggerEnrich = (): Promise<void> => post("/enrich");

export async function fetchJobStatus(): Promise<JobStatus> {
  const res = await fetch(`${API}/status`);
  if (!res.ok) throw new Error(`status HTTP ${res.status}`);
  return res.json();
}
