const BASE = "";  // Vite proxy forwards to http://localhost:8000

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DocInfo {
  doc_id: string;
  document: string;  // original filename
}

export interface IngestResponse {
  document: string;
  doc_id: string;
  pages: number;
  entities: number;
  relations: number;
}

export interface FileStatusModel {
  filename: string;
  status: "pending" | "processing" | "done" | "failed";
  pages: number;
  entities: number;
  relations: number;
  doc_id?: string;
  error?: string;
}

export interface BatchIngestResponse {
  job_id: string;
  total_files: number;
}

export interface BatchStatusResponse {
  job_id: string;
  overall_status: "pending" | "processing" | "done" | "failed";
  created_at: string;
  files: FileStatusModel[];
}

export interface QueryResponse {
  question: string;
  answer: string;
  draft: string;
  sources: string[];
}

export interface KGNode {
  data: { id: string; label: string; type: string };
}

export interface KGEdge {
  data: { id: string; source: string; target: string; label: string };
}

export interface KGGraphResponse {
  nodes: KGNode[];
  edges: KGEdge[];
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

export async function batchIngest(files: File[]): Promise<BatchIngestResponse> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await fetch(`${BASE}/batch/ingest`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getBatchStatus(jobId: string): Promise<BatchStatusResponse> {
  const res = await fetch(`${BASE}/batch/status/${jobId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function openBatchStream(jobId: string): EventSource {
  return new EventSource(`${BASE}/batch/stream/${jobId}`);
}

export async function query(
  question: string,
  topK = 5,
  docId?: string,
): Promise<QueryResponse> {
  const body: Record<string, unknown> = { question, top_k: topK };
  if (docId) body.doc_id = docId;
  const res = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getKGGraph(docId?: string): Promise<KGGraphResponse> {
  const url = docId ? `${BASE}/kg/graph?doc_id=${encodeURIComponent(docId)}` : `${BASE}/kg/graph`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
