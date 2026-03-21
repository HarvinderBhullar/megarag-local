import React, { useEffect, useRef, useState } from "react";
import { openBatchStream, FileStatusModel, getBatchStatus, DocInfo } from "../api/client";

interface Props {
  jobId: string;
  onComplete?: () => void;
  onDocIngested?: (doc: DocInfo) => void;
}

interface SSEFileEvent {
  job_id: string;
  filename: string;
  doc_id?: string;
  index?: number;
  total?: number;
  pages?: number;
  entities?: number;
  relations?: number;
  error?: string;
  overall_status?: string;
  failed?: number;
  total_files?: number;
}

export default function BatchProgress({ jobId, onComplete, onDocIngested }: Props) {
  const [files, setFiles] = useState<FileStatusModel[]>([]);
  const [overall, setOverall] = useState<"pending" | "processing" | "done" | "failed">("pending");
  const [connected, setConnected] = useState(false);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    // Load initial status via polling in case we missed some events
    getBatchStatus(jobId).then((s) => {
      setFiles(s.files);
      setOverall(s.overall_status);
    });

    const es = openBatchStream(jobId);
    esRef.current = es;
    setConnected(true);

    es.addEventListener("file_start", (e: MessageEvent) => {
      const data: SSEFileEvent = JSON.parse(e.data);
      setFiles((prev) =>
        prev.map((f) =>
          f.filename === data.filename ? { ...f, status: "processing" } : f
        )
      );
    });

    es.addEventListener("file_done", (e: MessageEvent) => {
      const data: SSEFileEvent = JSON.parse(e.data);
      setFiles((prev) =>
        prev.map((f) =>
          f.filename === data.filename
            ? {
                ...f,
                status: "done",
                doc_id: data.doc_id,
                pages: data.pages ?? 0,
                entities: data.entities ?? 0,
                relations: data.relations ?? 0,
              }
            : f
        )
      );
      // Notify App so it can track this doc for scoped queries / graph views
      if (data.doc_id) {
        onDocIngested?.({ doc_id: data.doc_id, document: data.filename });
      }
    });

    es.addEventListener("file_error", (e: MessageEvent) => {
      const data: SSEFileEvent = JSON.parse(e.data);
      setFiles((prev) =>
        prev.map((f) =>
          f.filename === data.filename
            ? { ...f, status: "failed", error: data.error }
            : f
        )
      );
    });

    es.addEventListener("batch_done", (e: MessageEvent) => {
      const data: SSEFileEvent = JSON.parse(e.data);
      setOverall((data.overall_status as typeof overall) ?? "done");
      setConnected(false);
      es.close();
      onComplete?.();
    });

    es.onerror = () => {
      setConnected(false);
    };

    return () => {
      es.close();
    };
  }, [jobId]);

  const done = files.filter((f) => f.status === "done").length;
  const total = files.length;

  const statusIcon = (s: FileStatusModel["status"]) => {
    if (s === "done") return "✅";
    if (s === "failed") return "❌";
    if (s === "processing") return "⏳";
    return "🕐";
  };

  const statusColor = (s: FileStatusModel["status"]) => {
    if (s === "done") return "var(--success)";
    if (s === "failed") return "var(--error)";
    if (s === "processing") return "var(--accent)";
    return "var(--text-muted)";
  };

  return (
    <div
      style={{
        maxWidth: 640,
        margin: "0 auto",
        background: "var(--surface)",
        borderRadius: "var(--radius)",
        padding: 20,
      }}
    >
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div>
          <span style={{ fontWeight: 700, fontSize: 15 }}>
            Batch Job — {jobId.slice(0, 8)}…
          </span>
          <span
            style={{
              marginLeft: 10,
              fontSize: 12,
              padding: "2px 8px",
              borderRadius: 99,
              background:
                overall === "done"
                  ? "rgba(76,175,125,0.15)"
                  : overall === "failed"
                  ? "rgba(245,101,101,0.15)"
                  : "rgba(108,142,245,0.15)",
              color:
                overall === "done"
                  ? "var(--success)"
                  : overall === "failed"
                  ? "var(--error)"
                  : "var(--accent)",
            }}
          >
            {overall}
          </span>
        </div>
        <span style={{ color: "var(--text-muted)", fontSize: 13 }}>
          {done}/{total} files
        </span>
      </div>

      {/* Overall progress bar */}
      {total > 0 && (
        <div
          style={{
            height: 6,
            background: "var(--surface2)",
            borderRadius: 3,
            marginBottom: 16,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${(done / total) * 100}%`,
              background: overall === "failed" ? "var(--error)" : "var(--accent)",
              borderRadius: 3,
              transition: "width 0.4s ease",
            }}
          />
        </div>
      )}

      {/* Per-file rows */}
      {files.map((f) => (
        <div
          key={f.filename}
          style={{
            display: "flex",
            flexDirection: "column",
            background: "var(--surface2)",
            borderRadius: 6,
            padding: "10px 14px",
            marginBottom: 8,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontWeight: 500, fontSize: 13, display: "flex", alignItems: "center", gap: 6 }}>
              <span>{statusIcon(f.status)}</span>
              <span style={{ color: statusColor(f.status), maxWidth: 340, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {f.filename}
              </span>
            </span>
            {f.status === "done" && (
              <span style={{ color: "var(--text-muted)", fontSize: 11 }}>
                {f.pages}p · {f.entities}e · {f.relations}r
              </span>
            )}
          </div>
          {/* Show doc_id badge once done so user knows the identifier */}
          {f.status === "done" && f.doc_id && (
            <span style={{ marginTop: 4, fontSize: 10, color: "var(--text-muted)", fontFamily: "monospace" }}>
              id: {f.doc_id}
            </span>
          )}
          {f.status === "processing" && (
            <div style={{ height: 3, background: "var(--border)", borderRadius: 2, marginTop: 8, overflow: "hidden" }}>
              <div
                style={{
                  height: "100%",
                  width: "40%",
                  background: "var(--accent)",
                  borderRadius: 2,
                  animation: "slide 1.2s infinite linear",
                }}
              />
            </div>
          )}
          {f.status === "failed" && f.error && (
            <p style={{ color: "var(--error)", fontSize: 11, marginTop: 4 }}>{f.error}</p>
          )}
        </div>
      ))}

      {connected && (
        <p style={{ textAlign: "center", color: "var(--text-muted)", fontSize: 12, marginTop: 8 }}>
          ● Live — connected
        </p>
      )}

      <style>{`
        @keyframes slide {
          0%   { transform: translateX(-100%); }
          100% { transform: translateX(350%); }
        }
      `}</style>
    </div>
  );
}
