import React, { useCallback, useRef, useState } from "react";
import { batchIngest, openBatchStream, FileStatusModel } from "../api/client";

interface Props {
  onJobStart: (jobId: string) => void;
}

export default function UploadZone({ onJobStart }: Props) {
  const [dragging, setDragging] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const addFiles = useCallback((incoming: FileList | null) => {
    if (!incoming) return;
    const pdfs = Array.from(incoming).filter(
      (f) => f.type === "application/pdf" || f.name.endsWith(".pdf")
    );
    setFiles((prev) => {
      const names = new Set(prev.map((f) => f.name));
      return [...prev, ...pdfs.filter((f) => !names.has(f.name))];
    });
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      addFiles(e.dataTransfer.files);
    },
    [addFiles]
  );

  const remove = (name: string) =>
    setFiles((prev) => prev.filter((f) => f.name !== name));

  const handleUpload = async () => {
    if (!files.length) return;
    setUploading(true);
    setError(null);
    try {
      const { job_id } = await batchIngest(files);
      setFiles([]);
      onJobStart(job_id);
    } catch (e) {
      setError(String(e));
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{ maxWidth: 640, margin: "0 auto" }}>
      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        style={{
          border: `2px dashed ${dragging ? "var(--accent)" : "var(--border)"}`,
          borderRadius: "var(--radius)",
          padding: "48px 24px",
          textAlign: "center",
          cursor: "pointer",
          background: dragging ? "rgba(108,142,245,0.08)" : "var(--surface)",
          transition: "all 0.2s",
          userSelect: "none",
        }}
      >
        <div style={{ fontSize: 40, marginBottom: 12 }}>📄</div>
        <p style={{ color: "var(--text)", fontWeight: 600, marginBottom: 4 }}>
          Drop PDF files here, or click to select
        </p>
        <p style={{ color: "var(--text-muted)", fontSize: 12 }}>
          Multiple files supported — each will be ingested into the knowledge graph
        </p>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,application/pdf"
          multiple
          style={{ display: "none" }}
          onChange={(e) => addFiles(e.target.files)}
        />
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div style={{ marginTop: 16 }}>
          {files.map((f) => (
            <div
              key={f.name}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                background: "var(--surface2)",
                borderRadius: 6,
                padding: "8px 12px",
                marginBottom: 6,
              }}
            >
              <span style={{ fontSize: 13, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: "80%" }}>
                📄 {f.name}{" "}
                <span style={{ color: "var(--text-muted)", fontSize: 11 }}>
                  ({(f.size / 1024 / 1024).toFixed(1)} MB)
                </span>
              </span>
              <button
                onClick={(e) => { e.stopPropagation(); remove(f.name); }}
                style={{
                  background: "none",
                  color: "var(--text-muted)",
                  fontSize: 16,
                  lineHeight: 1,
                  padding: "0 4px",
                }}
              >
                ×
              </button>
            </div>
          ))}

          {error && (
            <p style={{ color: "var(--error)", fontSize: 13, marginTop: 8 }}>{error}</p>
          )}

          <button
            onClick={handleUpload}
            disabled={uploading}
            style={{
              marginTop: 12,
              width: "100%",
              padding: "10px 0",
              borderRadius: 6,
              background: uploading ? "var(--surface2)" : "var(--accent)",
              color: uploading ? "var(--text-muted)" : "#fff",
              fontWeight: 600,
              fontSize: 14,
              transition: "background 0.2s",
            }}
          >
            {uploading ? "Uploading…" : `Ingest ${files.length} file${files.length > 1 ? "s" : ""}`}
          </button>
        </div>
      )}
    </div>
  );
}
