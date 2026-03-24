import React, { useState, useCallback, useEffect } from "react";
import { query, QueryResponse, DocInfo, listKGDocs } from "../api/client";

interface Props {
  docs: DocInfo[];
}

export default function QueryPanel({ docs }: Props) {
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [selectedDocId, setSelectedDocId] = useState<string>("");  // "" = all docs
  const [loading, setLoading] = useState(false);
  const [backendDocs, setBackendDocs] = useState<DocInfo[]>([]);

  // Load available docs from backend on mount — survives page refresh
  useEffect(() => {
    listKGDocs()
      .then((ids) => setBackendDocs(ids.map((id) => ({ doc_id: id, document: id }))))
      .catch(() => {});
  }, []);

  // Merge backend docs with SSE-driven docs (deduped by doc_id)
  const allDocs = Array.from(
    new Map([...backendDocs, ...docs].map((d) => [d.doc_id, d])).values()
  );
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lightbox, setLightbox] = useState<string | null>(null);

  const closeLightbox = useCallback((e: React.MouseEvent) => {
    if ((e.target as HTMLElement).dataset.overlay) setLightbox(null);
  }, []);

  const handleQuery = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await query(question, topK, selectedDocId || undefined);
      setResult(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 720, margin: "0 auto" }}>

      {/* Lightbox overlay */}
      {lightbox && (
        <div
          data-overlay="1"
          onClick={closeLightbox}
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.85)",
            zIndex: 1000,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div style={{ position: "relative", maxWidth: "90vw", maxHeight: "90vh" }}>
            <button
              onClick={() => setLightbox(null)}
              style={{
                position: "absolute",
                top: -36,
                right: 0,
                background: "transparent",
                border: "none",
                color: "#fff",
                fontSize: 28,
                cursor: "pointer",
                lineHeight: 1,
              }}
            >
              ×
            </button>
            <img
              src={lightbox}
              alt="Source page"
              style={{
                maxWidth: "90vw",
                maxHeight: "90vh",
                objectFit: "contain",
                borderRadius: 8,
                boxShadow: "0 8px 40px rgba(0,0,0,0.6)",
              }}
            />
          </div>
        </div>
      )}

      {/* Input */}
      <div
        style={{
          background: "var(--surface)",
          borderRadius: "var(--radius)",
          padding: 20,
          marginBottom: 20,
        }}
      >
        {/* Doc scope selector */}
        <div style={{ marginBottom: 14 }}>
          <label style={{ display: "block", fontWeight: 600, marginBottom: 6, fontSize: 13 }}>
            Scope
          </label>
          <select
            value={selectedDocId}
            onChange={(e) => setSelectedDocId(e.target.value)}
            style={{
              width: "100%",
              background: "var(--surface2)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              padding: "8px 10px",
              color: "var(--text)",
              fontSize: 13,
            }}
          >
            <option value="">All documents</option>
            {allDocs.map((d) => (
              <option key={d.doc_id} value={d.doc_id}>
                {d.document}
              </option>
            ))}
          </select>
        </div>

        <label style={{ display: "block", fontWeight: 600, marginBottom: 8 }}>
          Ask a question
        </label>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleQuery();
          }}
          placeholder="e.g. What are the main entities and their relationships?"
          rows={3}
          style={{
            width: "100%",
            background: "var(--surface2)",
            border: "1px solid var(--border)",
            borderRadius: 6,
            padding: "10px 12px",
            color: "var(--text)",
            fontSize: 14,
            resize: "vertical",
          }}
        />
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginTop: 12,
          }}
        >
          <label style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--text-muted)", fontSize: 13 }}>
            Top-K results:
            <input
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              style={{
                width: 60,
                background: "var(--surface2)",
                border: "1px solid var(--border)",
                borderRadius: 4,
                padding: "4px 8px",
                color: "var(--text)",
                fontSize: 13,
              }}
            />
          </label>
          <button
            onClick={handleQuery}
            disabled={loading || !question.trim()}
            style={{
              padding: "8px 24px",
              borderRadius: 6,
              background: loading || !question.trim() ? "var(--surface2)" : "var(--accent)",
              color: loading || !question.trim() ? "var(--text-muted)" : "#fff",
              fontWeight: 600,
              fontSize: 14,
              transition: "background 0.2s",
            }}
          >
            {loading ? "Thinking…" : "Ask ⌘↵"}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div
          style={{
            background: "rgba(245,101,101,0.1)",
            border: "1px solid var(--error)",
            borderRadius: "var(--radius)",
            padding: "12px 16px",
            color: "var(--error)",
            marginBottom: 20,
          }}
        >
          {error}
        </div>
      )}

      {/* Result */}
      {result && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

          {/* Answer */}
          <div
            style={{
              background: "var(--surface)",
              borderRadius: "var(--radius)",
              padding: 20,
            }}
          >
            <h3 style={{ marginBottom: 10, fontSize: 14, color: "var(--accent)", fontWeight: 700 }}>
              Answer
            </h3>
            <p style={{ lineHeight: 1.8, whiteSpace: "pre-wrap" }}>{result.answer}</p>
          </div>

          {/* Draft reasoning */}
          <details style={{ background: "var(--surface)", borderRadius: "var(--radius)", padding: 20 }}>
            <summary style={{ cursor: "pointer", color: "var(--text-muted)", fontSize: 13, fontWeight: 600 }}>
              KG Draft (knowledge graph reasoning)
            </summary>
            <p style={{ marginTop: 12, lineHeight: 1.8, whiteSpace: "pre-wrap", color: "var(--text-muted)", fontSize: 13 }}>
              {result.draft}
            </p>
          </details>

          {/* Source pages */}
          {result.sources.length > 0 && (
            <div
              style={{
                background: "var(--surface)",
                borderRadius: "var(--radius)",
                padding: 20,
              }}
            >
              <h3 style={{ marginBottom: 14, fontSize: 14, color: "var(--accent)", fontWeight: 700 }}>
                Source pages ({result.sources.length})
              </h3>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
                  gap: 12,
                }}
              >
                {result.sources.map((src, i) => (
                  <div
                    key={i}
                    onClick={() => setLightbox(src)}
                    title="Click to expand"
                    style={{
                      cursor: "pointer",
                      border: "1px solid var(--border)",
                      borderRadius: 6,
                      overflow: "hidden",
                      background: "var(--surface2)",
                      transition: "transform 0.15s, box-shadow 0.15s",
                    }}
                    onMouseEnter={(e) => {
                      (e.currentTarget as HTMLElement).style.transform = "scale(1.03)";
                      (e.currentTarget as HTMLElement).style.boxShadow = "0 4px 16px rgba(0,0,0,0.3)";
                    }}
                    onMouseLeave={(e) => {
                      (e.currentTarget as HTMLElement).style.transform = "scale(1)";
                      (e.currentTarget as HTMLElement).style.boxShadow = "none";
                    }}
                  >
                    <img
                      src={src}
                      alt={`Page ${i + 1}`}
                      loading="lazy"
                      style={{
                        width: "100%",
                        display: "block",
                        objectFit: "cover",
                        aspectRatio: "3/4",
                      }}
                      onError={(e) => {
                        const el = e.currentTarget.parentElement!;
                        el.innerHTML = `<div style="padding:12px 8px;font-size:11px;color:var(--text-muted);text-align:center;word-break:break-all;">📄 ${src.split("/").pop()}</div>`;
                      }}
                    />
                    <div
                      style={{
                        padding: "4px 8px",
                        fontSize: 11,
                        color: "var(--text-muted)",
                        textAlign: "center",
                        borderTop: "1px solid var(--border)",
                      }}
                    >
                      Page {i + 1}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>
      )}
    </div>
  );
}
