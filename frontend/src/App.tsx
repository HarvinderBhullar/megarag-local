import React, { useState } from "react";
import UploadZone from "./components/UploadZone";
import BatchProgress from "./components/BatchProgress";
import QueryPanel from "./components/QueryPanel";
import KnowledgeGraph from "./components/KnowledgeGraph";
import { DocInfo } from "./api/client";

type Tab = "upload" | "query" | "graph";

export default function App() {
  const [tab, setTab] = useState<Tab>("upload");
  const [activeJobs, setActiveJobs] = useState<string[]>([]);
  // All successfully ingested documents, accumulated across all batch jobs.
  const [docs, setDocs] = useState<DocInfo[]>([]);

  const handleJobStart = (jobId: string) => {
    setActiveJobs((prev) => [jobId, ...prev]);
  };

  const handleDocIngested = (doc: DocInfo) => {
    setDocs((prev) => {
      // Avoid duplicates (re-upload of same file)
      if (prev.some((d) => d.doc_id === doc.doc_id)) return prev;
      return [...prev, doc];
    });
  };

  const tabStyle = (t: Tab): React.CSSProperties => ({
    padding: "8px 20px",
    borderRadius: 6,
    fontWeight: 600,
    fontSize: 14,
    cursor: "pointer",
    background: tab === t ? "var(--accent)" : "transparent",
    color: tab === t ? "#fff" : "var(--text-muted)",
    border: "none",
    transition: "all 0.15s",
  });

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)" }}>
      {/* Header */}
      <header
        style={{
          background: "var(--surface)",
          borderBottom: "1px solid var(--border)",
          padding: "0 24px",
          display: "flex",
          alignItems: "center",
          gap: 32,
          height: 56,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 22 }}>🧠</span>
          <span style={{ fontWeight: 800, fontSize: 18, letterSpacing: -0.5 }}>
            MegaRAG
          </span>
        </div>

        {/* Tabs */}
        <nav style={{ display: "flex", gap: 4 }}>
          <button style={tabStyle("upload")} onClick={() => setTab("upload")}>
            📤 Upload
          </button>
          <button style={tabStyle("query")} onClick={() => setTab("query")}>
            💬 Query
          </button>
          <button style={tabStyle("graph")} onClick={() => setTab("graph")}>
            🕸 Knowledge Graph
          </button>
        </nav>

        {/* Doc count badge */}
        {docs.length > 0 && (
          <span style={{ marginLeft: "auto", color: "var(--text-muted)", fontSize: 12 }}>
            {docs.length} doc{docs.length !== 1 ? "s" : ""} ingested
          </span>
        )}
      </header>

      {/* Main */}
      <main style={{ padding: 24 }}>
        {tab === "upload" && (
          <div>
            <h2 style={{ marginBottom: 20, fontSize: 18, fontWeight: 700 }}>
              Ingest Documents
            </h2>
            <UploadZone onJobStart={handleJobStart} />

            {activeJobs.length > 0 && (
              <div style={{ marginTop: 32 }}>
                <h3
                  style={{
                    marginBottom: 16,
                    fontSize: 15,
                    fontWeight: 700,
                    color: "var(--text-muted)",
                  }}
                >
                  Ingestion Jobs
                </h3>
                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                  {activeJobs.map((jobId) => (
                    <BatchProgress
                      key={jobId}
                      jobId={jobId}
                      onDocIngested={handleDocIngested}
                      onComplete={() => {}}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {tab === "query" && (
          <div>
            <h2 style={{ marginBottom: 20, fontSize: 18, fontWeight: 700 }}>
              Query Knowledge Graph
            </h2>
            <QueryPanel docs={docs} />
          </div>
        )}

        {tab === "graph" && (
          <div>
            <h2 style={{ marginBottom: 20, fontSize: 18, fontWeight: 700 }}>
              Knowledge Graph
            </h2>
            <KnowledgeGraph docs={docs} />
          </div>
        )}
      </main>
    </div>
  );
}
