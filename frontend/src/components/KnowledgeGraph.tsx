import React, { useCallback, useEffect, useRef, useState } from "react";
import cytoscape, { Core, NodeSingular } from "cytoscape";
import { getKGGraph, KGNode, KGEdge, DocInfo } from "../api/client";

const TYPE_COLORS: Record<string, string> = {
  PERSON: "#6c8ef5",
  ORG: "#4caf7d",
  CONCEPT: "#f6ad55",
  LOCATION: "#f56565",
  EVENT: "#a78bfa",
  PRODUCT: "#38bdf8",
  OTHER: "#8b8fa8",
};

interface NodeInfo {
  id: string;
  label: string;
  type: string;
  edges: Array<{ label: string; target: string; direction: "out" | "in" }>;
}

interface Props {
  docs: DocInfo[];
}

export default function KnowledgeGraph({ docs }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [nodeCount, setNodeCount] = useState(0);
  const [edgeCount, setEdgeCount] = useState(0);
  const [selected, setSelected] = useState<NodeInfo | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedDocId, setSelectedDocId] = useState<string>("");  // "" = all docs

  const loadGraph = useCallback(async (docId?: string) => {
    if (!containerRef.current) return;
    setLoading(true);
    setError(null);
    try {
      const data = await getKGGraph(docId || undefined);
      setNodeCount(data.nodes.length);
      setEdgeCount(data.edges.length);

      if (cyRef.current) {
        cyRef.current.destroy();
        cyRef.current = null;
      }

      const elements = [
        ...data.nodes.map((n: KGNode) => ({
          group: "nodes" as const,
          data: n.data,
        })),
        ...data.edges.map((e: KGEdge) => ({
          group: "edges" as const,
          data: e.data,
        })),
      ];

      const cy = cytoscape({
        container: containerRef.current,
        elements,
        style: [
          {
            selector: "node",
            style: {
              "background-color": (ele: NodeSingular) =>
                TYPE_COLORS[ele.data("type")] ?? "#8b8fa8",
              label: "data(label)",
              color: "#e2e4f0",
              "font-size": 11,
              "text-valign": "center",
              "text-halign": "center",
              "text-wrap": "wrap",
              "text-max-width": "80px",
              width: 40,
              height: 40,
              "border-width": 2,
              "border-color": "#2e3148",
              "overlay-padding": "6px",
            },
          },
          {
            selector: "node:selected",
            style: {
              "border-color": "#fff",
              "border-width": 3,
            },
          },
          {
            selector: "edge",
            style: {
              width: 1.5,
              "line-color": "#2e3148",
              "target-arrow-color": "#2e3148",
              "target-arrow-shape": "triangle",
              "curve-style": "bezier",
              label: "data(label)",
              "font-size": 9,
              color: "#8b8fa8",
              "text-rotation": "autorotate",
              "text-margin-y": -8,
            },
          },
          {
            selector: "edge:selected",
            style: {
              "line-color": "var(--accent)",
              "target-arrow-color": "var(--accent)",
              color: "#e2e4f0",
            },
          },
          {
            selector: ".highlighted",
            style: {
              "background-color": "#fff",
              "border-color": "var(--accent)",
              "border-width": 3,
            },
          },
        ],
        layout: {
          name: "cose",
          animate: false,
          randomize: true,
          nodeRepulsion: () => 4500,
          idealEdgeLength: () => 100,
          edgeElasticity: () => 100,
          gravity: 80,
          numIter: 1000,
          initialTemp: 200,
          coolingFactor: 0.95,
          minTemp: 1.0,
        } as Parameters<Core["layout"]>[0],
      });

      cy.on("tap", "node", (evt) => {
        const node = evt.target;
        const connectedEdges = node.connectedEdges();
        const info: NodeInfo = {
          id: node.id(),
          label: node.data("label"),
          type: node.data("type"),
          edges: connectedEdges.map((e: ReturnType<Core["edges"]>[0]) => ({
            label: e.data("label"),
            target:
              e.data("source") === node.id()
                ? e.target().data("label")
                : e.source().data("label"),
            direction: e.data("source") === node.id() ? "out" : "in",
          })),
        };
        setSelected(info);
      });

      cy.on("tap", (evt) => {
        if (evt.target === cy) setSelected(null);
      });

      cyRef.current = cy;
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  // Reload graph whenever the selected doc changes
  useEffect(() => {
    loadGraph(selectedDocId || undefined);
    return () => {
      cyRef.current?.destroy();
    };
  }, [selectedDocId]);

  // Initial load
  useEffect(() => {
    loadGraph();
    return () => {
      cyRef.current?.destroy();
    };
  }, []);

  const handleSearch = (term: string) => {
    setSearchTerm(term);
    if (!cyRef.current) return;
    cyRef.current.elements().removeClass("highlighted");
    if (!term.trim()) return;
    const matches = cyRef.current
      .nodes()
      .filter((n) =>
        n.data("label").toLowerCase().includes(term.toLowerCase())
      );
    matches.addClass("highlighted");
    if (matches.length > 0) {
      cyRef.current.animate({ fit: { eles: matches, padding: 80 } } as Parameters<Core["animate"]>[0], { duration: 500 });
    }
  };

  const fitGraph = () => cyRef.current?.fit(undefined, 40);

  return (
    <div style={{ display: "flex", gap: 16, height: "calc(100vh - 180px)", minHeight: 500 }}>
      {/* Graph canvas */}
      <div style={{ flex: 1, position: "relative" }}>
        {/* Toolbar */}
        <div
          style={{
            position: "absolute",
            top: 12,
            left: 12,
            zIndex: 10,
            display: "flex",
            gap: 8,
            alignItems: "center",
            flexWrap: "wrap",
          }}
        >
          {/* Doc scope selector */}
          <select
            value={selectedDocId}
            onChange={(e) => setSelectedDocId(e.target.value)}
            style={{
              background: "var(--surface2)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              padding: "6px 10px",
              color: "var(--text)",
              fontSize: 12,
              maxWidth: 180,
            }}
          >
            <option value="">All documents</option>
            {docs.map((d) => (
              <option key={d.doc_id} value={d.doc_id}>
                {d.document}
              </option>
            ))}
          </select>

          <input
            value={searchTerm}
            onChange={(e) => handleSearch(e.target.value)}
            placeholder="Search entities…"
            style={{
              background: "var(--surface2)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              padding: "6px 12px",
              color: "var(--text)",
              fontSize: 13,
              width: 180,
            }}
          />
          <button
            onClick={fitGraph}
            title="Fit to view"
            style={{
              background: "var(--surface2)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              padding: "6px 10px",
              color: "var(--text-muted)",
              fontSize: 13,
            }}
          >
            ⊡
          </button>
          <button
            onClick={() => loadGraph(selectedDocId || undefined)}
            title="Reload graph"
            style={{
              background: "var(--surface2)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              padding: "6px 10px",
              color: "var(--text-muted)",
              fontSize: 13,
            }}
          >
            ↺
          </button>
          <span style={{ color: "var(--text-muted)", fontSize: 12 }}>
            {nodeCount} nodes · {edgeCount} edges
          </span>
        </div>

        {loading && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              background: "rgba(15,17,23,0.7)",
              zIndex: 20,
              borderRadius: "var(--radius)",
            }}
          >
            <span style={{ color: "var(--text-muted)" }}>Loading knowledge graph…</span>
          </div>
        )}

        {error && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              zIndex: 20,
            }}
          >
            <span style={{ color: "var(--error)" }}>{error}</span>
          </div>
        )}

        <div
          ref={containerRef}
          style={{
            width: "100%",
            height: "100%",
            background: "var(--surface)",
            borderRadius: "var(--radius)",
          }}
        />
      </div>

      {/* Node detail panel */}
      <div
        style={{
          width: 260,
          background: "var(--surface)",
          borderRadius: "var(--radius)",
          padding: 16,
          overflowY: "auto",
          flexShrink: 0,
        }}
      >
        {/* Legend */}
        <h4 style={{ marginBottom: 12, fontSize: 13, fontWeight: 700, color: "var(--text-muted)" }}>
          Entity types
        </h4>
        {Object.entries(TYPE_COLORS).map(([type, color]) => (
          <div key={type} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            <span
              style={{
                width: 12,
                height: 12,
                borderRadius: "50%",
                background: color,
                flexShrink: 0,
              }}
            />
            <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{type}</span>
          </div>
        ))}

        {/* Node detail */}
        {selected && (
          <>
            <hr style={{ border: "none", borderTop: "1px solid var(--border)", margin: "16px 0" }} />
            <h4 style={{ marginBottom: 8, fontSize: 13, fontWeight: 700 }}>{selected.label}</h4>
            <span
              style={{
                display: "inline-block",
                fontSize: 11,
                padding: "2px 8px",
                borderRadius: 99,
                background: `${TYPE_COLORS[selected.type] ?? "#8b8fa8"}22`,
                color: TYPE_COLORS[selected.type] ?? "#8b8fa8",
                marginBottom: 12,
              }}
            >
              {selected.type}
            </span>
            <p style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 8 }}>
              {selected.edges.length} connection{selected.edges.length !== 1 ? "s" : ""}
            </p>
            {selected.edges.slice(0, 10).map((e, i) => (
              <div
                key={i}
                style={{
                  fontSize: 12,
                  background: "var(--surface2)",
                  borderRadius: 4,
                  padding: "5px 8px",
                  marginBottom: 4,
                  color: "var(--text-muted)",
                }}
              >
                {e.direction === "out" ? "→" : "←"}{" "}
                <span style={{ color: "var(--accent)", fontStyle: "italic" }}>{e.label}</span>{" "}
                {e.target}
              </div>
            ))}
            {selected.edges.length > 10 && (
              <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>
                +{selected.edges.length - 10} more…
              </p>
            )}
          </>
        )}

        {!selected && (
          <p style={{ marginTop: 16, fontSize: 12, color: "var(--text-muted)" }}>
            Click a node to inspect its connections.
          </p>
        )}
      </div>
    </div>
  );
}
