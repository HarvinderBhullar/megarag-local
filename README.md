# megarag-local — Local-first Multimodal Knowledge Graph RAG

A fully local implementation inspired by the [MegaRAG](https://arxiv.org/abs/2512.20626) research paper. Ingests PDFs into per-document multimodal knowledge graphs and answers questions using ColQwen2 visual embeddings + DuckDB + Qdrant. No cloud APIs required.

> **Based on:**
> Chi-Hsiang Hsiao, Yi-Cheng Wang, Tzung-Sheng Lin, Yi-Ren Yeh, Chu-Song Chen.
> *"MegaRAG: Multimodal Knowledge Graph-Based Retrieval Augmented Generation"*
> arXiv:2512.20626 (2025). https://arxiv.org/abs/2512.20626

---

## Architecture

```
PDF → Docling (markdown per page)
    → Ollama LLM (entity + relation extraction) → DuckDB (per-doc KG schema)
    → PyMuPDF (page images) → ColQwen2 (multi-vector embeddings) → Qdrant (per-doc collection)
```

Each uploaded PDF gets its own isolated stores:
- **Qdrant collection** — `megarag_{doc_id}` (ColQwen2 multi-vector page embeddings)
- **DuckDB schema** — `doc_{doc_id}` (entities + relations knowledge graph)

**Backend:** FastAPI · DuckDB · Qdrant · Docling · ColQwen2
**Frontend:** React + Vite · Cytoscape.js
**LLM:** Ollama (fully local) — default `llama3.2`

---

## Query flow

```
User question
     │
     ├─ 1. Keyword parsing      Ollama extracts low-level (entities, names) and
     │                          high-level (themes, concepts) keywords
     │
     ├─ 2a. Vector retrieval    ColQwen2 embeds the question → MaxSim search in Qdrant
     │                          Returns top-K page hits (doc, page index, image path)
     │
     ├─ 2b. KG retrieval        Keyword search in DuckDB entities table
     │                          Expands to 1-hop relation subgraph
     │
     └─ 3. Answer generation
             Stage 1 (always)   KG subgraph + question → Ollama → draft answer
             Stage 2 (optional) Draft + top-5 page images → vision model → refined answer
                                Requires OLLAMA_VISION_MODEL set in .env
```

Queries can be **scoped to a single document** (via the Scope dropdown in the UI or `doc_id` in the API). When scoped, both Qdrant and DuckDB searches are limited to that document's stores only.

---

## Quick start

### Prerequisites

- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com) running locally
- Docker (for Qdrant)

### 1. Start Qdrant

```bash
docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 2. Start Ollama and pull models

```bash
ollama serve                        # starts on http://localhost:11434
ollama pull llama3.2                # text model — KG extraction + Stage 1 answers

# Optional: enable Stage 2 visual refinement
ollama pull llama3.2-vision         # vision model — refines answers using page images
```

### 3. Install backend and run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env                # tweak settings if needed

uvicorn megarag.api.main:app --reload
# → http://localhost:8000
```

### 4. Install frontend and run

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## Configuration (`.env`)

All settings have sensible defaults. Only override what you need.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Text model for KG extraction + Stage 1 answer generation |
| `OLLAMA_VISION_MODEL` | _(empty — Stage 2 disabled)_ | Vision model for Stage 2 visual refinement. Set to e.g. `llama3.2-vision` to enable |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `COLQWEN_DEVICE` | `mps` | Device for ColQwen2 embeddings (`mps` for Apple Silicon, `cuda`, `cpu`) |

### Enabling Stage 2 visual refinement

Stage 2 uses the ColQwen-retrieved page images to visually ground the answer. To enable:

```env
OLLAMA_VISION_MODEL=llama3.2-vision
```

Then pull the model:

```bash
ollama pull llama3.2-vision
```

When set, Stage 2 sends the top-5 retrieved page images (base64 encoded) + the Stage 1 draft to the vision model, producing an answer grounded in visual evidence from the original PDF pages.

---

## Frontend

Open `http://localhost:5173` after running `npm run dev`.

| Tab | Description |
|---|---|
| **Upload** | Drag-and-drop multi-PDF upload with real-time per-file SSE progress (pages, entities, relations) |
| **Query** | Ask questions against all documents or a specific one. Shows answer, KG reasoning draft, and source page thumbnails (click to expand) |
| **Knowledge Graph** | Interactive Cytoscape.js graph — colour-coded by entity type, click a node to inspect its connections. Scope to a single document or view the combined graph |

---

## API

### Ingest

```bash
# Batch upload — returns a job_id
curl -X POST http://localhost:8000/batch/ingest \
  -F "files=@doc1.pdf" -F "files=@doc2.pdf"

# Stream real-time progress (SSE)
curl http://localhost:8000/batch/stream/{job_id}

# Poll status
curl http://localhost:8000/batch/status/{job_id}

# Single file
curl -X POST http://localhost:8000/ingest -F "file=@document.pdf"
```

### Query

```bash
# Search across all documents
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?", "top_k": 5}'

# Scope to a single document (use doc_id returned by /ingest)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?", "doc_id": "my_paper"}'
```

### Knowledge graph

```bash
# List documents with completed KG data
curl http://localhost:8000/kg/docs

# Full graph (all documents, capped at 500 nodes / 1000 edges)
curl http://localhost:8000/kg/graph

# Scoped to one document
curl http://localhost:8000/kg/graph?doc_id=my_paper
```

---

## Recommended Ollama models

| Model | Size | Use case |
|---|---|---|
| `llama3.2` | 3.2B | **Default text model** — fast, good for most documents |
| `llama3.1` | 8B | Better quality KG extraction on complex documents |
| `mistral` | 7B | Strong structured JSON output for KG extraction |
| `llama3.2-vision` | 11B | **Stage 2 vision model** — visual answer refinement |
| `zeffmuks/universal-ner` | 7B | Fine-tuned for named entity recognition |

---

## Per-document isolation

Every uploaded PDF is fully isolated — no cross-contamination between documents:

- Qdrant: each document gets its own collection (`megarag_{doc_id}`)
- DuckDB: each document gets its own schema (`doc_{doc_id}`)
- Queries and KG views can be scoped to a single document or fanned-out across all

`doc_id` is derived from the PDF filename (lowercased, special chars replaced with `_`, max 40 chars).

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
