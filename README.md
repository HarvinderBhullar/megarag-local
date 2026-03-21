# MegaRAG — Multimodal Knowledge Graph RAG (prototype)

A local-first implementation inspired by the [MegaRAG](https://arxiv.org/abs/2512.20626) research paper. Ingests PDFs into a per-document multimodal knowledge graph and answers questions using ColQwen embeddings + DuckDB + Qdrant. No cloud APIs required.

> **Based on:**
> Chi-Hsiang Hsiao, Yi-Cheng Wang, Tzung-Sheng Lin, Yi-Ren Yeh, Chu-Song Chen.
> *"MegaRAG: Multimodal Knowledge Graph-Based Retrieval Augmented Generation"*
> arXiv:2512.20626 (2025). https://arxiv.org/abs/2512.20626

## Architecture

```
PDF → Docling (markdown) → Ollama (KG extraction) → DuckDB (entities/relations)
    → PyMuPDF (page images) → ColQwen (embeddings) → Qdrant (vector store)
```

**Backend:** FastAPI · DuckDB · Qdrant · Docling · ColQwen2
**Frontend:** React + Vite · Cytoscape.js
**LLM:** Ollama (local) — default `llama3.2`

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

### 2. Start Ollama and pull a model

```bash
ollama serve                  # starts on http://localhost:11434
ollama pull llama3.2          # default model (3.2B, fast)
# or: ollama pull llama3.1   # 8B, higher quality
```

### 3. Install backend and run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # tweak settings if needed

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
| `OLLAMA_MODEL` | `llama3.2` | Model used for entity/relation extraction |
| `OLLAMA_VISION_MODEL` | _(optional)_ | Vision model for Stage 2 visual refinement (e.g. `llava`) |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `COLQWEN_DEVICE` | `mps` | Device for ColQwen embeddings (`mps`, `cuda`, `cpu`) |

---

## API

### Batch ingest (multiple PDFs)

```bash
# Upload files — returns a job_id
curl -X POST http://localhost:8000/batch/ingest \
  -F "files=@doc1.pdf" -F "files=@doc2.pdf"

# Stream real-time progress (SSE)
curl http://localhost:8000/batch/stream/{job_id}

# Poll status
curl http://localhost:8000/batch/status/{job_id}
```

### Single document ingest

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@document.pdf"
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?"}'
```

### Knowledge graph

```bash
# Returns Cytoscape.js-compatible nodes + edges (capped at 500 nodes / 1000 edges)
curl http://localhost:8000/kg/graph
```

---

## Frontend

Open `http://localhost:5173` after running `npm run dev`.

| Tab | Description |
|---|---|
| **Upload** | Drag-and-drop multi-PDF upload with real-time per-file progress bars |
| **Query** | Ask questions; see answers with source page references |
| **Knowledge Graph** | Interactive Cytoscape.js graph — click a node to inspect its relations |

---

## Recommended Ollama models

| Model | Size | Notes |
|---|---|---|
| `llama3.2` | 3.2B | **Default** — fast, good for most documents |
| `llama3.1` | 8B | Better quality on complex documents |
| `mistral` | 7B | Strong structured JSON output |
| `zeffmuks/universal-ner` | 7B | Fine-tuned specifically for NER |
