import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config.settings import get_settings
from megarag.embedding.colqwen import _load_model  # warm up on startup
from megarag.api.routes.ingest import router as ingest_router
from megarag.api.routes.query import router as query_router
from megarag.api.routes.batch import router as batch_router
from megarag.api.routes.kg import router as kg_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    cfg.ensure_dirs()
    logger.info("[startup] loading ColQwen model...")
    _load_model()           # loads once, cached via lru_cache
    logger.info("[startup] ready.")
    yield


app = FastAPI(title="MegaRAG prototype", version="0.1.0", lifespan=lifespan)

# Allow the Vite dev server (port 5173) and any localhost origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(batch_router)
app.include_router(kg_router)

# Serve extracted page images so the frontend can display them.
# Accessible at /pages/<filename>.png
cfg_static = get_settings()
cfg_static.ensure_dirs()
app.mount("/pages", StaticFiles(directory=str(cfg_static.pages_dir)), name="pages")


@app.get("/health")
def health():
    return {"status": "ok"}
