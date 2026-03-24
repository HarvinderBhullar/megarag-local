import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import ray

from config.settings import get_settings
from megarag.embedding.colqwen import ColQwenActor
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

    # Initialise Ray cluster (single node, local)
    ray.init(ignore_reinit_error=True, logging_level=logging.WARNING)
    logger.info("[startup] Ray initialised — %s", ray.cluster_resources())

    # Create a named, long-lived actor that holds the single ColQwen model.
    # All Ray workers call this actor instead of loading their own GPU copy.
    logger.info("[startup] loading ColQwen model via Ray actor...")
    actor = ColQwenActor.options(
        name="colqwen",
        lifetime="detached",
        get_if_exists=True,
    ).remote()
    # Warm up: trigger model load now (blocks until ready)
    ray.get(actor.embed_query.remote("warmup"))
    logger.info("[startup] ColQwen actor ready.")

    yield

    ray.shutdown()
    logger.info("[shutdown] Ray shut down.")


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
