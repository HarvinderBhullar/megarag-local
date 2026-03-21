"""
Knowledge graph route.

GET /kg/graph               — returns combined graph across all documents
GET /kg/graph?doc_id=<id>   — scoped to a single document (doc_id from /ingest)
"""
import logging
from typing import Optional

from fastapi import APIRouter, Query

from config.settings import get_settings
from megarag.api.schemas import KGGraphResponse
from megarag.knowledge_graph.store import KGStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kg", tags=["knowledge-graph"])


@router.get("/graph", response_model=KGGraphResponse)
def get_kg_graph(
    doc_id: Optional[str] = Query(
        default=None,
        description="Scope graph to a single document. Use the 'doc_id' returned by /ingest.",
    )
) -> KGGraphResponse:
    """
    Return the knowledge graph as Cytoscape.js-compatible nodes + edges.

    If *doc_id* is given, only that document's entities and relations are returned.
    Otherwise the combined graph across all uploaded documents is returned.
    Large graphs are capped at 500 nodes / 1000 edges to keep the response fast.
    """
    cfg = get_settings()

    if doc_id:
        schemas = [f"doc_{doc_id}"]
    else:
        schemas = KGStore.list_doc_schemas(cfg.kg_db_path)

    if not schemas:
        return KGGraphResponse(nodes=[], edges=[])

    all_entity_rows: list[tuple] = []
    all_rel_rows: list[tuple] = []

    for schema in schemas:
        try:
            store = KGStore(cfg.kg_db_path, schema=schema, read_only=True)
        except FileNotFoundError:
            continue
        all_entity_rows.extend(store.get_all_entities(limit=500))
        all_rel_rows.extend(store.get_all_relations(limit=1000))

    # Cap totals after merging across schemas.
    entity_rows = all_entity_rows[:500]
    rel_rows = all_rel_rows[:1000]

    # Build a case-insensitive name → id lookup so edges can reference node ids.
    name_to_id: dict[str, str] = {
        row[1].strip().lower(): str(row[0]) for row in entity_rows
    }

    nodes = [
        {
            "data": {
                "id": str(row[0]),
                "label": row[1],
                "type": row[2],
                "description": row[3] or "",
            }
        }
        for row in entity_rows
    ]

    edges = []
    missing_endpoints: set[str] = set()
    for row in rel_rows:
        rel_id, source_name, relation, target_name, description, keywords = row
        source_id = name_to_id.get(source_name.strip().lower())
        target_id = name_to_id.get(target_name.strip().lower())
        if source_id and target_id:
            edges.append({
                "data": {
                    "id": f"r{rel_id}",
                    "source": source_id,
                    "target": target_id,
                    "label": relation,
                    "description": description or "",
                    "keywords": keywords or "",
                }
            })
        else:
            if not source_id:
                missing_endpoints.add(source_name)
            if not target_id:
                missing_endpoints.add(target_name)

    if missing_endpoints:
        logger.warning(
            "[kg] %d relation endpoint(s) had no matching entity node "
            "(edges dropped): %s",
            len(missing_endpoints),
            ", ".join(sorted(missing_endpoints)[:20]),
        )

    logger.info(
        "[kg] graph built — %d nodes, %d edges (%d dropped) across schema(s): %s",
        len(nodes), len(edges), len(missing_endpoints), schemas,
    )

    return KGGraphResponse(nodes=nodes, edges=edges)
