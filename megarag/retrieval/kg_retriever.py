"""Retrieve relevant KG subgraph using keyword-matched entities."""
import ray
from config.settings import get_settings
from megarag.knowledge_graph.store import KGStore


@ray.remote
def _search_one_schema(keywords: list[str], schema: str, db_path_str: str) -> dict:
    """Ray remote: entity + subgraph search in a single DuckDB schema.

    Opens read_only=True so multiple Ray workers can query DuckDB concurrently
    without conflicting with each other or with an ongoing ingest write lock.
    Data is visible because pipeline.py explicitly closes the write connection
    (triggering a WAL checkpoint) before ingest_document returns.
    """
    import duckdb
    from pathlib import Path
    db_path = Path(db_path_str)
    if not db_path.exists():
        return {"entities": [], "relations": []}
    try:
        conn = duckdb.connect(str(db_path), read_only=True)
    except Exception:
        return {"entities": [], "relations": []}
    try:
        e_tbl = f"{schema}.entities"
        r_tbl = f"{schema}.relations"
        if not keywords:
            return {"entities": [], "relations": []}
        clauses = " OR ".join("(name ILIKE ? OR description ILIKE ?)" for _ in keywords)
        params = [val for k in keywords for val in (f"%{k}%", f"%{k}%")]
        entity_rows = conn.execute(
            f"SELECT id, name, type, description, source FROM {e_tbl} WHERE {clauses} LIMIT 50",
            params,
        ).fetchall()
        cols_e = ["id", "name", "type", "description", "source"]
        entities = [dict(zip(cols_e, r)) for r in entity_rows]

        if not entities:
            return {"entities": [], "relations": []}

        entity_names = [e["name"] for e in entities]
        placeholders = ",".join("?" * len(entity_names))
        rel_rows = conn.execute(
            f"""SELECT id, source_ent, relation, target_ent, source_doc FROM {r_tbl}
                WHERE source_ent IN ({placeholders}) OR target_ent IN ({placeholders})
                LIMIT 100""",
            entity_names * 2,
        ).fetchall()
        cols_r = ["id", "source_ent", "relation", "target_ent", "source_doc"]
        relations = [dict(zip(cols_r, r)) for r in rel_rows]
        return {"entities": entities, "relations": relations}
    except Exception:
        return {"entities": [], "relations": []}
    finally:
        conn.close()


def retrieve_subgraph(
    keywords: list[str],
    schema: str | None = None,
) -> dict:
    """
    Search for entities matching *keywords*, then pull their relations.

    If *schema* is given, search only that document's DuckDB schema.
    Otherwise fan-out across all per-document schemas (prefix 'doc_') and merge.

    Schemas are searched in parallel via Ray.
    Returns dict with 'entities' and 'relations' lists.
    Returns empty result if no documents have been ingested yet.
    """
    cfg = get_settings()

    if not cfg.kg_db_path.exists():
        return {"entities": [], "relations": []}

    if schema:
        schemas = [schema]
    else:
        schemas = KGStore.list_doc_schemas(cfg.kg_db_path)

    if not schemas:
        return {"entities": [], "relations": []}

    # Search all schemas in parallel
    futures = [
        _search_one_schema.remote(keywords, s, str(cfg.kg_db_path))
        for s in schemas
    ]
    results = ray.get(futures)

    all_entities: list[dict] = []
    all_relations: list[dict] = []
    for r in results:
        all_entities.extend(r["entities"])
        all_relations.extend(r["relations"])

    return {"entities": all_entities, "relations": all_relations}
