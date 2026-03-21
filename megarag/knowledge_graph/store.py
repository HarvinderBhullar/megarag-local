import duckdb
from pathlib import Path
from megarag.knowledge_graph.schema import init_db


class KGStore:
    def __init__(self, db_path: Path, schema: str = "main", read_only: bool = False):
        self.schema = schema
        # Shorthand so every SQL reference is schema-qualified.
        # DuckDB accepts 'main.entities' just like bare 'entities'.
        self._e = f"{schema}.entities"
        self._r = f"{schema}.relations"

        if read_only:
            if not db_path.exists():
                raise FileNotFoundError(
                    f"Knowledge graph database not found at '{db_path}'. "
                    "Upload and ingest at least one document first."
                )
            self.conn: duckdb.DuckDBPyConnection = duckdb.connect(
                str(db_path), read_only=True
            )
        else:
            self.conn = init_db(db_path, schema)

    # ------------------------------------------------------------------
    # Class-level helpers — discover which per-doc schemas exist
    # ------------------------------------------------------------------

    @classmethod
    def list_doc_schemas(cls, db_path: Path) -> list[str]:
        """Return all per-document schema names (prefix 'doc_') in the database."""
        if not db_path.exists():
            return []
        conn = duckdb.connect(str(db_path), read_only=True)
        rows = conn.execute(
            "SELECT schema_name FROM information_schema.schemata "
            "WHERE schema_name LIKE 'doc_%'"
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert_entities(self, entities: list[dict]) -> None:
        if not entities:
            return
        self.conn.executemany(
            f"""INSERT OR IGNORE INTO {self._e} (name, type, description, source)
               VALUES (?, ?, ?, ?)""",
            [
                (e["name"], e.get("type"), e.get("description"), e["source"])
                for e in entities
            ],
        )

    def upsert_relations(self, relations: list[dict]) -> None:
        if not relations:
            return

        # Ensure every source/target endpoint exists as an entity node.
        endpoint_names: set[str] = set()
        for r in relations:
            src = (r.get("source") or "").strip()
            tgt = (r.get("target") or "").strip()
            if src:
                endpoint_names.add(src)
            if tgt:
                endpoint_names.add(tgt)

        source_doc = relations[0]["source_doc"]
        self.conn.executemany(
            f"""INSERT OR IGNORE INTO {self._e} (name, type, source)
               VALUES (?, 'OTHER', ?)""",
            [(name, source_doc) for name in endpoint_names if name],
        )

        self.conn.executemany(
            f"""INSERT OR IGNORE INTO {self._r}
               (source_ent, relation, target_ent, description, keywords, source_doc)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                (
                    r["source"], r["relation"], r["target"],
                    r.get("description"),
                    r.get("keywords"),
                    r["source_doc"],
                )
                for r in relations
                if (r.get("source") or "").strip()
                and (r.get("relation") or "").strip()
                and (r.get("target") or "").strip()
            ],
        )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def search_entities(self, keywords: list[str]) -> list[dict]:
        if not keywords:
            return []
        clauses = " OR ".join(
            "(name ILIKE ? OR description ILIKE ?)" for _ in keywords
        )
        params = [val for k in keywords for val in (f"%{k}%", f"%{k}%")]
        rows = self.conn.execute(
            f"SELECT id, name, type, description, source FROM {self._e} "
            f"WHERE {clauses} LIMIT 50",
            params,
        ).fetchall()
        cols = ["id", "name", "type", "description", "source"]
        return [dict(zip(cols, r)) for r in rows]

    def get_subgraph(self, entity_names: list[str]) -> list[dict]:
        if not entity_names:
            return []
        placeholders = ",".join("?" * len(entity_names))
        rows = self.conn.execute(
            f"""SELECT * FROM {self._r}
                WHERE source_ent IN ({placeholders})
                   OR target_ent IN ({placeholders})
                LIMIT 100""",
            entity_names * 2,
        ).fetchall()
        cols = ["id", "source_ent", "relation", "target_ent", "source_doc"]
        return [dict(zip(cols, r)) for r in rows]

    def get_all_entities(self, limit: int = 500) -> list[tuple]:
        return self.conn.execute(
            f"SELECT id, name, type, description FROM {self._e} LIMIT {limit}"
        ).fetchall()

    def get_all_relations(self, limit: int = 1000) -> list[tuple]:
        return self.conn.execute(
            f"SELECT id, source_ent, relation, target_ent, description, keywords "
            f"FROM {self._r} LIMIT {limit}"
        ).fetchall()
