import duckdb
from pathlib import Path


def _schema_ddl(schema: str) -> list[str]:
    """Return DDL statements that create all KG objects inside *schema*."""
    # Skip CREATE SCHEMA for the built-in 'main' schema.
    stmts = []
    if schema != "main":
        stmts.append(f"CREATE SCHEMA IF NOT EXISTS {schema}")

    stmts += [
        f"CREATE SEQUENCE IF NOT EXISTS {schema}.entities_id_seq START 1",
        f"CREATE SEQUENCE IF NOT EXISTS {schema}.relations_id_seq START 1",
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.entities (
            id          INTEGER PRIMARY KEY DEFAULT nextval('{schema}.entities_id_seq'),
            name        VARCHAR NOT NULL,
            type        VARCHAR,
            description VARCHAR,
            source      VARCHAR,
            UNIQUE(name, source)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.relations (
            id          INTEGER PRIMARY KEY DEFAULT nextval('{schema}.relations_id_seq'),
            source_ent  VARCHAR NOT NULL,
            relation    VARCHAR NOT NULL,
            target_ent  VARCHAR NOT NULL,
            description VARCHAR,
            keywords    VARCHAR,
            source_doc  VARCHAR,
            UNIQUE(source_ent, relation, target_ent, source_doc)
        )
        """,
        "INSTALL fts; LOAD fts;",
    ]
    return stmts


def init_db(db_path: Path, schema: str = "main") -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    for stmt in _schema_ddl(schema):
        conn.execute(stmt)
    conn.commit()
    return conn
