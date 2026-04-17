"""
shared/snowflake_conn.py — NeighbourWise AI
Centralized Snowflake connection, query execution, and Cortex LLM calls.
All agents import from here — no more per-agent connection logic.

Usage:
    from shared.snowflake_conn import get_conn, run_query, cortex_complete
"""

import os
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

# ── Config ─────────────────────────────────────────────────────────────────────
_SF_CONFIG = {
    "account":         os.environ.get("SNOWFLAKE_ACCOUNT", ""),
    "user":            os.environ.get("SNOWFLAKE_USER", ""),
    "password":        os.environ.get("SNOWFLAKE_PASSWORD", ""),
    "warehouse":       os.environ.get("SNOWFLAKE_WAREHOUSE", "NEIGHBOURWISE_AI"),
    "database":        os.environ.get("SNOWFLAKE_DATABASE", "NEIGHBOURWISE_DOMAINS"),
    "role":            os.environ.get("SNOWFLAKE_ROLE", "TRAINING_ROLE"),
    "insecure_mode":   True,
    "network_timeout": 120,
    "login_timeout":   60,
}

# ── Model constants ────────────────────────────────────────────────────────────
# Generation models (output producers)
MODEL_GENERATE = "mistral-large2"

# Validation models (cross-model validators — never same as generator)
MODEL_VALIDATE = "claude-sonnet-4-6"

# Embedding model
MODEL_EMBED    = "e5-base-v2"


# ── Connection ─────────────────────────────────────────────────────────────────
def get_conn(schema: str = "MARTS") -> snowflake.connector.SnowflakeConnection:
    """
    Open a fresh Snowflake connection.
    schema: default "MARTS" — pass "CRIME_ANALYSIS", "ANALYTICS", etc. as needed.
    Callers are responsible for closing: conn.close()
    """
    return snowflake.connector.connect(**_SF_CONFIG, schema=schema)


# ── Query execution ────────────────────────────────────────────────────────────
def run_query(
    sql: str,
    conn: snowflake.connector.SnowflakeConnection,
    params: Optional[tuple] = None,
) -> pd.DataFrame:
    """
    Execute a parameterized SQL query and return a DataFrame.
    Always uses %s placeholders — never f-string interpolation.

    Example:
        df = run_query(
            "SELECT * FROM MARTS.MASTER_LOCATION WHERE UPPER(NEIGHBORHOOD_NAME) = %s",
            conn, ("FENWAY",)
        )
    """
    cur = None
    try:
        cur = conn.cursor()
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        if cur.description:
            cols = [d[0] for d in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols)
        return pd.DataFrame()
    except Exception as e:
        print(f"[Snowflake] Query failed: {e}")
        print(f"[Snowflake] SQL: {sql[:200]}")
        raise
    finally:
        if cur is not None:
            cur.close()


def run_query_as_dicts(
    sql: str,
    conn: snowflake.connector.SnowflakeConnection,
    params: Optional[tuple] = None,
) -> list[dict]:
    """
    Same as run_query() but returns list of dicts instead of DataFrame.
    Useful for the chatbot/API response path.
    """
    df = run_query(sql, conn, params)
    return df.to_dict(orient="records")


# ── Cortex LLM calls ──────────────────────────────────────────────────────────
def cortex_complete(
    prompt: str,
    conn: snowflake.connector.SnowflakeConnection,
    model: str = MODEL_GENERATE,
    max_chars: int = 12000,
) -> str:
    """
    Call Snowflake Cortex COMPLETE with the given model.

    For generation:  model=MODEL_GENERATE  (mistral-large2)
    For validation:  model=MODEL_VALIDATE  (claude-sonnet-4-6)

    Never pass the same model for both generation and validation
    on the same task — cross-model validation is intentional.
    """
    cur = None
    try:
        safe = prompt.replace("'", "\\'")[:max_chars]
        cur  = conn.cursor()
        cur.execute(
            f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model}', '{safe}')"
        )
        row = cur.fetchone()
        return row[0].strip() if row and row[0] else ""
    except Exception as e:
        print(f"[Cortex] {model} call failed: {e}")
        return ""
    finally:
        if cur is not None:
            cur.close()


def cortex_embed(
    text: str,
    conn: snowflake.connector.SnowflakeConnection,
    prefix: str = "query: ",
) -> Optional[list]:
    """
    Generate an E5 embedding vector via Cortex.
    prefix: "query: " for search, "passage: " for ingestion.
    """
    cur = None
    try:
        safe = (prefix + text).replace("'", "\\'")[:2000]
        cur  = conn.cursor()
        cur.execute(
            f"SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768('{MODEL_EMBED}', '{safe}')"
        )
        row = cur.fetchone()
        return row[0] if row else None
    except Exception as e:
        print(f"[Cortex] Embed failed: {e}")
        return None
    finally:
        if cur is not None:
            cur.close()


def rag_search(
    query: str,
    conn: snowflake.connector.SnowflakeConnection,
    domain_filter: Optional[str] = None,
    top_k: int = 3,
    min_similarity: float = 0.60,
) -> list[dict]:
    """
    Vector similarity search against RAW_DOMAIN_CHUNKS.
    Returns top_k chunks above min_similarity threshold.

    domain_filter: "CRIME" | "TRANSIT" | "SCHOOLS" | etc. — None = search all.
    """
    safe_q       = query.replace("'", "\\'")[:2000]
    domain_clause = (
        f"AND UPPER(DOMAIN) = '{domain_filter.upper()}'"
        if domain_filter and domain_filter.upper() != "ALL"
        else ""
    )
    sql = f"""
        SELECT
            CHUNK_TEXT,
            DOMAIN,
            SOURCE_FILE,
            VECTOR_COSINE_SIMILARITY(
                CHUNK_EMBEDDING,
                SNOWFLAKE.CORTEX.EMBED_TEXT_768('{MODEL_EMBED}', 'query: {safe_q}')
            ) AS SIMILARITY
        FROM NEIGHBOURWISE_DOMAINS.RAW_UNSTRUCTURED.RAW_DOMAIN_CHUNKS
        WHERE 1=1 {domain_clause}
        ORDER BY SIMILARITY DESC
        LIMIT {top_k}
    """
    cur = None
    try:
        cur  = conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        return [r for r in rows if float(r.get("SIMILARITY", 0)) >= min_similarity]
    except Exception as e:
        print(f"[RAG] Search failed: {e}")
        return []
    finally:
        if cur is not None:
            cur.close()


# ── Neighborhood lookup helpers ────────────────────────────────────────────────
def get_all_neighborhoods(
    conn: snowflake.connector.SnowflakeConnection,
) -> list[str]:
    """Return all 51 neighborhood names from MASTER_LOCATION."""
    df = run_query(
        "SELECT NEIGHBORHOOD_NAME FROM MARTS.MASTER_LOCATION ORDER BY NEIGHBORHOOD_NAME",
        conn,
    )
    return df["NEIGHBORHOOD_NAME"].tolist() if not df.empty else []


def get_neighborhood_city(
    neighborhood: str,
    conn: snowflake.connector.SnowflakeConnection,
) -> Optional[str]:
    """Return the CITY for a given neighborhood name."""
    df = run_query(
        "SELECT CITY FROM MARTS.MASTER_LOCATION "
        "WHERE UPPER(NEIGHBORHOOD_NAME) = %s LIMIT 1",
        conn,
        (neighborhood.strip().upper(),),
    )
    return str(df.iloc[0]["CITY"]).title() if not df.empty else None


def validate_neighborhood_exists(
    neighborhood: str,
    conn: snowflake.connector.SnowflakeConnection,
) -> tuple[bool, Optional[str]]:
    """
    Check if neighborhood exists in MASTER_LOCATION.
    Returns (exists: bool, city: Optional[str])
    """
    df = run_query(
        """
        SELECT NEIGHBORHOOD_NAME, CITY
        FROM MARTS.MASTER_LOCATION
        WHERE UPPER(NEIGHBORHOOD_NAME) = %s
        LIMIT 1
        """,
        conn,
        (neighborhood.strip().upper(),),
    )
    if df.empty:
        return False, None
    return True, str(df.iloc[0]["CITY"]).title()